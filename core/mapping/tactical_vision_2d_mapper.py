"""
TacticalVision AI — Phase 9: 2D Field Mapping

Projects player/ball detections from image space onto a 2D minimap
using PnLCalib per-frame camera calibration with multi-layer temporal
stabilisation.

Architecture (two-pass offline):
  Pass 1 — Calibrate: Run PnLCalib on every frame, collect raw camera
    parameters (R, K, t) with metadata (reprojection error, source).
  Smoothing — Bidirectional SO(3) Lie algebra ESKF-Lite:
    Forward pass:  velocity-predictive EMA with adaptive α.
    Backward pass: same algorithm in reverse.
    Merge: geodesic midpoint interpolation on SO(3).
  Pass 2 — Render: Re-read images, project using smoothed H_inv.

Key stabilisation features:
  1. Velocity prediction: angular velocity ω is EMA-tracked and used to
     predict R_pred = R_smooth · exp(ω).  Outlier gate compares against
     R_pred (not R_smooth), so fast paneos are accepted correctly.
  2. Adaptive α: EMA weight modulated by measurement confidence
     (reprojection error, residual magnitude, calibration source).
  3. Bidirectional: forward+backward pass eliminates causal lag.
  4. Dual-path calibration: falls back to ground-plane homography
     (heuristic_voting_ground) when full 3D calibration fails.

Mathematical reference:
  Sola et al., 'A micro Lie theory for state estimation in robotics'
  (arXiv:1812.01537).  BroadTrack (WACV 2025) for broadcast camera
  tracking principles.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import json
import os
import glob
import argparse
import csv
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from vendor.pnlcalib.model import cls_hrnet, cls_hrnet_l
from vendor.pnlcalib.utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    coords_to_dict,
    complete_keypoints
)
from vendor.pnlcalib.utils.utils_calib import (
    FramebyFrameCalib,
    rotation_matrix_to_pan_tilt_roll
)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def build_P_from_cam_params(cam_params):
    """
    Build the 3x4 projection matrix P from a PnLCalib cam_params dict.
    Identical to `projection_from_cam_params` in the original repo.
    """
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)
    return P


def ground_homography_from_P(P):
    """
    Extract the 3x3 ground-plane (Z=0) homography from a 3x4 projection matrix.
    H = [P[:,0] | P[:,1] | P[:,3]].  Returns H_inv (image -> world).
    """
    H = P[:, [0, 1, 3]]
    H_inv = np.linalg.inv(H)
    return H_inv


# ---------------------------------------------------------------------------
# BidirectionalLieSmoother — predict → correct → merge (forward+backward)
# ---------------------------------------------------------------------------

class BidirectionalLieSmoother:
    """
    Offline bidirectional camera stabiliser using SO(3) Lie algebra with
    velocity-predictive ESKF-Lite and forward-backward smoothing.

    This smoother operates in three phases:
      Phase 1 (Collection): collect_measurement() stores raw PnLCalib
        outputs per frame with metadata (reprojection error, source).
      Phase 2 (Smoothing): smooth_all() runs a forward pass with
        velocity-predictive EMA, then a backward pass, and merges both
        via geodesic midpoint interpolation on SO(3).  This eliminates
        the causal lag inherent in forward-only filtering.
      Phase 3 (Retrieval): get_H_inv(frame_idx) returns the smoothed
        homography for any frame.

    Key improvements over the previous CameraParamsSmoother:
      1. PREDICTION: Angular velocity ω is EMA-tracked and used to predict
         R_pred = R_smooth · exp(ω).  The outlier gate compares measurements
         against R_pred (not R_smooth), so fast paneos are accepted correctly.
      2. ADAPTIVE α: The EMA weight is modulated by measurement confidence
         (reprojection error, residual magnitude, calibration source).
      3. BIDIRECTIONAL: A backward pass produces a second smoothed sequence
         that is merged with the forward pass via geodesic interpolation,
         completely eliminating directional lag.

    Mathematical reference:
      Sola et al., 'A micro Lie theory for state estimation in robotics'
      (arXiv:1812.01537).  BroadTrack (WACV 2025) for broadcast camera
      tracking principles.
    """

    # ── Base EMA weights (per-parameter family) ──
    ALPHA_ROTATION = 0.30    # SO(3) rotation — base before confidence scaling
    ALPHA_FOCAL    = 0.05    # fx, fy — nearly constant in broadcast
    ALPHA_POS      = 0.10    # camera position — nearly constant (tripod)
    BETA_OMEGA     = 0.15    # angular velocity EMA — very smooth

    # ── Adaptive α bounds ──
    ALPHA_ROT_MIN  = 0.08    # minimum rotation α (very noisy measurement)
    ALPHA_ROT_MAX  = 0.50    # maximum rotation α (high-confidence measurement)

    # ── Outlier detection thresholds ──
    MAX_ROTATION_CHANGE_DEG = 7.0    # gate vs prediction (wider than before)
    MAX_FOCAL_RATIO         = 0.20   # 20% relative change
    MAX_POS_CHANGE_M        = 20.0   # metres
    MAX_REPROJ_ERR_PX       = 20.0   # pixels (PnLCalib's own metric)

    # ── Recovery ──
    MAX_CONSECUTIVE_REJECTS = 15
    WARMUP_FRAMES           = 3

    # ── Bidirectional merge weight ──
    # 0.5 = equal forward/backward.  Slightly favour forward for causality.
    BIDIR_FORWARD_WEIGHT    = 0.5

    # ── Gap filling ──
    # Los huecos sin calibracion se rellenan sosteniendo/prediciendo estado,
    # pero SOLO hasta esta longitud (frames). Huecos mas largos (p.ej. 238
    # frames en SNMOT-148) devolvian una camara congelada/promediada que
    # proyectaba posiciones FICTICIAS en minimapa y tracking_2d.csv durante
    # segundos; ahora esos frames quedan sin H_inv (visible=0, honesto).
    MAX_GAP_FILL_FRAMES     = 25   # ~1 s a 25 fps

    def __init__(self):
        # Raw measurements collected during Phase 1
        self._measurements = []    # list of dicts or None (per frame)

        # Smoothed output (populated by smooth_all)
        self._smoothed_H_inv = []  # list of np.ndarray or None
        self._smoothed_params = [] # list of dict with R, fx, fy, cx, cy, pos

        self.stats = {
            'accepted_fwd': 0,
            'rejected_outlier_fwd': 0,
            'rejected_reproj_fwd': 0,
            'force_accepted_fwd': 0,
            'accepted_bwd': 0,
            'rejected_bwd': 0,
            'total_frames': 0,
            'fallback': 0,
        }

    # ──────────────────────────────────────────────────────────────
    #  Phase 1: Collect raw measurements
    # ──────────────────────────────────────────────────────────────

    def collect_measurement(self, cam_params, rep_err=0.0, source='voting'):
        """
        Store a raw PnLCalib measurement for later batch smoothing.
        Call with cam_params=None if calibration failed for this frame.
        """
        if cam_params is not None:
            R_mat = np.array(cam_params['rotation_matrix'])
            pos = np.array(cam_params['position_meters'],
                           dtype=np.float64).flatten()
            self._measurements.append({
                'R': Rotation.from_matrix(R_mat),
                'fx': cam_params['x_focal_length'],
                'fy': cam_params['y_focal_length'],
                'cx': cam_params['principal_point'][0],
                'cy': cam_params['principal_point'][1],
                'pos': pos,
                'rep_err': rep_err,
                'source': source,
            })
        else:
            self._measurements.append(None)
            self.stats['fallback'] += 1

    # ──────────────────────────────────────────────────────────────
    #  Phase 2: Bidirectional smoothing
    # ──────────────────────────────────────────────────────────────

    def smooth_all(self):
        """
        Run forward + backward smoothing passes and merge results.
        Must be called after all measurements are collected.

        Forward pass: velocity-predictive EMA (ESKF-Lite) — captures
          camera motion inertia, reduces lag during paneos.
        Backward pass: pure SO(3) EMA WITHOUT velocity prediction —
          velocity prediction is physically invalid in reverse time and
          causes oscillation when merged with the forward pass.
        Merge: geodesic midpoint on SO(3) + linear average for Euclidean.
        """
        N = len(self._measurements)
        self.stats['total_frames'] = N

        # Forward pass — with velocity prediction
        fwd_params = self._run_pass(
            self._measurements, direction='forward', use_prediction=True)

        # Backward pass — pure EMA, NO velocity prediction
        bwd_params = self._run_pass(
            list(reversed(self._measurements)),
            direction='backward', use_prediction=False)
        bwd_params = list(reversed(bwd_params))

        # Merge forward + backward
        w = self.BIDIR_FORWARD_WEIGHT
        merged_params = []
        for i in range(N):
            f = fwd_params[i]
            b = bwd_params[i]

            if f is None and b is None:
                merged_params.append(None)
            elif f is None:
                merged_params.append(b)
            elif b is None:
                merged_params.append(f)
            else:
                # Geodesic interpolation on SO(3) for rotation
                R_fwd = f['R']
                R_bwd = b['R']
                R_delta = R_fwd.inv() * R_bwd
                omega_delta = R_delta.as_rotvec()
                # Weighted geodesic: move (1-w) of the way from fwd to bwd
                R_merged = R_fwd * Rotation.from_rotvec((1 - w) * omega_delta)

                # Linear interpolation for Euclidean parameters
                merged = {
                    'R':  R_merged,
                    'fx': w * f['fx'] + (1 - w) * b['fx'],
                    'fy': w * f['fy'] + (1 - w) * b['fy'],
                    'cx': w * f['cx'] + (1 - w) * b['cx'],
                    'cy': w * f['cy'] + (1 - w) * b['cy'],
                    'pos': w * f['pos'] + (1 - w) * b['pos'],
                }
                merged_params.append(merged)

        # ── Invalidate long measurement gaps ──
        # merged_params rellena huecos con estado sostenido; para huecos
        # mas largos que MAX_GAP_FILL_FRAMES eso fabrica posiciones.
        i = 0
        while i < N:
            if self._measurements[i] is None:
                j = i
                while j < N and self._measurements[j] is None:
                    j += 1
                if (j - i) > self.MAX_GAP_FILL_FRAMES:
                    for k in range(i, j):
                        merged_params[k] = None
                    self.stats['gap_invalidated'] = (
                        self.stats.get('gap_invalidated', 0) + (j - i))
                i = j
            else:
                i += 1

        # Build H_inv for each frame
        self._smoothed_params = merged_params
        self._smoothed_H_inv = []
        for p in merged_params:
            if p is not None:
                self._smoothed_H_inv.append(
                    self._build_H_inv_static(
                        p['R'], p['fx'], p['fy'], p['cx'], p['cy'], p['pos']))
            else:
                self._smoothed_H_inv.append(None)

    def _compute_confidence(self, rep_err, source, residual_norm_deg):
        """
        Compute a [0, 1] confidence score for the measurement.

        High confidence = low rep_err + 'voting' source + small residual.
        This modulates α: confident measurements get higher α (trust more).
        """
        # Reprojection error contribution: decays from 1.0 at err=0 to ~0.2 at err=15
        c_reproj = np.exp(-rep_err / 8.0)

        # Source contribution: voting is more reliable than ground-plane
        c_source = 1.0 if source == 'voting' else 0.7

        # Residual contribution: small residual = consistent with prediction
        c_residual = np.exp(-residual_norm_deg / 4.0)

        return np.clip(c_reproj * c_source * c_residual, 0.15, 1.0)

    def _run_pass(self, measurements, direction='forward',
                  use_prediction=True):
        """
        Single-direction smoothing pass.

        When use_prediction=True (forward pass):
          1. PREDICT: R_pred = R_smooth · exp(ω_smooth)
          2. GATE: compare measurement vs R_pred
          3. CORRECT: EMA in so(3) tangent space with adaptive α
          4. UPDATE VELOCITY: ω_smooth via EMA

        When use_prediction=False (backward pass):
          Pure EMA on SO(3) — like the original CameraParamsSmoother.
          Gate compares against R_smooth directly. No velocity state.
          This avoids the oscillation caused by conflicting velocity
          predictions between forward and backward passes.
        """
        N = len(measurements)
        result = [None] * N

        # State
        R_smooth = None
        fx_s, fy_s, cx_s, cy_s = None, None, None, None
        pos_s = None
        omega_smooth = np.zeros(3)      # angular velocity in rad/frame
        reject_count = 0
        frame_count = 0

        # Velocity decay rate during rejections
        OMEGA_REJECT_DECAY = 0.80   # aggressive decay to prevent cascade drift

        tag_accept = f'accepted_{direction[:3]}'
        tag_reject_out = f'rejected_outlier_{direction[:3]}'
        tag_reject_rep = f'rejected_reproj_{direction[:3]}'
        tag_force = f'force_accepted_{direction[:3]}'

        # Ensure stat keys exist for backward pass
        for k in [tag_accept, tag_reject_out, tag_reject_rep, tag_force]:
            if k not in self.stats:
                self.stats[k] = 0

        for i in range(N):
            m = measurements[i]
            frame_count += 1

            if m is None:
                # No measurement — hold state (optionally predict forward)
                if R_smooth is not None:
                    if use_prediction:
                        R_smooth = R_smooth * Rotation.from_rotvec(omega_smooth)
                        omega_smooth *= OMEGA_REJECT_DECAY
                    result[i] = {
                        'R': R_smooth, 'fx': fx_s, 'fy': fy_s,
                        'cx': cx_s, 'cy': cy_s, 'pos': pos_s.copy(),
                    }
                continue

            R_new = m['R']
            fx, fy = m['fx'], m['fy']
            cx, cy = m['cx'], m['cy']
            pos = m['pos']
            rep_err = m['rep_err']
            source = m['source']

            # ── Warm-up: accept directly ──
            if frame_count <= self.WARMUP_FRAMES or R_smooth is None:
                R_smooth = R_new
                fx_s, fy_s = fx, fy
                cx_s, cy_s = cx, cy
                pos_s = pos.copy()
                omega_smooth = np.zeros(3)
                self.stats[tag_accept] += 1
                result[i] = {
                    'R': R_smooth, 'fx': fx_s, 'fy': fy_s,
                    'cx': cx_s, 'cy': cy_s, 'pos': pos_s.copy(),
                }
                continue

            # ── Determine reference for gating ──
            if use_prediction:
                R_ref = R_smooth * Rotation.from_rotvec(omega_smooth)
            else:
                R_ref = R_smooth  # pure EMA: compare against smooth state

            # Reprojection error pre-check
            if rep_err > self.MAX_REPROJ_ERR_PX:
                reject_count += 1
                self.stats[tag_reject_rep] += 1
                if use_prediction:
                    R_smooth = R_ref  # advance to prediction
                    omega_smooth *= OMEGA_REJECT_DECAY
                # else: hold R_smooth as-is
                result[i] = {
                    'R': R_smooth, 'fx': fx_s, 'fy': fy_s,
                    'cx': cx_s, 'cy': cy_s, 'pos': pos_s.copy(),
                }
                continue

            # ── GATE phase: compare against R_ref ──
            R_delta_ref = R_ref.inv() * R_new
            residual = R_delta_ref.as_rotvec()
            residual_norm_deg = np.rad2deg(np.linalg.norm(residual))

            # Focal and position checks (against smoothed state)
            focal_ok = True
            if fx_s > 0:
                focal_ok = (abs(fx - fx_s) / fx_s <= self.MAX_FOCAL_RATIO)
            pos_ok = (np.linalg.norm(pos - pos_s) <= self.MAX_POS_CHANGE_M)

            # Use tighter gate for backward (no prediction to absorb motion)
            max_rot = self.MAX_ROTATION_CHANGE_DEG
            if not use_prediction:
                max_rot = 5.0  # tighter gate for pure EMA backward

            rotation_ok = (residual_norm_deg <= max_rot)

            if rotation_ok and focal_ok and pos_ok:
                # ── ACCEPTED ──
                confidence = self._compute_confidence(
                    rep_err, source, residual_norm_deg)
                alpha_r = np.clip(
                    self.ALPHA_ROTATION * confidence,
                    self.ALPHA_ROT_MIN, self.ALPHA_ROT_MAX)
                alpha_f = self.ALPHA_FOCAL
                alpha_p = self.ALPHA_POS
                reject_count = 0
                self.stats[tag_accept] += 1

            elif reject_count >= self.MAX_CONSECUTIVE_REJECTS:
                # ── FORCE ACCEPT ──
                alpha_r = 0.8
                alpha_f = 0.5
                alpha_p = 0.8
                reject_count = 0
                self.stats[tag_force] += 1
                # Recompute residual against smooth (not pred) for force
                R_delta_ref = R_smooth.inv() * R_new
                residual = R_delta_ref.as_rotvec()

            else:
                # ── REJECTED — hold or predict ──
                reject_count += 1
                if not rotation_ok:
                    self.stats[tag_reject_out] += 1
                else:
                    self.stats[tag_reject_rep] += 1
                if use_prediction:
                    R_smooth = R_ref  # advance to prediction
                    omega_smooth *= OMEGA_REJECT_DECAY
                # else: hold R_smooth as-is
                result[i] = {
                    'R': R_smooth, 'fx': fx_s, 'fy': fy_s,
                    'cx': cx_s, 'cy': cy_s, 'pos': pos_s.copy(),
                }
                continue

            # ── CORRECTION phase ──
            R_prev = R_smooth

            # EMA correction on SO(3) via so(3) tangent space
            omega_corr = alpha_r * residual
            R_smooth = R_ref * Rotation.from_rotvec(omega_corr)

            # ── UPDATE VELOCITY (only for forward predictive pass) ──
            if use_prediction:
                omega_raw = (R_prev.inv() * R_smooth).as_rotvec()
                omega_smooth = (self.BETA_OMEGA * omega_raw +
                                (1 - self.BETA_OMEGA) * omega_smooth)

            # ── Standard EMA for Euclidean parameters ──
            fx_s = alpha_f * fx + (1 - alpha_f) * fx_s
            fy_s = alpha_f * fy + (1 - alpha_f) * fy_s
            cx_s = alpha_f * cx + (1 - alpha_f) * cx_s
            cy_s = alpha_f * cy + (1 - alpha_f) * cy_s
            pos_s = alpha_p * pos + (1 - alpha_p) * pos_s

            result[i] = {
                'R': R_smooth, 'fx': fx_s, 'fy': fy_s,
                'cx': cx_s, 'cy': cy_s, 'pos': pos_s.copy(),
            }

        return result

    # ──────────────────────────────────────────────────────────────
    #  Phase 3: Retrieval
    # ──────────────────────────────────────────────────────────────

    def get_H_inv(self, frame_idx):
        """Get the smoothed H_inv for a specific frame index."""
        if frame_idx < len(self._smoothed_H_inv):
            return self._smoothed_H_inv[frame_idx]
        return None

    def get_smoothed_rotation(self, frame_idx):
        """Get the smoothed Rotation for a specific frame (for diagnostics)."""
        if frame_idx < len(self._smoothed_params):
            p = self._smoothed_params[frame_idx]
            if p is not None:
                return p['R']
        return None

    def get_smoothed_params(self, frame_idx):
        """Get all smoothed params for a specific frame (for diagnostics)."""
        if frame_idx < len(self._smoothed_params):
            return self._smoothed_params[frame_idx]
        return None

    @staticmethod
    def _build_H_inv_static(R_rot, fx, fy, cx, cy, pos):
        """
        Reconstruct H_inv from camera parameters.
        Static version for use in batch processing.
        """
        rotation = R_rot.as_matrix()

        Q = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ])

        It = np.eye(4)[:-1]
        It[:, -1] = -pos

        P = Q @ (rotation @ It)
        H = P[:, [0, 1, 3]]   # ground plane (Z=0)
        try:
            H_inv = np.linalg.inv(H)
            return H_inv
        except np.linalg.LinAlgError:
            return None

    def summary(self):
        s = self.stats
        return (f"  Forward  — accepted: {s.get('accepted_fwd',0)}, "
                f"rejected(outlier): {s.get('rejected_outlier_fwd',0)}, "
                f"rejected(reproj): {s.get('rejected_reproj_fwd',0)}, "
                f"force: {s.get('force_accepted_fwd',0)}\n"
                f"  Backward — accepted: {s.get('accepted_bwd',0)}, "
                f"rejected: {s.get('rejected_bwd',0)}\n"
                f"  Fallback (no calib): {s.get('fallback',0)}\n"
                f"  Gap frames invalidated (>{self.MAX_GAP_FILL_FRAMES}f): "
                f"{s.get('gap_invalidated',0)}\n"
                f"  Total frames: {s.get('total_frames',0)}")


# ---------------------------------------------------------------------------
# RobustOfflineSmoother — rechazo robusto + gauss fase-cero (produccion)
# ---------------------------------------------------------------------------

class RobustOfflineSmoother(BidirectionalLieSmoother):
    """
    Suavizador OFFLINE robusto (auditoria 2026-07-05, informe de
    estabilizacion del minimapa): el mapper ya es de dos pasadas, asi que
    un suavizado batch de fase cero es legitimo y domina al EMA causal
    en tramos continuos (SNMOT-148: desv mediana 0.25 vs 0.82 m, jitter
    0.014 vs 0.048 m/f^2, p95 3.4 vs 7.8 m vs el Lie bidireccional).

      1. Rechazo por rep_err y por proyeccion insana (puntos de imagen
         proyectados fuera de un entorno del campo — atrapa las soluciones
         espejadas de PnLCalib que llegan con rep_err aceptable).
      2. Rechazo por residuo vs mediana deslizante del rotvec.
      3. Interpolacion lineal + filtro gaussiano de FASE CERO sobre
         rotvec / focal / principal point / posicion.
      4. Huecos de medicion > MAX_GAP_FILL_FRAMES quedan sin H_inv
         (misma regla heredada: no fabricar posiciones).

    Reproduce `run_variant_gauss_robust` de scripts/analyze_lie_smoothing.py.
    """

    GAUSS_SIGMA      = 3.0    # frames (fase cero — sin retardo direccional)
    MED_WIN          = 11     # ventana de la mediana deslizante
    MED_REJECT_DEG   = 4.0    # residuo maximo vs mediana (grados)
    SANE_HALF_X_M    = 120.0  # sanidad de proyeccion (campo 105x68 + margen)
    SANE_HALF_Y_M    = 80.0

    def __init__(self, image_size=(1920, 1080)):
        super().__init__()
        w, h = image_size
        # puntos de imagen para el test de sanidad (mitad inferior ~ campo)
        self._sane_pts = np.array([
            [0.50 * w, 0.907 * h, 1.0],
            [0.25 * w, 0.759 * h, 1.0],
            [0.75 * w, 0.759 * h, 1.0],
        ], dtype=np.float64).T

    def _measurement_is_sane(self, m):
        H = self._build_H_inv_static(m["R"], m["fx"], m["fy"],
                                     m["cx"], m["cy"], m["pos"])
        if H is None:
            return False
        w = H @ self._sane_pts
        p = (w[:2] / w[2:3]).T
        return bool((np.abs(p[:, 0]) <= self.SANE_HALF_X_M).all()
                    and (np.abs(p[:, 1]) <= self.SANE_HALF_Y_M).all())

    def smooth_all(self):
        from scipy.ndimage import gaussian_filter1d, median_filter
        ms = self._measurements
        N = len(ms)
        self.stats['total_frames'] = N

        ok = []
        for i, m in enumerate(ms):
            if m is None:
                continue
            if m["rep_err"] > self.MAX_REPROJ_ERR_PX:
                self.stats['rejected_reproj'] = (
                    self.stats.get('rejected_reproj', 0) + 1)
                continue
            if not self._measurement_is_sane(m):
                self.stats['rejected_insane'] = (
                    self.stats.get('rejected_insane', 0) + 1)
                continue
            ok.append(i)

        self._smoothed_params = [None] * N
        self._smoothed_H_inv = [None] * N
        if len(ok) < 5:
            return

        ref = ms[ok[0]]["R"]
        rotvecs = np.full((N, 3), np.nan)
        scal = np.full((N, 7), np.nan)   # fx fy cx cy pos(3)
        for i in ok:
            m = ms[i]
            rotvecs[i] = (ref.inv() * m["R"]).as_rotvec()
            scal[i] = [m["fx"], m["fy"], m["cx"], m["cy"], *m["pos"]]

        # rechazo por mediana deslizante (outliers espejados/aislados)
        rv_ok = rotvecs[ok]
        med = np.stack([median_filter(rv_ok[:, c], size=self.MED_WIN,
                                      mode="nearest") for c in range(3)],
                       axis=1)
        resid_deg = np.rad2deg(np.linalg.norm(rv_ok - med, axis=1))
        keep = resid_deg <= self.MED_REJECT_DEG
        self.stats['rejected_median'] = int((~keep).sum())
        ok = [i for i, k in zip(ok, keep) if k]
        if len(ok) < 5:
            return
        drop = ~np.isin(np.arange(N), ok)
        rotvecs[drop] = np.nan
        scal[drop] = np.nan
        self.stats['accepted'] = len(ok)

        # interpolar + gauss fase cero
        idx = np.arange(N)
        good = ~np.isnan(rotvecs[:, 0])
        for arr in (rotvecs, scal):
            for c in range(arr.shape[1]):
                col = arr[:, c]
                g = ~np.isnan(col)
                arr[:, c] = np.interp(idx, idx[g], col[g])
        rotvecs = gaussian_filter1d(rotvecs, self.GAUSS_SIGMA, axis=0,
                                    mode="nearest")
        scal = gaussian_filter1d(scal, self.GAUSS_SIGMA, axis=0,
                                 mode="nearest")

        params = [{
            "R": ref * Rotation.from_rotvec(rotvecs[i]),
            "fx": scal[i, 0], "fy": scal[i, 1],
            "cx": scal[i, 2], "cy": scal[i, 3], "pos": scal[i, 4:7]}
            for i in range(N)]

        # huecos largos entre mediciones ACEPTADAS -> sin H_inv
        i = 0
        while i < N:
            if not good[i]:
                j = i
                while j < N and not good[j]:
                    j += 1
                if (j - i) > self.MAX_GAP_FILL_FRAMES:
                    for k in range(i, j):
                        params[k] = None
                    self.stats['gap_invalidated'] = (
                        self.stats.get('gap_invalidated', 0) + (j - i))
                i = j
            else:
                i += 1

        self._smoothed_params = params
        self._smoothed_H_inv = [
            None if p is None else self._build_H_inv_static(
                p["R"], p["fx"], p["fy"], p["cx"], p["cy"], p["pos"])
            for p in params]

    def summary(self):
        s = self.stats
        return (f"  Accepted: {s.get('accepted',0)} | "
                f"rejected reproj: {s.get('rejected_reproj',0)}, "
                f"insane: {s.get('rejected_insane',0)}, "
                f"median: {s.get('rejected_median',0)}\n"
                f"  Fallback (no calib): {s.get('fallback',0)}\n"
                f"  Gap frames invalidated (>{self.MAX_GAP_FILL_FRAMES}f): "
                f"{s.get('gap_invalidated',0)}\n"
                f"  Total frames: {s.get('total_frames',0)}")


# ---------------------------------------------------------------------------
# Minimap drawing
# ---------------------------------------------------------------------------

def draw_pitch_cv2(scale=10, margin=50):
    """Draw a 2D pitch minimap using OpenCV."""
    w = int(105 * scale)
    h = int(68 * scale)
    img_w, img_h = w + 2 * margin, h + 2 * margin

    pitch = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    pitch[:] = (60, 120, 50)  # BGR green

    white = (255, 255, 255)
    thickness = 2

    pt = lambda x, y: (int(x * scale) + margin, int(y * scale) + margin)

    # Field outline
    cv2.rectangle(pitch, pt(0, 0), pt(105, 68), white, thickness)
    # Half-way line
    cv2.line(pitch, pt(52.5, 0), pt(52.5, 68), white, thickness)
    # Centre circle
    cv2.circle(pitch, pt(52.5, 34), int(9.15 * scale), white, thickness)
    cv2.circle(pitch, pt(52.5, 34), 2, white, -1)
    # Goal areas
    cv2.rectangle(pitch, pt(0, 24.85), pt(5.5, 43.15), white, thickness)
    cv2.rectangle(pitch, pt(105 - 5.5, 24.85), pt(105, 43.15), white, thickness)
    # Penalty areas
    cv2.rectangle(pitch, pt(0, 13.85), pt(16.5, 54.15), white, thickness)
    cv2.rectangle(pitch, pt(105 - 16.5, 13.85), pt(105, 54.15), white, thickness)
    # Penalty spots
    cv2.circle(pitch, pt(11, 34), 2, white, -1)
    cv2.circle(pitch, pt(105 - 11, 34), 2, white, -1)
    # Penalty arcs
    cv2.ellipse(pitch, pt(11, 34), (int(9.15 * scale), int(9.15 * scale)),
                0, -53.13, 53.13, white, thickness)
    cv2.ellipse(pitch, pt(105 - 11, 34), (int(9.15 * scale), int(9.15 * scale)),
                0, 126.87, 233.13, white, thickness)

    return pitch


# ---------------------------------------------------------------------------
# Projection: image pixel  →  minimap pixel   via  H_inv
# ---------------------------------------------------------------------------

def project_point_to_world(x_img, y_img, H_inv):
    """
    Project image point to world coordinates (metres, top-left origin).
    Returns (x_world, y_world) or (None, None) if out of bounds / invalid.
    """
    pt_world = H_inv @ np.array([x_img, y_img, 1.0])
    if abs(pt_world[2]) < 1e-9:
        return None, None
    pt_world /= pt_world[2]

    x_world = pt_world[0] + 52.5
    y_world = pt_world[1] + 34.0

    if x_world < -5 or x_world > 110 or y_world < -5 or y_world > 73:
        return None, None
    return x_world, y_world


def project_point(x_img, y_img, H_inv, scale, margin, w_pitch, h_pitch):
    """
    Project an image-space point (x_img, y_img) onto the minimap using H_inv.
    H_inv maps image coords → PnLCalib world coords (centred on pitch centre).
    Returns (mx, my) minimap pixel coords, or (None, None) if out of bounds.
    """
    x_world, y_world = project_point_to_world(x_img, y_img, H_inv)
    if x_world is None:
        return None, None

    mx = int(x_world * scale) + margin
    my = int(y_world * scale) + margin

    if 0 <= mx < w_pitch and 0 <= my < h_pitch:
        return mx, my
    return None, None

    mx = int(x_world * scale) + margin
    my = int(y_world * scale) + margin

    if 0 <= mx < w_pitch and 0 <= my < h_pitch:
        return mx, my
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--team_assignments", type=str, default=None,
                        help="Path to team_assignments.json from Phase 8 (clustering)")
    parser.add_argument("--pnlcalib_kp_weights", type=str, default="models/SV_kp")
    parser.add_argument("--pnlcalib_line_weights", type=str, default="models/SV_lines")
    parser.add_argument("--pnlcalib_kp_cfg", type=str,
                        default="vendor/pnlcalib/config/hrnetv2_w48.yaml")
    parser.add_argument("--pnlcalib_l_cfg", type=str,
                        default="vendor/pnlcalib/config/hrnetv2_w48_l.yaml")
    parser.add_argument("--kp_threshold", type=float, default=0.3434)
    parser.add_argument("--line_threshold", type=float, default=0.7867)
    parser.add_argument("--disable_pnl_refine", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame calibration diagnostics")
    parser.add_argument("--output", type=str, default="",
                        help="Path to output video (.mp4)")
    parser.add_argument("--output_csv", type=str, default="",
                        help="Optional: path to export tracking 2D metric CSV")
    parser.add_argument("--output_calibration", type=str, default="",
                        help="Optional: path to export per-frame smoothed H_inv "
                             "(calibration_hinv.json, image -> centred pitch metres)")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video rendering (useful when only CSV is needed)")
    parser.add_argument("--smoothing_mode", type=str, default="robust_offline",
                        choices=["robust_offline", "lie_bidir"],
                        help="robust_offline (default): rechazo robusto + gauss "
                             "fase-cero batch, domina en tramos continuos; "
                             "lie_bidir: ESKF-Lite bidireccional anterior")
    parser.add_argument("--dump_raw_calib", type=str, default="",
                        help="Optional: path to dump RAW per-frame PnLCalib "
                             "measurements (pre-smoothing) as JSON, for offline "
                             "analysis of smoothing variants")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Frames per second of the sequence")
    parser.add_argument("--show_track_ids", action="store_true",
                        help="debug: etiqueta '#<track_id>' en jugadores sin dorsal "
                             "(por defecto NO se muestra para no confundir con dorsales)")
    parser.add_argument("--jersey_json", type=str, default=None,
                        help="Optional jersey identity JSON (Phase 10 output): "
                             "locked/tentative numbers replace track_id labels")
    args = parser.parse_args()

    if not args.no_video and not args.output:
        parser.error("--output is required unless --no_video is set")
    if args.no_video and not args.output_csv:
        parser.error("--output_csv is required when --no_video is set")

    # --- Load HRNet configs ---
    with open(args.pnlcalib_kp_cfg, 'r') as fh:
        cfg = yaml.safe_load(fh)
    with open(args.pnlcalib_l_cfg, 'r') as fh:
        cfgl = yaml.safe_load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load HRNet models ---
    print("Loading PnLCalib Keypoints HRNet...")
    model_kp = cls_hrnet.get_cls_net(cfg)
    model_kp.load_state_dict(
        torch.load(args.pnlcalib_kp_weights, map_location=device, weights_only=False))
    model_kp.to(device).eval()

    print("Loading PnLCalib Lines HRNet...")
    model_l = cls_hrnet_l.get_cls_net(cfgl)
    model_l.load_state_dict(
        torch.load(args.pnlcalib_line_weights, map_location=device, weights_only=False))
    model_l.to(device).eval()

    transform_resize = T.Resize((540, 960))

    # --- Load tracking data ---
    print("Loading tracking data...")
    with open(args.detections, "r") as fh:
        detections = {d["frame_id"]: d for d in json.load(fh)}
    with open(args.trajectory, "r") as fh:
        trajectory = {t["frame_id"]: t for t in json.load(fh)}

    # --- Load team clustering data (if available) ---
    team_map = {}   # track_id -> {"team_id": int, "role": str}
    if args.team_assignments and os.path.exists(args.team_assignments):
        print(f"Loading team assignments from {args.team_assignments}...")
        with open(args.team_assignments, "r") as fh:
            for entry in json.load(fh):
                tid = entry["track_id"]
                team_map[tid] = {
                    "team_id": entry.get("team_id", -1),
                    "role": entry.get("role", "player")
                }
        print(f"  {len(team_map)} tracks with team assignment.")
    else:
        print("No clustering data — players rendered without team distinction.")

    # --- Load jersey identities (if available) ---
    jersey_map = {}  # track_id -> {"number": int, "state": str}
    if args.jersey_json and os.path.exists(args.jersey_json):
        print(f"Loading jersey identities from {args.jersey_json}...")
        with open(args.jersey_json, "r") as fh:
            jersey_data = json.load(fh)
        for t in jersey_data.get("tracklets", []):
            num = t.get("predicted_number")
            state = t.get("state", "unknown")
            if num is not None and state in ("locked", "tentative"):
                jersey_map[t["track_id"]] = {"number": int(num), "state": state}
        n_locked = sum(1 for v in jersey_map.values() if v["state"] == "locked")
        print(f"  {len(jersey_map)} tracks with jersey number "
              f"({n_locked} locked, {len(jersey_map) - n_locked} tentative).")

    def get_display_label(track_id):
        """Jersey number if known (tentative marked with '?').

        Sin dorsal identificado NO se muestra numero (evita leer el track_id
        como si fuera un dorsal imposible, p.ej. '#432'); con
        --show_track_ids se recupera la etiqueta de debug '#<id>'."""
        jinfo = jersey_map.get(track_id)
        if jinfo is None:
            return (f"#{track_id}", False) if args.show_track_ids else ("", False)
        suffix = "" if jinfo["state"] == "locked" else "?"
        return f"{jinfo['number']}{suffix}", True

    # Team color palette (BGR)
    TEAM_COLORS = {
        0:  (255, 100, 50),   # Team A – orange-blue
        1:  (50, 255, 100),   # Team B – green
        -1: (180, 180, 180),  # Outlier – grey
        -2: (0, 255, 255),    # Referee – yellow
    }

    def get_player_color(track_id):
        """Return (color_bgr, is_gk, is_ref) for a track_id."""
        info = team_map.get(track_id)
        if info is None:
            return (255, 0, 0), False, False  # fallback blue
        role = info["role"]
        team_id = info["team_id"]
        if role == "referee":
            return (0, 255, 255), False, True
        color = TEAM_COLORS.get(team_id, (180, 180, 180))
        return color, (role == "goalkeeper"), False

    img_dir = os.path.join(args.sequence_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        print("Error: No images found in sequence directory.")
        return

    frame_0 = cv2.imread(images[0])
    h_ori, w_ori = frame_0.shape[:2]

    # --- Calibration objects ---
    cam = FramebyFrameCalib(iwidth=w_ori, iheight=h_ori, denormalize=True)
    if args.smoothing_mode == "robust_offline":
        smoother = RobustOfflineSmoother(image_size=(w_ori, h_ori))
    else:
        smoother = BidirectionalLieSmoother()

    pnl_refine = not args.disable_pnl_refine

    calib_voting = 0
    calib_ground = 0
    calib_fail   = 0

    # ═══════════════════════════════════════════════════════════════
    #  PASS 1: PnLCalib inference → collect measurements
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"Pass 1/2: Calibrating {len(images)} frames "
          f"(PnL refine={'ON' if pnl_refine else 'OFF'})...")
    print(f"{'='*50}")

    frame_ids = []   # ordered list of frame_ids matching measurement indices
    raw_calib_dump = []  # raw measurements (only filled with --dump_raw_calib)

    for img_path in tqdm(images, desc="Calibrating"):
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
        frame_ids.append(frame_id)
        frame = cv2.imread(img_path)

        # ── PnLCalib inference ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = TF.to_tensor(pil).float().unsqueeze(0)
        if tensor.size()[-1] != 960:
            tensor = transform_resize(tensor)
        tensor = tensor.to(device)
        _, _, h_t, w_t = tensor.size()

        with torch.no_grad():
            hm_kp = model_kp(tensor)
            hm_l  = model_l(tensor)

        # Exclude background channel (last channel)
        kp_coords   = get_keypoints_from_heatmap_batch_maxpool(hm_kp[:, :-1, :, :])
        line_coords  = get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :])

        kp_dict    = coords_to_dict(kp_coords,   threshold=args.kp_threshold)
        lines_dict = coords_to_dict(line_coords,  threshold=args.line_threshold)

        kp_dict, lines_dict = complete_keypoints(
            kp_dict[0], lines_dict[0], w=w_t, h=h_t, normalize=True
        )

        cam.update(kp_dict, lines_dict)

        # ── Dual-path calibration ──
        cam_params_dict = None
        rep_err = 999.0
        source = 'hold'

        # Path A: Full 3D calibration (heuristic_voting)
        result = cam.heuristic_voting(refine_lines=pnl_refine)

        if result is not None:
            cam_params_dict = result['cam_params']
            rep_err = result['rep_err']
            calib_voting += 1
            source = 'voting'
        else:
            # Path B: Ground-plane homography (heuristic_voting_ground)
            try:
                result_g = cam.heuristic_voting_ground(refine_lines=pnl_refine)
                if (result_g is not None
                        and cam.calibration is not None
                        and cam.rotation is not None
                        and cam.position is not None):
                    pan_r, tilt_r, roll_r = rotation_matrix_to_pan_tilt_roll(
                        cam.rotation)
                    pos_flat = np.array(cam.position).flatten()
                    cam_params_dict = {
                        'pan_degrees':    float(np.rad2deg(pan_r)),
                        'tilt_degrees':   float(np.rad2deg(tilt_r)),
                        'roll_degrees':   float(np.rad2deg(roll_r)),
                        'x_focal_length': float(cam.calibration[0, 0]),
                        'y_focal_length': float(cam.calibration[1, 1]),
                        'principal_point': [float(cam.calibration[0, 2]),
                                           float(cam.calibration[1, 2])],
                        'position_meters': [float(pos_flat[0]),
                                            float(pos_flat[1]),
                                            float(pos_flat[2])],
                        'rotation_matrix': cam.rotation.tolist()
                    }
                    rep_err = result_g.get('rep_err', 10.0)
                    calib_ground += 1
                    source = 'ground'
            except Exception:
                cam_params_dict = None

        if cam_params_dict is None:
            calib_fail += 1

        # ── Collect into smoother ──
        smoother.collect_measurement(cam_params_dict, rep_err, source)
        if args.dump_raw_calib:
            raw_calib_dump.append(
                None if cam_params_dict is None else
                {"frame_id": int(frame_id), "rep_err": float(rep_err),
                 "source": source, **cam_params_dict})

    if args.dump_raw_calib:
        with open(args.dump_raw_calib, "w", encoding="utf-8") as fh:
            json.dump({"frame_ids": [int(f) for f in frame_ids],
                       "measurements": raw_calib_dump}, fh)
        print(f"Raw calibration dumped: {args.dump_raw_calib}")

    # ═══════════════════════════════════════════════════════════════
    #  BIDIRECTIONAL SMOOTHING (forward + backward + merge)
    # ═══════════════════════════════════════════════════════════════
    print(f"\nRunning calibration smoothing ({args.smoothing_mode})...")
    smoother.smooth_all()

    # ── Debug: print a sample of smoothed rotations ──
    if args.debug:
        print("\n  Sample smoothed rotations (every 25 frames):")
        for idx, fid in enumerate(frame_ids):
            if fid % 25 == 0:
                R_s = smoother.get_smoothed_rotation(idx)
                p = smoother.get_smoothed_params(idx)
                if R_s is not None:
                    euler = R_s.as_euler('ZXZ', degrees=True)
                    print(f"  [{fid:>5d}] euler_ZXZ=[{euler[0]:7.1f},"
                          f"{euler[1]:6.1f},{euler[2]:6.1f}]° "
                          f"fx={p['fx']:7.0f}")

    # ── Optional: export smoothed per-frame calibration ──
    if args.output_calibration:
        calib_out = {}
        for idx, fid in enumerate(frame_ids):
            H_inv = smoother.get_H_inv(idx)
            if H_inv is None:
                continue
            calib_out[str(fid)] = {"H_inv": np.asarray(H_inv).tolist(),
                                   "time_s": round((fid - 1) / args.fps, 4)}
        with open(args.output_calibration, "w", encoding="utf-8") as fh:
            json.dump(calib_out, fh)
        print(f"Calibration exported: {args.output_calibration} "
              f"({len(calib_out)}/{len(frame_ids)} frames)")

    # ═══════════════════════════════════════════════════════════════
    #  PASS 2: Render minimap with smoothed H_inv + CSV export
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*50}")
    if args.no_video:
        print(f"Pass 2/2: Processing {len(images)} frames (no video)...")
    else:
        print(f"Pass 2/2: Rendering {len(images)} frames with smoothed calibration...")
    print(f"{'='*50}")

    # --- Video layout ---
    scale = 8
    margin = 40
    base_pitch = draw_pitch_cv2(scale, margin)
    h_pitch, w_pitch = base_pitch.shape[:2]

    target_video_h = max(h_ori, h_pitch)
    scale_ori = target_video_h / h_ori
    new_w_ori = int(w_ori * scale_ori)

    out_w = new_w_ori + w_pitch
    out_h = target_video_h

    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))

    # --- CSV export setup ---
    csv_file = None
    csv_writer = None
    if args.output_csv:
        csv_file = open(args.output_csv, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame_id', 'time_s', 'entity_type', 'track_id', 'team_id', 'role',
            'x_m', 'y_m', 'visible', 'source', 'confidence'
        ])

    dt = 1.0 / args.fps

    for idx, img_path in enumerate(tqdm(images, desc="Processing")):
        frame_id = frame_ids[idx]
        if not args.no_video:
            frame = cv2.imread(img_path)
        else:
            frame = None

        # ── Get pre-computed smoothed H_inv ──
        H_inv = smoother.get_H_inv(idx)

        # ── Render minimap ──
        if not args.no_video:
            pitch_frame = base_pitch.copy()

        if H_inv is not None:
            # -- Players (team-aware) --
            frame_data = detections.get(frame_id, {"players": []})
            for p in frame_data.get("players", []):
                x_min = p["x_min"]
                y_min = p["y_min"]
                x_max = p["x_max"]
                y_max = p["y_max"]
                track_id = p.get("track_id", -1)

                color, is_gk, is_ref = get_player_color(track_id)

                if not args.no_video:
                    # Draw bbox on video frame with team color
                    cv2.rectangle(frame,
                                  (int(x_min), int(y_min)),
                                  (int(x_max), int(y_max)),
                                  color, 2)

                    # Label on video frame (jersey number when identified)
                    info_team = team_map.get(track_id)
                    if info_team:
                        label, has_jersey = get_display_label(track_id)
                        if is_ref:
                            label = f"REF {label}".rstrip()
                        elif is_gk:
                            label = f"GK {label}".rstrip()
                        if label:
                            font_scale = 0.55 if has_jersey else 0.4
                            cv2.putText(frame, label,
                                        (int(x_min), int(y_min) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                                        2 if has_jersey else 1,
                                        cv2.LINE_AA)

                # Metric projection (foot-point)
                x_foot = (x_min + x_max) / 2.0
                y_foot = y_max
                x_world, y_world = project_point_to_world(x_foot, y_foot, H_inv)
                visible = x_world is not None

                if csv_writer:
                    info_team = team_map.get(track_id, {})
                    role_str = info_team.get("role", "unknown")
                    team_id = info_team.get("team_id", -1)
                    csv_writer.writerow([
                        frame_id, round((frame_id - 1) * dt, 3),
                        'player', track_id, team_id, role_str,
                        round(x_world, 3) if visible else '',
                        round(y_world, 3) if visible else '',
                        1 if visible else 0,
                        'detection',
                        ''
                    ])

                if not args.no_video and visible:
                    mx = int(x_world * scale) + margin
                    my = int(y_world * scale) + margin
                    if is_ref:
                        pts = np.array([
                            [mx, my - 7], [mx + 5, my],
                            [mx, my + 7], [mx - 5, my]
                        ], dtype=np.int32)
                        cv2.fillPoly(pitch_frame, [pts], color)
                    elif is_gk:
                        cv2.circle(pitch_frame, (mx, my), 8, color, -1)
                        cv2.circle(pitch_frame, (mx, my), 8, (255, 255, 255), 2)
                    else:
                        cv2.circle(pitch_frame, (mx, my), 6, color, -1)
                    map_label, map_has_jersey = get_display_label(track_id)
                    if map_label:
                        cv2.putText(pitch_frame, map_label.lstrip("#"), (mx + 8, my),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45 if map_has_jersey else 0.4,
                                    (255, 255, 255), 2 if map_has_jersey else 1)

            # -- Ball (Viterbi trajectory) --
            ball = trajectory.get(frame_id)
            if ball is not None:
                bx, by = ball["x"], ball["y"]
                if not args.no_video:
                    is_dummy = ball.get("is_dummy", False)
                    bcolor = (0, 0, 255) if is_dummy else (0, 255, 255)
                    cv2.circle(frame, (int(bx), int(by)), 5, bcolor, -1)

                x_world, y_world = project_point_to_world(bx, by, H_inv)
                visible = x_world is not None and not ball.get("is_dummy", False)

                if csv_writer:
                    csv_writer.writerow([
                        frame_id, round((frame_id - 1) * dt, 3),
                        'ball', -1, -1, 'ball',
                        round(x_world, 3) if visible else '',
                        round(y_world, 3) if visible else '',
                        1 if visible else 0,
                        'viterbi',
                        round(ball.get("score", 0.0), 3)
                    ])

                if not args.no_video and visible:
                    mx = int(x_world * scale) + margin
                    my = int(y_world * scale) + margin
                    cv2.circle(pitch_frame, (mx, my), 5, (0, 165, 255), -1)

        if not args.no_video:
            # ── Compose side-by-side ──
            frame_resized = cv2.resize(frame, (new_w_ori, target_video_h))
            pad_top = (target_video_h - h_pitch) // 2
            pad_bot = target_video_h - h_pitch - pad_top
            pitch_padded = cv2.copyMakeBorder(
                pitch_frame, pad_top, pad_bot, 0, 0,
                cv2.BORDER_CONSTANT, value=[0, 0, 0])
            out_video.write(np.hstack((frame_resized, pitch_padded)))

    if not args.no_video:
        out_video.release()

    if csv_file:
        csv_file.close()
        print(f"\nTracking 2D CSV saved to: {args.output_csv}")

    # ── Final report ──
    print(f"\n{'='*50}")
    print(f"Calibration sources:")
    print(f"  Voting (3D):   {calib_voting}/{len(images)}")
    print(f"  Ground (2D):   {calib_ground}/{len(images)}")
    print(f"  Failed (hold): {calib_fail}/{len(images)}")
    print(f"\nBidirectional Smoother statistics:")
    print(smoother.summary())
    if not args.no_video:
        print(f"\nVideo saved to: {args.output}")


if __name__ == '__main__':
    main()

