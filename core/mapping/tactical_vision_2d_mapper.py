"""
TacticalVision AI — Phase 9: 2D Field Mapping

Projects player/ball detections from image space onto a 2D minimap
using PnLCalib per-frame camera calibration with multi-layer temporal
stabilisation.

Key improvements over naive approach:
  1. Decompose → Smooth → Reconstruct: camera params are EMA-smoothed in
     their natural spaces (Euler angles, focal length, 3D position)
     instead of raw H_inv matrix interpolation.
  2. Outlier gating: rejects calibration results that are geometrically
     inconsistent with the smoothed state (rate-of-change thresholds).
  3. Dual-path calibration: falls back to ground-plane homography
     (heuristic_voting_ground) when full 3D calibration fails.
"""

import cv2
import numpy as np
import json
import os
import glob
import argparse
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
    pan_tilt_roll_to_orientation,
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
# CameraParamsSmoother — decompose → smooth → reconstruct
# ---------------------------------------------------------------------------

class CameraParamsSmoother:
    """
    Temporal stabiliser for PnLCalib camera calibration.

    Instead of smoothing the raw H_inv matrix (which is mathematically
    invalid for projective transforms), this class:
      1. Decomposes each calibration into natural parameter spaces
         (pan/tilt/roll, focal length, 3D position).
      2. Applies independent EMA smoothing per parameter, with circular
         EMA for angles.
      3. Reconstructs the projection matrix from smoothed parameters.
      4. Rejects outlier calibrations that change too fast (rate-of-change
         thresholds on each parameter).
    """

    # ── EMA weights (per-parameter family) ──
    ALPHA_ANGLES = 0.25      # pan, tilt, roll — moderate responsiveness
    ALPHA_FOCAL  = 0.05      # fx, fy — nearly constant in broadcast
    ALPHA_POS    = 0.10      # camera position — nearly constant (tripod)

    # ── Outlier detection: max change per frame ──
    MAX_PAN_CHANGE_DEG   = 5.0     # degrees
    MAX_TILT_CHANGE_DEG  = 3.0
    MAX_ROLL_CHANGE_DEG  = 2.0
    MAX_FOCAL_RATIO      = 0.20    # 20 % relative change
    MAX_POS_CHANGE_M     = 20.0    # metres
    MAX_REPROJ_ERR_PX    = 20.0    # pixels (PnLCalib's own metric)

    # ── Recovery ──
    MAX_CONSECUTIVE_REJECTS = 15   # after this, force-accept
    WARMUP_FRAMES           = 3    # accept everything during warm-up

    def __init__(self):
        # Smoothed state
        self.pan  = None   # radians
        self.tilt = None
        self.roll = None
        self.fx   = None
        self.fy   = None
        self.cx   = None
        self.cy   = None
        self.pos  = None   # np.array([x, y, z])

        self.frame_count  = 0
        self.reject_count = 0

        self.stats = {
            'accepted': 0,
            'rejected_outlier': 0,
            'rejected_reproj': 0,
            'force_accepted': 0,
            'fallback': 0,
        }

    # ── Angle helpers ──

    @staticmethod
    def _wrap(diff):
        """Wrap angle difference to [-π, π]."""
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def _circular_ema(self, old, new, alpha):
        """EMA for angles that handles wrap-around."""
        return old + alpha * self._wrap(new - old)

    # ── Consistency check ──

    def _is_consistent(self, pan, tilt, roll, fx, fy, pos, rep_err):
        """
        Returns (ok: bool, reason: str).
        Checks PnLCalib reprojection error AND per-parameter rate-of-change.
        """
        if rep_err > self.MAX_REPROJ_ERR_PX:
            return False, 'reproj'

        if self.pan is None:
            return True, ''

        d_pan  = abs(np.rad2deg(self._wrap(pan  - self.pan)))
        d_tilt = abs(np.rad2deg(self._wrap(tilt - self.tilt)))
        d_roll = abs(np.rad2deg(self._wrap(roll - self.roll)))

        if d_pan  > self.MAX_PAN_CHANGE_DEG:
            return False, f'pan Δ{d_pan:.1f}°'
        if d_tilt > self.MAX_TILT_CHANGE_DEG:
            return False, f'tilt Δ{d_tilt:.1f}°'
        if d_roll > self.MAX_ROLL_CHANGE_DEG:
            return False, f'roll Δ{d_roll:.1f}°'

        if self.fx > 0:
            if abs(fx - self.fx) / self.fx > self.MAX_FOCAL_RATIO:
                return False, 'focal'

        if np.linalg.norm(pos - self.pos) > self.MAX_POS_CHANGE_M:
            return False, 'position'

        return True, ''

    # ── Core update ──

    def update(self, cam_params, rep_err=0.0):
        """
        Feed a new PnLCalib result.  Returns smoothed H_inv (image → world).
        """
        self.frame_count += 1

        # Decompose into natural spaces
        pan  = np.deg2rad(cam_params['pan_degrees'])
        tilt = np.deg2rad(cam_params['tilt_degrees'])
        roll = np.deg2rad(cam_params['roll_degrees'])
        fx   = cam_params['x_focal_length']
        fy   = cam_params['y_focal_length']
        cx, cy = cam_params['principal_point']
        pos  = np.array(cam_params['position_meters'],
                        dtype=np.float64).flatten()

        # ── Warm-up: accept with alpha=1 ──
        if self.frame_count <= self.WARMUP_FRAMES:
            self.pan, self.tilt, self.roll = pan, tilt, roll
            self.fx, self.fy = fx, fy
            self.cx, self.cy = cx, cy
            self.pos = pos.copy()
            self.stats['accepted'] += 1
            return self._build_H_inv()

        # ── Consistency gate ──
        ok, reason = self._is_consistent(pan, tilt, roll, fx, fy, pos, rep_err)

        if ok:
            alpha_a = self.ALPHA_ANGLES
            alpha_f = self.ALPHA_FOCAL
            alpha_p = self.ALPHA_POS
            self.reject_count = 0
            self.stats['accepted'] += 1

        elif self.reject_count >= self.MAX_CONSECUTIVE_REJECTS:
            # Force-accept with high alpha to recover
            alpha_a = 0.8
            alpha_f = 0.5
            alpha_p = 0.8
            self.reject_count = 0
            self.stats['force_accepted'] += 1

        else:
            # Reject — hold smoothed state
            self.reject_count += 1
            if 'reproj' in reason:
                self.stats['rejected_reproj'] += 1
            else:
                self.stats['rejected_outlier'] += 1
            return self._build_H_inv()

        # ── Apply EMA ──
        self.pan  = self._circular_ema(self.pan,  pan,  alpha_a)
        self.tilt = self._circular_ema(self.tilt, tilt, alpha_a)
        self.roll = self._circular_ema(self.roll, roll, alpha_a)

        self.fx = alpha_f * fx + (1 - alpha_f) * self.fx
        self.fy = alpha_f * fy + (1 - alpha_f) * self.fy
        self.cx = alpha_f * cx + (1 - alpha_f) * self.cx
        self.cy = alpha_f * cy + (1 - alpha_f) * self.cy

        self.pos = alpha_p * pos + (1 - alpha_p) * self.pos

        return self._build_H_inv()

    # ── Reconstruct from smoothed state ──

    def _build_H_inv(self):
        """Reconstruct H_inv from smoothed camera parameters."""
        if self.pan is None:
            return None

        # pan_tilt_roll_to_orientation returns R^T (orientation)
        orientation = pan_tilt_roll_to_orientation(self.pan, self.tilt, self.roll)
        rotation = orientation.T  # R = orientation^T

        Q = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1]
        ])

        It = np.eye(4)[:-1]
        It[:, -1] = -self.pos

        P = Q @ (rotation @ It)
        H = P[:, [0, 1, 3]]   # ground plane (Z=0)
        try:
            H_inv = np.linalg.inv(H)
            return H_inv
        except np.linalg.LinAlgError:
            return None

    def get_current_H_inv(self):
        """Get current smoothed H_inv without feeding new data."""
        if self.pan is None:
            return None
        return self._build_H_inv()

    def summary(self):
        s = self.stats
        total = sum(s.values())
        return (f"  Accepted:          {s['accepted']}\n"
                f"  Rejected (outlier):{s['rejected_outlier']}\n"
                f"  Rejected (reproj): {s['rejected_reproj']}\n"
                f"  Force-accepted:    {s['force_accepted']}\n"
                f"  Fallback (hold):   {s['fallback']}\n"
                f"  Total frames:      {total}")


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

def project_point(x_img, y_img, H_inv, scale, margin, w_pitch, h_pitch):
    """
    Project an image-space point (x_img, y_img) onto the minimap using H_inv.
    H_inv maps image coords → PnLCalib world coords (centred on pitch centre).
    Returns (mx, my) minimap pixel coords, or (None, None) if out of bounds.
    """
    pt_world = H_inv @ np.array([x_img, y_img, 1.0])
    pt_world /= pt_world[2]

    # PnLCalib world has origin at pitch centre → shift to top-left origin
    x_world = pt_world[0] + 52.5
    y_world = pt_world[1] + 34.0

    # Loose sanity bounds
    if x_world < -5 or x_world > 110 or y_world < -5 or y_world > 73:
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
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, 25.0, (out_w, out_h))

    # --- Calibration objects ---
    cam = FramebyFrameCalib(iwidth=w_ori, iheight=h_ori, denormalize=True)
    smoother = CameraParamsSmoother()

    pnl_refine = not args.disable_pnl_refine

    calib_voting = 0
    calib_ground = 0
    calib_fail   = 0

    print(f"Processing {len(images)} frames (PnL refine={'ON' if pnl_refine else 'OFF'})...")

    for img_path in tqdm(images):
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
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

        # ── Feed into temporal smoother ──
        if cam_params_dict is not None:
            H_inv = smoother.update(cam_params_dict, rep_err)
        else:
            H_inv = smoother.get_current_H_inv()
            smoother.stats['fallback'] += 1
            calib_fail += 1
            source = 'hold'

        if args.debug and frame_id % 25 == 0:
            if smoother.pan is not None:
                tqdm.write(
                    f"  [{frame_id:>5d}] src={source:6s} "
                    f"pan={np.rad2deg(smoother.pan):7.1f}° "
                    f"tilt={np.rad2deg(smoother.tilt):6.1f}° "
                    f"fx={smoother.fx:7.0f} "
                    f"rej={smoother.reject_count}")

        # ── Render minimap ──
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

                # Draw bbox on video frame with team color
                cv2.rectangle(frame,
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),
                              color, 2)

                # Label on video frame
                info_team = team_map.get(track_id)
                if info_team:
                    label = f"#{track_id}"
                    if is_ref:
                        label = f"REF #{track_id}"
                    elif is_gk:
                        label = f"GK #{track_id}"
                    cv2.putText(frame, label,
                                (int(x_min), int(y_min) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                                cv2.LINE_AA)

                # Project foot-point (bottom-centre of bbox)
                x_foot = (x_min + x_max) / 2.0
                y_foot = y_max

                mx, my = project_point(
                    x_foot, y_foot, H_inv, scale, margin, w_pitch, h_pitch
                )
                if mx is not None:
                    # Minimap: team color dot, special markers for GK/REF
                    if is_ref:
                        # Referees: diamond shape
                        pts = np.array([
                            [mx, my - 7], [mx + 5, my],
                            [mx, my + 7], [mx - 5, my]
                        ], dtype=np.int32)
                        cv2.fillPoly(pitch_frame, [pts], color)
                    elif is_gk:
                        # Goalkeepers: larger circle + ring
                        cv2.circle(pitch_frame, (mx, my), 8, color, -1)
                        cv2.circle(pitch_frame, (mx, my), 8, (255, 255, 255), 2)
                    else:
                        cv2.circle(pitch_frame, (mx, my), 6, color, -1)

                    cv2.putText(pitch_frame, str(track_id), (mx + 8, my),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1)

            # -- Ball (Viterbi trajectory) --
            ball = trajectory.get(frame_id)
            if ball is not None:
                bx, by = ball["x"], ball["y"]
                is_dummy = ball.get("is_dummy", False)
                bcolor = (0, 0, 255) if is_dummy else (0, 255, 255)
                cv2.circle(frame, (int(bx), int(by)), 5, bcolor, -1)

                mx, my = project_point(
                    bx, by, H_inv, scale, margin, w_pitch, h_pitch
                )
                if mx is not None:
                    cv2.circle(pitch_frame, (mx, my), 5, (0, 165, 255), -1)

        # ── Compose side-by-side ──
        frame_resized = cv2.resize(frame, (new_w_ori, target_video_h))
        pad_top = (target_video_h - h_pitch) // 2
        pad_bot = target_video_h - h_pitch - pad_top
        pitch_padded = cv2.copyMakeBorder(
            pitch_frame, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=[0, 0, 0])

        out_video.write(np.hstack((frame_resized, pitch_padded)))

    out_video.release()

    # ── Final report ──
    print(f"\n{'='*50}")
    print(f"Calibration sources:")
    print(f"  Voting (3D):   {calib_voting}/{len(images)}")
    print(f"  Ground (2D):   {calib_ground}/{len(images)}")
    print(f"  Failed (hold): {calib_fail}/{len(images)}")
    print(f"\nSmoother statistics:")
    print(smoother.summary())
    print(f"\nVideo saved to: {args.output}")


if __name__ == '__main__':
    main()
