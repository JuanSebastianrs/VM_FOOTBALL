# core/scanning_v2/supervised/feature_extractor.py
"""
Extraccion de features POR EVENTO (un sample = un event_id) desde los outputs V2.

Las features resumen la ventana temporal previa a la recepcion (de head_pose.parquet)
mas senales de evento y vision map. Son orientaciones APROXIMADAS (no gaze real).

Garantias metodologicas:
  - NO se usan etiquetas (`scan_label_gt`, visibility, confidence, notes) como feature.
  - La prediccion heuristica V2 (`scan_label_pred`) se guarda como
    `heuristic_scan_label_pred` y solo se usa como feature si la config lo permite.
  - NaN se manejan explicitamente; se anota `feature_version`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from core.scanning.circular import circular_delta, orientation_entropy
from core.scanning_v2.head_turn_detector import HeadTurnDetector
from .schema import FEATURE_VERSION, HEURISTIC_PRED_COLUMN

_BACKENDS = ["yolo_pose_body", "mediapipe", "sixdrepnet", "body_orientation"]


class FeatureExtractor:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.fps = float(c.get("fps", 25.0))
        self.min_dur = int(c.get("min_turn_duration_frames", 4))
        self.min_conf = float(c.get("min_valid_confidence", 0.35))
        self.ball_cone = math.radians(float(c.get("ball_cone_deg", 35.0)))
        self.feature_version = str(c.get("feature_version", FEATURE_VERSION))
        self.include_vision_map = bool(c.get("include_vision_map", True))

    # ------------------------------------------------------------------
    def extract(self, events: pd.DataFrame, head_pose: pd.DataFrame,
                scanning: Optional[pd.DataFrame] = None,
                vision_map: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        scan_idx = (scanning.set_index("event_id") if scanning is not None
                    and len(scanning) else None)
        vis_idx = (vision_map.set_index("event_id") if vision_map is not None
                   and len(vision_map) else None)
        rows = []
        for ev in events.to_dict("records"):
            eid = ev["event_id"]
            hp = head_pose[head_pose["event_id"] == eid] if len(head_pose) else head_pose
            rows.append(self._event_features(ev, hp, scan_idx, vis_idx))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _event_features(self, ev, hp, scan_idx, vis_idx) -> dict:
        eid = ev["event_id"]
        f = {
            "event_id": eid, "video_id": ev.get("video_id"),
            "receiver_track_id": _int(ev.get("receiver_track_id")),
            "receiver_role": ev.get("receiver_role"),
            "event_source": ev.get("source", ev.get("event_source")),
            "event_confidence": _float(ev.get("event_confidence")),
            "feature_version": self.feature_version,
        }
        hp = hp.sort_values("frame_id") if len(hp) else hp
        n = int(len(hp))
        f["n_frames"] = n

        # ---- yaw series (camara-relativo, suavizado) ----
        yaw_deg = [None if pd.isna(v) else float(v) for v in hp["yaw_smooth"]] if n else []
        valid = [y for y in yaw_deg if y is not None]
        yaws_rad_full = [None if y is None else math.radians(y) for y in yaw_deg]
        f["yaw_valid_count"] = len(valid)
        # rango CIRCULAR (menor arco que contiene todos los angulos); evita el
        # falso 358° con saltos +179/-179.
        f["yaw_range_deg"] = _circular_range_deg(valid)
        f["yaw_std_deg"] = float(np.std(valid)) if len(valid) > 1 else 0.0

        deltas = []
        last = None
        for y in valid:
            if last is not None:
                deltas.append(abs(math.degrees(circular_delta(math.radians(y),
                                                              math.radians(last)))))
            last = y
        f["yaw_mean_abs_delta_deg"] = float(np.mean(deltas)) if deltas else 0.0
        f["yaw_max_delta_deg"] = float(np.max(deltas)) if deltas else 0.0
        f["yaw_num_direction_changes"] = self._direction_changes(valid)
        f["yaw_entropy"] = float(orientation_entropy([math.radians(y) for y in valid])) if valid else 0.0

        confs_unit = [1.0] * n  # conteo geometrico (no filtra por confianza)
        for thr in (20.0, 30.0, 40.0):
            cnt, mx = self._sustained(yaws_rad_full, confs_unit, thr)
            f[f"sustained_turn_count_{int(thr)}deg"] = int(cnt)
            if thr == 20.0:
                f["max_sustained_turn_deg"] = float(mx)

        # ---- temporal ----
        t2r = [float(t) for t, y in zip(hp["time_to_reception"], yaw_deg)
               if y is not None and pd.notna(t)] if n else []
        f["first_valid_time_to_reception"] = float(max(t2r)) if t2r else np.nan
        f["last_valid_time_to_reception"] = float(min(t2r)) if t2r else np.nan
        for sec in (1.0, 2.0, 3.0):
            f[f"scan_candidate_before_{int(sec)}s"] = self._candidate_before(hp, yaws_rad_full, sec)

        # ---- calidad ----
        f["valid_pose_ratio"] = self._scan_or(scan_idx, eid, "valid_pose_ratio",
                                              (len(valid) / n) if n else 0.0)
        f["mean_head_pose_confidence"] = _meanc(hp, "head_pose_confidence")
        f["mean_yaw_confidence_smooth"] = _meanc(hp, "yaw_confidence_smooth")
        f["head_crop_valid_ratio"] = _meanbool(hp, "head_crop_valid")
        f["mean_head_crop_quality"] = _meanc(hp, "head_crop_quality")
        bu = hp["head_pose_backend_used"].value_counts().to_dict() if n else {}
        for b in _BACKENDS:
            f[f"backend_{b}_ratio"] = (bu.get(b, 0) / n) if n else 0.0

        # ---- ball / body ----
        dist = pd.to_numeric(hp["distance_to_ball"], errors="coerce").dropna() if n else pd.Series([], dtype=float)
        f["mean_distance_to_ball"] = float(dist.mean()) if len(dist) else np.nan
        f["min_distance_to_ball"] = float(dist.min()) if len(dist) else np.nan
        away, back = self._look_counts(hp) if n else (0, 0)
        f["look_away_from_ball_count"] = int(away)
        f["look_back_to_ball_count"] = int(back)
        f["look_away_ratio"] = float(away / (away + back)) if (away + back) else 0.0
        f["theta_body_field_valid_ratio"] = _notna_ratio(hp, "theta_body_field")

        # ---- vision map (opcional) ----
        f.update(self._vision_features(vis_idx, eid))

        # ---- heuristica V2 (NO feature por defecto; etiqueta de comparacion) ----
        f[HEURISTIC_PRED_COLUMN] = self._scan_or(scan_idx, eid, "scan_label_pred", 0)
        return f

    # ------------------------------------------------------------------
    def _sustained(self, yaws_rad_full, confs, thr_deg):
        det = HeadTurnDetector({"yaw_turn_threshold_deg": thr_deg,
                                "min_turn_duration_frames": self.min_dur,
                                "min_mean_confidence": 0.0})
        return det._count_turns(yaws_rad_full, confs)

    def _candidate_before(self, hp, yaws_rad_full, sec):
        if not len(hp):
            return 0
        mask = (pd.to_numeric(hp["time_to_reception"], errors="coerce") <= sec).tolist()
        sub_yaws = [y for y, m in zip(yaws_rad_full, mask) if m]
        if len([y for y in sub_yaws if y is not None]) < self.min_dur:
            return 0
        cnt, _ = self._sustained(sub_yaws, [1.0] * len(sub_yaws), 20.0)
        return int(cnt > 0)

    @staticmethod
    def _direction_changes(valid):
        floor = math.radians(2.0)
        signs, prev = [], None
        for y in valid:
            if prev is not None:
                d = circular_delta(math.radians(y), math.radians(prev))
                if abs(d) >= floor:
                    signs.append(1 if d > 0 else -1)
            prev = y
        return int(sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b))

    def _look_counts(self, hp):
        away = back = 0
        for _, r in hp.iterrows():
            tb, br = r.get("theta_body_field"), r.get("ball_relative_angle_field")
            if pd.isna(tb) or pd.isna(br):
                continue
            d = circular_delta(math.radians(float(tb)), math.radians(float(br)))
            if abs(d) <= self.ball_cone:
                back += 1
            else:
                away += 1
        return away, back

    def _vision_features(self, vis_idx, eid):
        d = {"vision_map_available": 0, "used_orientation_body": 0,
             "used_orientation_head": 0, "vision_map_confidence": 0.0,
             "n_visible_teammates": 0, "n_visible_opponents": 0,
             "visible_ball": 0, "observed_space_score": 0.0}
        if not self.include_vision_map or vis_idx is None or eid not in vis_idx.index:
            return d
        r = vis_idx.loc[eid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        used = str(r.get("used_orientation"))
        d.update(
            vision_map_available=int(bool(r.get("vision_map_available"))),
            used_orientation_body=int(used == "body"),
            used_orientation_head=int(used == "head"),
            vision_map_confidence=_float(r.get("vision_map_confidence")) or 0.0,
            n_visible_teammates=_int(r.get("n_visible_teammates")) or 0,
            n_visible_opponents=_int(r.get("n_visible_opponents")) or 0,
            visible_ball=int(bool(r.get("visible_ball"))),
            observed_space_score=_float(r.get("observed_space_score")) or 0.0)
        return d

    @staticmethod
    def _scan_or(scan_idx, eid, col, default):
        if scan_idx is None or eid not in scan_idx.index:
            return default
        r = scan_idx.loc[eid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        v = r.get(col)
        return default if pd.isna(v) else v


def _circular_range_deg(values_deg) -> float:
    """Menor arco (en grados) que contiene todos los angulos. El rango circular
    es 2*pi - (mayor hueco vacio entre angulos consecutivos ordenados)."""
    if not values_deg or len(values_deg) < 2:
        return 0.0
    ang = sorted((math.radians(v) % (2 * math.pi)) for v in values_deg)
    gaps = [b - a for a, b in zip(ang[:-1], ang[1:])]
    gaps.append((ang[0] + 2 * math.pi) - ang[-1])   # hueco que cruza el wrap
    return float(math.degrees(2 * math.pi - max(gaps)))


def _int(v):
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def _meanc(hp, col):
    if not len(hp) or col not in hp.columns:
        return np.nan
    s = pd.to_numeric(hp[col], errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _meanbool(hp, col):
    if not len(hp) or col not in hp.columns:
        return 0.0
    return float(hp[col].astype(bool).mean())


def _notna_ratio(hp, col):
    if not len(hp) or col not in hp.columns:
        return 0.0
    return float(hp[col].notna().mean())
