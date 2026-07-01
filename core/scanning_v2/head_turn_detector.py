# core/scanning_v2/head_turn_detector.py
"""
FASE 7 — Deteccion de head-turn / scanning antes de la recepcion.

Trabaja sobre `yaw_smooth` (camara-relativo) del receptor en la ventana previa a
la recepcion. Un head-turn es un giro SOSTENIDO (no jitter de un frame):

  scan_label_pred = 1  si head_turn_count>0 y valid_pose_ratio y confianza media
                       superan los umbrales.

`look_away/back_from_ball` se computan con la orientacion de CUERPO en cancha
(`theta_body_field`) frente a `ball_relative_angle_field` — NO con el yaw de
cabeza (camara-relativo). Esto evita mezclar marcos de referencia.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from core.scanning.circular import circular_delta, orientation_entropy
from .schema import SCANNING_V2_COLUMNS


class HeadTurnDetector:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.fps = float(c.get("fps", 25.0))
        self.before = float(c.get("seconds_before_reception", 3.0))
        self.after = float(c.get("seconds_after_reception", 0.2))
        self.turn_thr = math.radians(float(c.get("yaw_turn_threshold_deg", 40.0)))
        self.min_dur = int(c.get("min_turn_duration_frames", 4))
        self.min_valid_ratio = float(c.get("min_valid_pose_ratio", 0.4))
        self.min_mean_conf = float(c.get("min_mean_confidence", 0.35))
        self.ball_cone = math.radians(float(c.get("ball_cone_deg", 35.0)))
        # un giro de CUERPO (fallback body_orientation) NO debe disparar head-turn:
        # se excluyen esas filas del conteo de giros salvo que se permita por config.
        self.allow_body_fallback_for_scan = bool(
            c.get("allow_body_fallback_for_scan", False))

    def _count_turns(self, yaws_rad, confs):
        """Giros SOSTENIDOS (no jitter): deltas consecutivos del mismo signo por
        encima de un piso de ruido. Un frame plano (|delta|<piso) o un cambio de
        signo cierra el segmento. `min_turn_duration_frames` se mide en numero de
        deltas del segmento. (count, max_change_deg)."""
        seq = [(y, c) for y, c in zip(yaws_rad, confs)
               if y is not None and c >= self.min_mean_conf]
        if len(seq) < 2:
            return 0, 0.0
        floor = math.radians(2.0)        # piso de ruido por frame
        count, max_turn = 0, 0.0
        accum, length, sign = 0.0, 0, 0
        cs = []

        def close():
            nonlocal count, max_turn
            if (length >= self.min_dur and abs(accum) >= self.turn_thr
                    and (np.mean(cs) if cs else 0.0) >= self.min_mean_conf):
                count += 1
                max_turn = max(max_turn, abs(accum))

        for (p, _), (cur, cc) in zip(seq[:-1], seq[1:]):
            d = circular_delta(cur, p)
            s = 0 if abs(d) < floor else (1 if d > 0 else -1)
            if s == 0:                                   # frame plano: cierra
                close(); accum, length, sign, cs = 0.0, 0, 0, []
            elif sign == 0 or s == sign:                 # extiende el giro
                accum += d; length += 1; sign = s; cs.append(cc)
            else:                                        # cambio de direccion
                close(); accum, length, sign, cs = d, 1, s, [cc]
        close()
        return count, math.degrees(max_turn)

    def evaluate(self, event: dict, hp: pd.DataFrame) -> dict:
        f_rec = int(event["frame_reception"])
        # clamp: la ventana nunca empieza en un frame negativo / antes del disponible
        floor_frame = int(hp["frame_id"].min()) if len(hp) else 0
        w_start = max(max(0, floor_frame), int(f_rec - self.before * self.fps))
        w_end = int(f_rec - self.after * self.fps)
        if w_end < w_start:
            w_end = w_start
        win = hp[(hp["frame_id"] >= w_start) & (hp["frame_id"] <= w_end)].sort_values("frame_id")

        # confianza: preferir la suavizada (yaw_confidence_smooth) si existe
        conf_col = ("yaw_confidence_smooth"
                    if "yaw_confidence_smooth" in win.columns
                    and win["yaw_confidence_smooth"].notna().any()
                    else "head_pose_confidence")
        # excluir fallback corporal del conteo de giros (giro de cuerpo != head-turn)
        is_body = (win["head_pose_backend_used"] == "body_orientation"
                   if "head_pose_backend_used" in win.columns
                   else pd.Series(False, index=win.index))

        yaws, confs = [], []
        for (_, row), body in zip(win.iterrows(), is_body):
            v = row.get("yaw_smooth")
            c = row.get(conf_col)
            if body and not self.allow_body_fallback_for_scan:
                yaws.append(None); confs.append(0.0)
                continue
            yaws.append(None if pd.isna(v) else math.radians(float(v)))
            confs.append(float(c) if pd.notna(c) else 0.0)
        turn_count, max_change = self._count_turns(yaws, confs)

        valid = [(y, c) for y, c in zip(yaws, confs) if y is not None and c >= self.min_mean_conf]
        deltas = [abs(circular_delta(b[0], a[0])) for a, b in zip(valid[:-1], valid[1:])]
        mean_change = math.degrees(float(np.mean(deltas))) if deltas else 0.0
        entropy = orientation_entropy([y for y, _ in valid]) if valid else 0.0

        # look toward/away: CUERPO en cancha vs balon en cancha (mismo marco)
        away = back = 0
        for _, row in win.iterrows():
            tb = row.get("theta_body_field"); br = row.get("ball_relative_angle_field")
            if pd.isna(tb) or pd.isna(br):
                continue
            if abs(circular_delta(math.radians(float(tb)), math.radians(float(br)))) <= self.ball_cone:
                back += 1
            else:
                away += 1

        win_frames = max(1, w_end - w_start + 1)
        valid_ratio = len(valid) / win_frames
        mean_conf = float(np.mean(confs)) if confs else 0.0
        quality = float(np.clip(valid_ratio * mean_conf, 0.0, 1.0))

        scan_label = int(turn_count > 0 and valid_ratio >= self.min_valid_ratio
                         and mean_conf >= self.min_mean_conf)

        return {
            "event_id": event["event_id"], "video_id": event["video_id"],
            "receiver_track_id": int(event["receiver_track_id"]),
            "receiver_role": event.get("receiver_role"),
            "frame_reception": f_rec, "window_start": w_start, "window_end": w_end,
            "scan_label_pred": scan_label, "head_turn_count": int(turn_count),
            "max_yaw_change_deg": round(float(max_change), 2),
            "mean_yaw_change_deg": round(float(mean_change), 2),
            "yaw_entropy": round(float(entropy), 3),
            "look_away_from_ball_count": int(away),
            "look_back_to_ball_count": int(back),
            "valid_pose_ratio": round(float(valid_ratio), 3),
            "mean_head_pose_confidence": round(float(mean_conf), 3),
            "quality_score": round(float(quality), 3),
            "source": event.get("source", "heuristic"),
        }

    def evaluate_all(self, events: pd.DataFrame, head_pose: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for ev in events.to_dict("records"):
            sub = head_pose[head_pose["event_id"] == ev["event_id"]]
            rows.append(self.evaluate(ev, sub))
        return pd.DataFrame(rows, columns=SCANNING_V2_COLUMNS)
