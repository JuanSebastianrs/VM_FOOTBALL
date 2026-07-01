# core/scanning_v2/visualization.py
"""
FASE 9 — Visualizacion por evento (V2).

Genera por evento:
  {event_id}_video.mp4   : receptor resaltado + bbox + track/rol/equipo + yaw +
                           marca de head-turn + recepcion + balon + backend + source.
  {event_id}_minimap.mp4 : receptor + balon + companeros/rivales + (si hay
                           orientacion en cancha) flecha/cono de cuerpo; si NO la
                           hay, muestra el aviso y NO dibuja una flecha falsa.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from core.scanning.circular import circular_delta
from core.scanning.data_io import SequenceData
from core.scanning.visualization import (
    ScanningVisualizer, _arrow, _team_color, _BALL_COLOR, _SCAN_COLOR)

_RECV = (40, 220, 255)
_GREEN = (40, 255, 40)


class ScanningV2Visualizer:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self._pitch = ScanningVisualizer(c.get("vision_map", c))
        self.fps = float(c.get("fps", 25.0))
        ht = c.get("head_turn", {}) if isinstance(c.get("head_turn"), dict) else {}
        self.turn_thr = float(ht.get("yaw_turn_threshold_deg", 40.0))
        self.min_dur = int(ht.get("min_turn_duration_frames", 4))

    # ------------------------------------------------------------------
    def _turning_frames(self, hp) -> set:
        """Frames donde realmente hay un giro sostenido (no todo el clip):
        |yaw_smooth(f) - yaw_smooth(f-min_dur)| >= turn_thr/2."""
        ser = (hp.dropna(subset=["yaw_smooth"]).sort_values("frame_id")
               [["frame_id", "yaw_smooth"]])
        if len(ser) < self.min_dur + 1:
            return set()
        fids = ser["frame_id"].astype(int).tolist()
        yaws = [math.radians(float(v)) for v in ser["yaw_smooth"].tolist()]
        turning = set()
        thr = math.radians(self.turn_thr * 0.5)
        for i in range(self.min_dur, len(yaws)):
            if abs(circular_delta(yaws[i], yaws[i - self.min_dur])) >= thr:
                for k in range(i - self.min_dur, i + 1):
                    turning.add(fids[k])
        return turning

    @staticmethod
    def _draw_yaw_curve(canvas, hp, cur_fid):
        """Dibuja la curva de yaw_smooth de la ventana en una esquina del video."""
        ser = (hp.dropna(subset=["yaw_smooth"]).sort_values("frame_id")
               [["frame_id", "yaw_smooth"]])
        H, W = canvas.shape[:2]
        pw, ph = 280, 90
        x0, y0 = W - pw - 12, 12
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (200, 200, 200), 1)
        cv2.putText(canvas, "yaw_smooth (cam-rel)", (x0 + 6, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        if len(ser) < 2:
            return
        fids = ser["frame_id"].astype(int).tolist()
        yaws = [float(v) for v in ser["yaw_smooth"].tolist()]
        f_min, f_max = fids[0], fids[-1]
        span = max(1, f_max - f_min)
        mid = y0 + ph // 2 + 8
        pts = []
        for f, y in zip(fids, yaws):
            px = x0 + 6 + int((f - f_min) / span * (pw - 12))
            py = int(mid - (y / 90.0) * (ph / 2 - 12))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, (40, 220, 255), 1)
        # marcador del frame actual
        cx = x0 + 6 + int((cur_fid - f_min) / span * (pw - 12))
        cv2.line(canvas, (cx, y0 + 16), (cx, y0 + ph), (40, 255, 40), 1)

    @staticmethod
    def _overlay_head_crop(canvas, crop):
        """Miniatura del head crop en la esquina inferior izquierda (si existe)."""
        if crop is None or crop.size == 0:
            return
        th = 96
        scale = th / max(1, crop.shape[0])
        tw = max(1, int(crop.shape[1] * scale))
        thumb = cv2.resize(crop, (tw, th))
        H = canvas.shape[0]
        y0, x0 = H - th - 12, 12
        canvas[y0:y0 + th, x0:x0 + tw] = thumb
        cv2.rectangle(canvas, (x0, y0), (x0 + tw, y0 + th), (200, 200, 200), 1)
        cv2.putText(canvas, "head crop", (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    def render_event(self, seq: SequenceData, gs: pd.DataFrame, head_pose: pd.DataFrame,
                     scan_row: dict, event: dict, out_dir: Path,
                     scale: int = 11, margin: int = 24) -> Optional[Path]:
        eid = event["event_id"]
        tid = int(event["receiver_track_id"])
        f_rec = int(event["frame_reception"])
        ws, we = int(scan_row["window_start"]), f_rec
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

        w, h = seq.frame_size or (1920, 1080)
        vw = cv2.VideoWriter(str(out_dir / f"{eid}_video.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (w, h))
        base = self._pitch._draw_pitch(scale, margin)
        mh, mw = base.shape[:2]
        mwr = cv2.VideoWriter(str(out_dir / f"{eid}_minimap.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (mw, mh))

        hp_ev = head_pose[head_pose["event_id"] == eid]
        hp = hp_ev.set_index("frame_id")
        scan_label = int(scan_row.get("scan_label_pred", 0))
        backend = scan_row.get("source", "heuristic")
        # frames con giro REAL (no se marca todo el clip)
        turning = self._turning_frames(hp_ev) if scan_label else set()
        head_dir = Path(out_dir).parent / "windows" / str(eid) / "head_crops"

        for fid in range(ws, f_rec + 1):
            ipath = seq.image_paths.get(fid)
            img = cv2.imread(str(ipath)) if ipath else None
            pitch = base.copy()
            is_rec = (fid == f_rec)
            gframe = gs[gs["frame_id"] == fid]
            recv = gframe[gframe["track_id"] == tid]

            # --- minimap: todos los candidatos + receptor ---
            for _, r in gframe.iterrows():
                X, Y = r.get("x_field"), r.get("y_field")
                if pd.isna(X) or pd.isna(Y):
                    continue
                mx, my = self._pitch._to_minimap(float(X), float(Y), scale, margin)
                if not (0 <= mx < mw and 0 <= my < mh):
                    continue
                col = _team_color(r.get("team_id")) if not r.get("is_referee") else (180, 180, 180)
                cv2.circle(pitch, (mx, my), 5 if int(r["track_id"]) == tid else 3, col, -1)

            # --- video + minimap del receptor ---
            yaw = roll = None; conf = 0.0; hpb = "?"; tbf = None
            if fid in hp.index:
                row = hp.loc[fid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                yaw = None if pd.isna(row.get("yaw_smooth")) else float(row["yaw_smooth"])
                conf = float(row.get("head_pose_confidence") or 0.0)
                hpb = str(row.get("head_pose_backend_used"))
                tbf = None if pd.isna(row.get("theta_body_field")) else float(row["theta_body_field"])

            is_turning = (fid in turning)
            if img is not None and not recv.empty:
                rr = recv.iloc[0]
                x1, y1, x2, y2 = int(rr["bbox_x1"]), int(rr["bbox_y1"]), int(rr["bbox_x2"]), int(rr["bbox_y2"])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(img, (x1, y1), (x2, y2), _RECV, 2)
                conf_ev = scan_row.get("event_confidence", event.get("event_confidence"))
                lab = (f"#{tid} {rr.get('role')}/{rr.get('team_id')} "
                       f"src={backend} ev_c={conf_ev}")
                lab2 = f"yaw={yaw:.0f} c={conf:.2f} hp={hpb}" if yaw is not None else f"yaw=NA hp={hpb}"
                cv2.putText(img, lab, (x1, max(14, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _RECV, 2)
                cv2.putText(img, lab2, (x1, max(30, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _RECV, 1)
                # anillo: recepcion=verde, giro REAL=scan-color, normal=gris
                ring = _GREEN if is_rec else (_SCAN_COLOR if is_turning else (120, 120, 120))
                cv2.circle(img, (cx, cy), 26 if is_rec else 20, ring, 3 if is_rec else 2)
                if is_rec:
                    cv2.putText(img, "RECEPTION", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2)
                elif is_turning:
                    cv2.putText(img, "HEAD-TURN", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _SCAN_COLOR, 2)

            if not recv.empty:
                rr = recv.iloc[0]
                X, Y = rr.get("x_field"), rr.get("y_field")
                if not (pd.isna(X) or pd.isna(Y)):
                    mx, my = self._pitch._to_minimap(float(X), float(Y), scale, margin)
                    if 0 <= mx < mw and 0 <= my < mh:
                        cv2.circle(pitch, (mx, my), 7, _RECV, 2)
                        if tbf is not None:   # orientacion de CUERPO en cancha
                            thr = math.radians(tbf)
                            self._pitch._draw_cone(pitch, mx, my, thr, _RECV, scale)
                            _arrow(pitch, mx, my, thr, 18, _RECV, 2)
                        else:
                            cv2.putText(pitch, "orientation field unavailable",
                                        (10, mh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        (60, 60, 255), 1)

            # balon
            ball = seq.ball(fid)
            if img is not None and ball is not None and ball.valid:
                cv2.circle(img, (int(ball.x), int(ball.y)), 6, _BALL_COLOR, -1)
            bxy = seq.ball_field_xy(fid) if seq.homography.available else None
            if bxy is not None:
                mx, my = self._pitch._to_minimap(bxy[0], bxy[1], scale, margin)
                if 0 <= mx < mw and 0 <= my < mh:
                    cv2.circle(pitch, (mx, my), 4, _BALL_COLOR, -1)

            if img is not None:
                # curva de yaw_smooth de la ventana + marcador del frame actual
                self._draw_yaw_curve(img, hp_ev, fid)
                # miniatura del head crop (si se guardo en windows/{eid}/head_crops)
                hc_path = head_dir / f"f{fid:06d}.jpg"
                if hc_path.exists():
                    self._overlay_head_crop(img, cv2.imread(str(hc_path)))
                cv2.putText(img, f"{eid}  frame {fid}  t-{(f_rec - fid) / seq.fps:.1f}s",
                            (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                vw.write(img)
            mwr.write(pitch)
        vw.release(); mwr.release()
        return out_dir / f"{eid}_video.mp4"
