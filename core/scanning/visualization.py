# core/scanning/visualization.py
"""
Visualizaciones de scanning sobre video y sobre minimapa.

A. Video overlay : bbox + track_id + flecha de orientacion suavizada +
                   confianza + marca de SCANNING/RECEPTION + balon.
B. Minimap       : posicion en cancha + flecha theta_field + cono de vision +
                   balon + evento de recepcion.

Consume el DataFrame de orientacion (en grados, convencion matematica) y la
lista de eventos de scanning. Es independiente del calculo: solo dibuja.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .data_io import SequenceData

_TEAM_COLORS = {0: (60, 60, 230), 1: (230, 170, 40)}   # BGR
_DEFAULT_COLOR = (160, 160, 160)
_SCAN_COLOR = (40, 220, 255)
_BALL_COLOR = (255, 255, 255)


def _team_color(team_id) -> tuple:
    return _TEAM_COLORS.get(team_id, _DEFAULT_COLOR)


def _col(row, *names):
    """Primer valor no-nulo/no-NaN entre varias columnas (compat de nombres)."""
    for n in names:
        if n in row:
            v = row[n]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return v
    return None


def _arrow(img, cx, cy, theta_rad, length, color, thickness=2):
    ex = int(cx + length * math.cos(theta_rad))
    ey = int(cy - length * math.sin(theta_rad))   # imagen: +y abajo
    cv2.arrowedLine(img, (int(cx), int(cy)), (ex, ey), color, thickness,
                    tipLength=0.3)


class ScanningVisualizer:
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.fov = math.radians(float(config.get("fov_deg", 120.0)))
        self.pitch_length = float(config.get("pitch_length", 105.0))
        self.pitch_width = float(config.get("pitch_width", 68.0))

    # ----- indexar datos -----
    @staticmethod
    def _index_orientation(df: pd.DataFrame) -> Dict[int, List[dict]]:
        by_frame: Dict[int, List[dict]] = {}
        for row in df.to_dict("records"):
            by_frame.setdefault(int(row["frame_id"]), []).append(row)
        return by_frame

    @staticmethod
    def _scan_windows(events: List[dict]) -> List[dict]:
        return events or []

    def _scan_state(self, events, track_id, frame_id):
        """Devuelve 'reception' | 'scanning' | None para resaltar el receptor."""
        state = None
        for e in events:
            if e["receiver_track_id"] != track_id:
                continue
            if frame_id == e["frame_reception"]:
                return "reception"
            if e["window_start"] <= frame_id <= e["window_end"] and e.get("scan_count", 0) > 0:
                state = "scanning"
        return state

    # ----- A. video overlay -----
    def render_video(
        self,
        seq: SequenceData,
        orientation_df: pd.DataFrame,
        scanning_events: List[dict],
        output_path: str | Path,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Path:
        if not seq.image_paths:
            raise RuntimeError("No hay imagenes de la secuencia para el overlay de video.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        by_frame = self._index_orientation(orientation_df)
        w, h = seq.frame_size or (1920, 1080)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps or seq.fps, (w, h))
        frames = seq.frame_ids[:max_frames] if max_frames else seq.frame_ids
        for fid in frames:
            ipath = seq.image_paths.get(fid)
            if ipath is None:
                continue
            img = cv2.imread(str(ipath))
            if img is None:
                continue
            for row in by_frame.get(fid, []):
                color = _team_color(row.get("team_id"))
                x1, y1, x2, y2 = (int(row["bbox_x1"]), int(row["bbox_y1"]),
                                  int(row["bbox_x2"]), int(row["bbox_y2"]))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                state = self._scan_state(scanning_events, row["track_id"], fid)
                label = f"#{row['track_id']}"
                backend = _col(row, "keypoint_backend_used")
                if backend:
                    label += f" {backend}"
                # video overlay: orientacion en IMAGEN
                theta = _col(row, "theta_visual_smooth_img", "theta_visual_smooth")
                conf = row.get("orientation_confidence") or 0.0
                if theta is not None:
                    L = 25 + 35 * float(conf)
                    _arrow(img, cx, cy, math.radians(float(theta)), L, color, 2)
                if state == "scanning":
                    cv2.circle(img, (cx, cy), 22, _SCAN_COLOR, 2)
                    label += " SCAN"
                elif state == "reception":
                    cv2.circle(img, (cx, cy), 26, (40, 255, 40), 3)
                    label += " RECEPTION"
                cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
            ball = seq.ball(fid)
            if ball is not None and ball.valid:
                cv2.circle(img, (int(ball.x), int(ball.y)), 6, _BALL_COLOR, -1)
                cv2.circle(img, (int(ball.x), int(ball.y)), 7, (0, 0, 0), 1)
            writer.write(img)
        writer.release()
        return output_path

    # ----- B. minimap overlay -----
    def _draw_pitch(self, scale: int, margin: int) -> np.ndarray:
        L, W = self.pitch_length, self.pitch_width
        h = int(W * scale) + 2 * margin
        w = int(L * scale) + 2 * margin
        pitch = np.full((h, w, 3), (40, 110, 40), dtype=np.uint8)
        line = (230, 230, 230)
        cv2.rectangle(pitch, (margin, margin), (w - margin, h - margin), line, 2)
        cv2.line(pitch, (w // 2, margin), (w // 2, h - margin), line, 2)
        cv2.circle(pitch, (w // 2, h // 2), int(9.15 * scale), line, 2)
        return pitch

    def _to_minimap(self, X, Y, scale, margin):
        mx = int((X + self.pitch_length / 2) * scale) + margin
        my = int((Y + self.pitch_width / 2) * scale) + margin
        return mx, my

    def render_minimap(
        self,
        seq: SequenceData,
        orientation_df: pd.DataFrame,
        scanning_events: List[dict],
        output_path: str | Path,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        scale: int = 9,
        margin: int = 20,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        by_frame = self._index_orientation(orientation_df)
        base = self._draw_pitch(scale, margin)
        h, w = base.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps or seq.fps, (w, h))
        frames = seq.frame_ids[:max_frames] if max_frames else seq.frame_ids
        for fid in frames:
            pitch = base.copy()
            for row in by_frame.get(fid, []):
                X, Y = row.get("x_field"), row.get("y_field")
                if X is None or Y is None or (isinstance(X, float) and math.isnan(X)):
                    continue
                mx, my = self._to_minimap(X, Y, scale, margin)
                if not (0 <= mx < w and 0 <= my < h):
                    continue
                color = _team_color(row.get("team_id"))
                state = self._scan_state(scanning_events, row["track_id"], fid)
                cv2.circle(pitch, (mx, my), 5, color, -1)
                # flecha de orientacion en CANCHA (theta_visual_smooth_field)
                th = _col(row, "theta_visual_smooth_field", "theta_visual_smooth")
                if th is not None:
                    thr = math.radians(float(th))
                    self._draw_cone(pitch, mx, my, thr, color, scale)
                    _arrow(pitch, mx, my, thr, 18, color, 2)
                if state == "scanning":
                    cv2.circle(pitch, (mx, my), 9, _SCAN_COLOR, 2)
                elif state == "reception":
                    cv2.circle(pitch, (mx, my), 11, (40, 255, 40), 2)
            bxy = seq.ball_field_xy(fid) if seq.homography.available else None
            if bxy is not None:
                mx, my = self._to_minimap(bxy[0], bxy[1], scale, margin)
                if 0 <= mx < w and 0 <= my < h:
                    cv2.circle(pitch, (mx, my), 4, _BALL_COLOR, -1)
            writer.write(pitch)
        writer.release()
        return output_path

    def _draw_cone(self, img, mx, my, theta, color, scale, reach_m=18.0):
        half = self.fov / 2.0
        r = reach_m * scale
        pts = [(mx, my)]
        for a in np.linspace(theta - half, theta + half, 14):
            pts.append((int(mx + r * math.cos(a)), int(my - r * math.sin(a))))
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], color)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
