# core/scanning/exporter.py
"""
Exportacion de resultados de scanning a Parquet/CSV.

  - player_orientation : una fila por (frame, jugador) con la orientacion.
  - scanning_events    : una fila por recepcion con metricas de scanning.
  - reception_events   : una fila por recepcion detectada.

El formato (parquet/csv) se elige por la extension del path de salida.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

ORIENTATION_COLUMNS = [
    "video_id", "frame_id", "timestamp", "track_id", "team_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "x_field", "y_field", "vx", "vy",
    # orientacion: imagen vs cancha, cruda vs suavizada (todas en grados)
    "theta_visual_img", "theta_visual_field",
    "theta_visual_smooth_img", "theta_visual_smooth_field",
    "theta_source", "orientation_confidence",
    "head_angle", "torso_angle", "shoulder_angle", "run_angle",
    "ball_relative_angle_img", "ball_relative_angle_field",
    "pose_valid", "crop_quality",
    "keypoint_backend_used", "keypoint_backend_attempted",
]


def _deg(x: Optional[float]) -> Optional[float]:
    return None if x is None else math.degrees(x)


def _write(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


class Exporter:
    def __init__(self, output_dir: str | Path = "outputs/scanning"):
        self.output_dir = Path(output_dir)

    def export_orientation(self, records: Sequence[dict],
                           path: Optional[str | Path] = None) -> Path:
        path = path or self.output_dir / "player_orientation.parquet"
        df = pd.DataFrame(list(records))
        for col in ORIENTATION_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[ORIENTATION_COLUMNS]
        return _write(df, path)

    def export_scanning(self, events: Sequence[dict],
                        path: Optional[str | Path] = None) -> Path:
        path = path or self.output_dir / "scanning_events.parquet"
        df = pd.DataFrame(list(events))
        return _write(df, path)

    def export_receptions(self, events: Sequence[dict],
                          path: Optional[str | Path] = None) -> Path:
        path = path or self.output_dir / "reception_events.parquet"
        df = pd.DataFrame(list(events))
        return _write(df, path)


def orientation_record(
    video_id: str,
    ori,                       # OrientationResult
    smooth_img,                # SmoothResult (orientacion en imagen)
    smooth_field,              # SmoothResult | None (orientacion en cancha)
    player,                    # PlayerDetection
    fps: float,
    field_xy: Optional[tuple] = None,
    velocity: Optional[tuple] = None,
    crop_quality: float = 1.0,
    keypoint_backend_used: str = "",
    keypoint_backend_attempted: str = "",
) -> dict:
    """Construye una fila de player_orientation a partir de los objetos del pipeline.

    Los angulos se exportan en GRADOS (convencion matematica) para legibilidad.
    La orientacion se separa en marco IMAGEN y marco CANCHA (`_img` / `_field`),
    cada uno con su version cruda y suavizada.
    """
    vx, vy = (velocity if velocity is not None else (None, None))
    bx1, by1, bx2, by2 = player.bbox.tolist()
    field_smooth = smooth_field.theta_visual_smooth if smooth_field is not None else None
    return {
        "video_id": video_id,
        "frame_id": ori.frame_id,
        "timestamp": ori.frame_id / fps,
        "track_id": ori.track_id,
        "team_id": player.team_id,
        "bbox_x1": bx1, "bbox_y1": by1, "bbox_x2": bx2, "bbox_y2": by2,
        "x_field": field_xy[0] if field_xy else None,
        "y_field": field_xy[1] if field_xy else None,
        "vx": vx, "vy": vy,
        "theta_visual_img": _deg(ori.theta_visual_img),
        "theta_visual_field": _deg(ori.theta_visual_field),
        "theta_visual_smooth_img": _deg(smooth_img.theta_visual_smooth),
        "theta_visual_smooth_field": _deg(field_smooth),
        "theta_source": ori.theta_source,
        "orientation_confidence": ori.orientation_confidence,
        "head_angle": _deg(ori.head_angle),
        "torso_angle": _deg(ori.torso_angle),
        "shoulder_angle": _deg(ori.shoulder_angle),
        "run_angle": _deg(ori.run_angle),
        "ball_relative_angle_img": _deg(ori.ball_relative_angle),
        "ball_relative_angle_field": _deg(ori.ball_relative_angle_field),
        "pose_valid": ori.pose_valid,
        "crop_quality": crop_quality,
        "keypoint_backend_used": keypoint_backend_used,
        "keypoint_backend_attempted": keypoint_backend_attempted,
    }
