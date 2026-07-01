# core/scanning_v2/exporter.py
"""Exportacion de los artefactos del scanning V2 (parquet/csv)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from core.events.schema import EVENT_COLUMNS, REJECTION_COLUMNS
from core.game_state.schema import GAME_STATE_COLUMNS
from .schema import HEAD_POSE_COLUMNS, SCANNING_V2_COLUMNS


def _write(df: pd.DataFrame, path: Path, columns: Optional[Sequence[str]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        for c in columns:
            if c not in df.columns:
                df[c] = None
        df = df[list(columns)]
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


def export_game_state(df, out_dir) -> Path:
    return _write(df, Path(out_dir) / "game_state.parquet", GAME_STATE_COLUMNS)


def export_events(df, out_dir) -> Path:
    return _write(df, Path(out_dir) / "pass_reception_events.parquet", EVENT_COLUMNS)


def export_rejected(df, out_dir) -> Path:
    return _write(df, Path(out_dir) / "rejected_reception_candidates.csv", REJECTION_COLUMNS)


def export_head_pose(df, out_dir) -> Path:
    return _write(df, Path(out_dir) / "head_pose.parquet", HEAD_POSE_COLUMNS)


def export_scanning(df, out_dir) -> tuple:
    p = _write(df.copy(), Path(out_dir) / "scanning_events.parquet", SCANNING_V2_COLUMNS)
    c = _write(df.copy(), Path(out_dir) / "scanning_events.csv", SCANNING_V2_COLUMNS)
    return p, c


VISION_MAP_COLUMNS = [
    "event_id", "video_id", "receiver_track_id", "frame_reception",
    "vision_map_available", "used_orientation", "vision_map_confidence",
    "n_visible_teammates", "n_visible_opponents", "visible_ball",
    "observed_space_score",
]


def export_vision_map(df, out_dir) -> Path:
    return _write(df.copy() if df is not None else pd.DataFrame(),
                  Path(out_dir) / "vision_map_metrics.csv", VISION_MAP_COLUMNS)
