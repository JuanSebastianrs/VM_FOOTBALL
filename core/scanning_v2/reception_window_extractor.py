# core/scanning_v2/reception_window_extractor.py
"""
FASE 3 — Ventanas temporales antes de la recepcion.

Para cada evento define [frame_reception - before, frame_reception - after] y
reune las filas del game_state del receptor en esa ventana. Opcionalmente escribe
windows/{event_id}/metadata.json (+ carpetas frames/, player_crops/, head_crops/
que llena el pipeline).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class WindowSpec:
    event: dict
    window_start: int
    window_end: int
    frames: List[int] = field(default_factory=list)


def _json_safe(v):
    """Convierte a un valor JSON valido: NaN/NA/inf -> None; numpy -> python."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):          # numpy scalar
        try:
            return v.item()
        except Exception:
            return v
    return v


class ReceptionWindowExtractor:
    def __init__(self, config: dict):
        c = config or {}
        self.fps = float(c.get("fps", 25.0))
        self.before = float(c.get("seconds_before_reception", 3.0))
        self.after = float(c.get("seconds_after_reception", 0.2))

    def extract(self, events: pd.DataFrame, gs: pd.DataFrame) -> List[WindowSpec]:
        specs = []
        gframes = set(int(f) for f in gs["frame_id"].unique().tolist())
        first_frame = min(gframes) if gframes else 0
        for ev in events.to_dict("records"):
            f_rec = int(ev["frame_reception"])
            # clamp: la ventana nunca empieza antes del primer frame disponible
            ws = max(first_frame, int(f_rec - self.before * self.fps))
            we = int(f_rec - self.after * self.fps)
            if we < ws:
                we = ws
            tid = int(ev["receiver_track_id"])
            rows = gs[(gs["track_id"] == tid) & (gs["frame_id"] >= ws)
                      & (gs["frame_id"] <= we)]
            frames = sorted(int(f) for f in rows["frame_id"].unique() if f in gframes)
            specs.append(WindowSpec(ev, ws, we, frames))
        return specs

    @staticmethod
    def write_metadata(out_root: Path, spec: WindowSpec) -> Path:
        d = out_root / "windows" / str(spec.event["event_id"])
        (d / "frames").mkdir(parents=True, exist_ok=True)
        (d / "player_crops").mkdir(parents=True, exist_ok=True)
        (d / "head_crops").mkdir(parents=True, exist_ok=True)
        meta = {
            "event_id": spec.event["event_id"],
            "video_id": spec.event["video_id"],
            "receiver_track_id": _json_safe(spec.event["receiver_track_id"]),
            "receiver_role": _json_safe(spec.event.get("receiver_role")),
            "team_id": _json_safe(spec.event.get("receiver_team_id")),
            "frame_pass": _json_safe(spec.event.get("frame_pass")),
            "frame_reception": int(spec.event["frame_reception"]),
            "window_start": int(spec.window_start),
            "window_end": int(spec.window_end),
            "n_frames": len(spec.frames),
            "source": _json_safe(spec.event.get("source")),
            "event_confidence": _json_safe(spec.event.get("event_confidence")),
        }
        # allow_nan=False garantiza JSON valido (sin NaN/Infinity)
        (d / "metadata.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
        return d
