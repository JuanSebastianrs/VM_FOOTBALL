# core/events/schema.py
"""Esquema de eventos de pase/recepcion (V2)."""

from __future__ import annotations

EVENT_COLUMNS = [
    "event_id", "video_id", "frame_pass", "frame_reception",
    "passer_track_id", "receiver_track_id", "passer_team_id", "receiver_team_id",
    "receiver_role", "event_type", "event_confidence", "source", "rejection_reason",
]

REJECTION_COLUMNS = [
    "frame_id", "candidate_track_id", "role", "team_id", "distance_to_ball", "reason",
]

# Prioridad de fuentes (mayor primero). Cualquier source fuera de esta lista se
# normaliza a "heuristic" (la mas baja) para no sobre-confiar en datos opacos.
SOURCE_PRIORITY = ["ground_truth", "manual", "model", "heuristic"]


def normalize_source(value) -> str:
    """Devuelve un source valido segun SOURCE_PRIORITY; si no, 'heuristic'."""
    import pandas as pd
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "heuristic"
    s = str(value).strip().lower()
    return s if s in SOURCE_PRIORITY else "heuristic"


REJECTION_REASONS = {
    "referee", "unknown_role", "ball_too_far", "unstable_track",
    "ambiguous_receiver", "low_confidence", "missing_ball", "missing_homography",
    "missing_receiver_in_gt", "invalid_attached_player",
}
