# core/game_state/schema.py
"""Esquema del game state unificado por (frame, track)."""

from __future__ import annotations

CANDIDATE_ROLES = ("player", "goalkeeper")
REFEREE_ROLE = "referee"

GAME_STATE_COLUMNS = [
    "video_id", "frame_id", "timestamp", "track_id", "role", "team_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_height", "bbox_width",
    "x_img", "y_img", "x_field", "y_field",
    "vx_field", "vy_field", "speed_field",
    "ball_x_img", "ball_y_img", "ball_x_field", "ball_y_field", "ball_speed",
    "distance_to_ball", "is_candidate_receiver", "is_referee",
]
