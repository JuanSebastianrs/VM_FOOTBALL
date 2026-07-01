# core/scanning_v2/schema.py
"""Esquemas de salida del scanning V2."""

from __future__ import annotations

HEAD_POSE_COLUMNS = [
    "event_id", "video_id", "frame_id", "track_id",
    "yaw_raw", "pitch_raw", "roll_raw", "yaw_smooth", "yaw_confidence_smooth",
    "head_pose_confidence", "head_pose_backend_used", "head_pose_backend_attempted",
    "head_crop_valid", "head_crop_quality", "head_crop_source",
    "theta_body_img", "theta_body_field", "theta_head_img", "theta_head_field",
    "ball_relative_angle_field", "distance_to_ball", "time_to_reception",
]

SCANNING_V2_COLUMNS = [
    "event_id", "video_id", "receiver_track_id", "receiver_role",
    "frame_reception", "window_start", "window_end",
    "scan_label_pred", "head_turn_count",
    "max_yaw_change_deg", "mean_yaw_change_deg", "yaw_entropy",
    "look_away_from_ball_count", "look_back_to_ball_count",
    "valid_pose_ratio", "mean_head_pose_confidence", "quality_score", "source",
]
