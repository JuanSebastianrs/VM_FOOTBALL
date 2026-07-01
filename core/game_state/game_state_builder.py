# core/game_state/game_state_builder.py
"""
FASE 1 — Game state limpio.

Unifica por (frame, track) el tracking de jugadores, el balon, roles, equipos,
homografia, velocidades y distancia al balon. Es la unica capa que decide quien
es jugador valido (player/goalkeeper) y quien es arbitro (referee). El resto del
pipeline V2 NO vuelve a tocar esa decision.

Reutiliza el loader y la homografia ya probados de la V1 (core.scanning.data_io)
para no duplicar parsing de los JSON del pipeline tactico.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.scanning.data_io import SequenceData, load_sequence
from .schema import CANDIDATE_ROLES, GAME_STATE_COLUMNS, REFEREE_ROLE


def _speed(prev_xy, cur_xy, dt: float) -> Tuple[Optional[float], Optional[float], float]:
    if prev_xy is None or cur_xy is None or dt <= 0:
        return None, None, 0.0
    vx = (cur_xy[0] - prev_xy[0]) / dt
    vy = (cur_xy[1] - prev_xy[1]) / dt
    return vx, vy, float(np.hypot(vx, vy))


def build_game_state(
    video_id: str,
    detections_json: str,
    trajectory_json: Optional[str] = None,
    team_assignments_json: Optional[str] = None,
    calibration_json: Optional[str] = None,
    sequence_dir: Optional[str] = None,
    fps: float = 25.0,
) -> Tuple[pd.DataFrame, dict]:
    """Devuelve (game_state_df, validation_report)."""
    seq: SequenceData = load_sequence(
        video_id, detections_json, trajectory_json, team_assignments_json,
        calibration_json, sequence_dir=sequence_dir, fps=fps)
    has_homography = seq.homography.available
    dt = 1.0 / fps

    rows = []
    prev_field: Dict[int, Tuple[float, float]] = {}
    prev_ball_field: Optional[Tuple[float, float]] = None

    for fid in seq.frame_ids:
        ball = seq.ball(fid)
        ball_img = (ball.x, ball.y) if (ball and ball.valid) else (None, None)
        ball_field = seq.ball_field_xy(fid) if has_homography else None
        ball_speed = 0.0
        if ball_field is not None and prev_ball_field is not None:
            ball_speed = float(np.hypot(ball_field[0] - prev_ball_field[0],
                                        ball_field[1] - prev_ball_field[1]) / dt)
        if ball_field is not None:
            prev_ball_field = ball_field

        for p in seq.players(fid):
            tid = p.track_id
            # rol faltante -> "unknown" (NUNCA se asume "player")
            role = (p.role or "unknown").lower()
            is_ref = role == REFEREE_ROLE
            is_cand = role in CANDIDATE_ROLES   # solo player/goalkeeper
            fx, fy = p.foot
            field_xy = seq.player_field_xy(fid, tid) if has_homography else None
            vx, vy, spd = _speed(prev_field.get(tid), field_xy, dt)
            if field_xy is not None:
                prev_field[tid] = field_xy

            # distancia al balon: cancha si hay homografia, si no imagen
            dist = None
            if field_xy is not None and ball_field is not None:
                dist = float(np.hypot(field_xy[0] - ball_field[0],
                                      field_xy[1] - ball_field[1]))
            elif ball_img[0] is not None:
                dist = float(np.hypot(fx - ball_img[0], fy - ball_img[1]))

            rows.append({
                "video_id": video_id, "frame_id": fid, "timestamp": fid / fps,
                "track_id": tid, "role": role, "team_id": p.team_id,
                "bbox_x1": float(p.bbox[0]), "bbox_y1": float(p.bbox[1]),
                "bbox_x2": float(p.bbox[2]), "bbox_y2": float(p.bbox[3]),
                "bbox_height": p.height, "bbox_width": p.width,
                "x_img": fx, "y_img": fy,
                "x_field": field_xy[0] if field_xy else None,
                "y_field": field_xy[1] if field_xy else None,
                "vx_field": vx, "vy_field": vy, "speed_field": spd,
                "ball_x_img": ball_img[0], "ball_y_img": ball_img[1],
                "ball_x_field": ball_field[0] if ball_field else None,
                "ball_y_field": ball_field[1] if ball_field else None,
                "ball_speed": ball_speed,
                "distance_to_ball": dist,
                "is_candidate_receiver": bool(is_cand),
                "is_referee": bool(is_ref),
            })

    df = pd.DataFrame(rows, columns=GAME_STATE_COLUMNS)

    # ---- validaciones ----
    uniq = df.drop_duplicates("track_id")
    roles = uniq["role"].value_counts().to_dict()
    n_tracks = df["track_id"].nunique()
    # team_id ausente: NaN o el sentinela -2 (sin equipo asignado)
    no_team = sorted(df[df["team_id"].isna() | (df["team_id"] == -2)]
                     ["track_id"].unique().tolist())
    unknown_tracks = sorted(uniq[~uniq["is_candidate_receiver"] & ~uniq["is_referee"]]
                            ["track_id"].unique().tolist())
    frames_no_ball = int(sum(1 for fid in seq.frame_ids
                             if not (seq.ball(fid) and seq.ball(fid).valid)))
    report = {
        "video_id": video_id,
        "frames": len(seq.frame_ids),
        "tracks": int(n_tracks),
        "roles": {str(k): int(v) for k, v in roles.items()},
        "referee_track_ids": sorted(
            df[df["is_referee"]]["track_id"].unique().tolist()),
        "candidate_tracks": int(df[df["is_candidate_receiver"]]["track_id"].nunique()),
        # tracks que NO pueden ser receptores por rol unknown (ni player/gk ni referee)
        "unknown_role_track_ids": unknown_tracks,
        "tracks_without_team": no_team,
        "frames_without_ball": frames_no_ball,
        "has_homography": bool(has_homography),
        "player_frame_rows": int(len(df)),
    }
    return df, report
