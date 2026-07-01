# core/scanning/reception_detector.py
"""
Deteccion de recepciones de balon.

Si el pipeline ya provee eventos de recepcion, se consumen tal cual. Si no, se
deriva un detector heuristico a partir de la posesion frame-a-frame:

  1. Posesor por frame  = trajectory.attached_player_id si existe, si no el
     jugador mas cercano al balon dentro de un umbral.
  2. Posesion confirmada = el mismo posesor durante >= min_possession_frames.
  3. Recepcion           = transicion de posesion confirmada hacia un nuevo
     jugador; el evento se emite en el primer frame de esa posesion.

Trabaja en coordenadas de cancha (metros) cuando hay homografia; si no, cae a
distancia en pixeles.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from .data_io import SequenceData


@dataclass
class ReceptionEvent:
    event_id: str
    receiver_track_id: int
    frame_reception: int
    timestamp_reception: float
    team_id: Optional[int]
    ball_position_field: Optional[List[float]]
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


class ReceptionDetector:
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.fps = float(config.get("fps", 25.0))
        self.max_ball_distance_m = float(config.get("max_ball_distance_m", 2.0))
        self.max_ball_distance_px = float(config.get("max_ball_distance_px", 80.0))
        self.min_possession_frames = int(config.get("min_possession_frames", 3))
        self.use_attached = bool(config.get("use_attached_player_id", True))

    # -- posesor por frame --
    def _possessor(self, seq: SequenceData, frame_id: int):
        ball = seq.ball(frame_id)
        if ball is None or not ball.valid:
            return None, 0.0
        # 1) attached_player_id (mas fiable)
        if self.use_attached and ball.attached_player_id is not None:
            tid = int(ball.attached_player_id)
            if seq.player(frame_id, tid) is not None:
                return tid, 1.0
        # 2) jugador mas cercano
        ball_field = seq.ball_field_xy(frame_id) if seq.homography.available else None
        best_tid, best_d = None, np.inf
        for p in seq.players(frame_id):
            if ball_field is not None:
                pf = seq.player_field_xy(frame_id, p.track_id)
                if pf is None:
                    continue
                d = float(np.hypot(pf[0] - ball_field[0], pf[1] - ball_field[1]))
                thr = self.max_ball_distance_m
            else:
                fx, fy = p.foot
                d = float(np.hypot(fx - ball.x, fy - ball.y))
                thr = self.max_ball_distance_px
            if d < best_d:
                best_d, best_tid = d, p.track_id
        if best_tid is not None and best_d <= thr:
            conf = float(np.clip(1.0 - best_d / thr, 0.1, 1.0))
            return best_tid, conf
        return None, 0.0

    def detect(self, seq: SequenceData) -> List[ReceptionEvent]:
        # posesor crudo por frame
        raw: Dict[int, tuple] = {}
        for fid in seq.frame_ids:
            raw[fid] = self._possessor(seq, fid)

        # confirmar posesion: runs de >= min_possession_frames del mismo posesor
        events: List[ReceptionEvent] = []
        prev_confirmed: Optional[int] = None
        i = 0
        frames = seq.frame_ids
        n = len(frames)
        while i < n:
            tid, _ = raw[frames[i]]
            if tid is None:
                i += 1
                continue
            # longitud del run
            j = i
            while j < n and raw[frames[j]][0] == tid:
                j += 1
            run_len = j - i
            if run_len >= self.min_possession_frames and tid != prev_confirmed:
                fid0 = frames[i]
                confs = [raw[frames[k]][1] for k in range(i, j)]
                ball_field = (seq.ball_field_xy(fid0)
                              if seq.homography.available else None)
                events.append(ReceptionEvent(
                    event_id=f"{seq.video_id}_rcp_{len(events):04d}",
                    receiver_track_id=int(tid),
                    frame_reception=int(fid0),
                    timestamp_reception=float(fid0 / self.fps),
                    team_id=seq.team_by_track.get(tid),
                    ball_position_field=(list(ball_field) if ball_field else None),
                    confidence=float(np.mean(confs)),
                ))
                prev_confirmed = tid
            i = j
        return events
