# core/events/possession_estimator.py
"""
Estimador de posesion por frame — SOLO sobre receptores candidatos.

Regla central (corrige el bug de la V1): NUNCA se considera a un arbitro
(is_referee) ni a un track no candidato. El balon "cerca de un arbitro" no
genera posesion.

Por frame devuelve PossessionFrame con:
  track_id (candidato poseedor o None), confidence, distance, ambiguous
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PossessionFrame:
    frame_id: int
    track_id: Optional[int]
    confidence: float
    distance: Optional[float]
    ambiguous: bool
    referee_was_closest: bool   # diagnostico: un arbitro estaba mas cerca que el poseedor
    source: str = "distance"    # "attached" | "distance" | "none"
    invalid_attached: bool = False  # habia attached_player_id pero no paso validacion


class PossessionEstimator:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.max_dist = float(c.get("max_ball_receiver_distance_m", 2.0))
        self.ambiguous_margin = float(c.get("ambiguous_margin_m", 0.6))
        self.use_attached = bool(c.get("use_attached_player_id", True))
        # distancia maxima para confiar en attached_player_id (mas estricta)
        self.attached_max_dist = float(c.get("attached_max_distance_m", self.max_dist))

    def per_frame(self, gs: pd.DataFrame,
                  attached_by_frame: Optional[Dict[int, int]] = None) -> Dict[int, PossessionFrame]:
        attached_by_frame = attached_by_frame or {}
        out: Dict[int, PossessionFrame] = {}
        for fid, g in gs.groupby("frame_id"):
            fid = int(fid)
            ball_ok = g["distance_to_ball"].notna().any()
            cand = g[g["is_candidate_receiver"] & g["distance_to_ball"].notna()]
            ref = g[g["is_referee"] & g["distance_to_ball"].notna()]

            ref_closest = False
            if not cand.empty and not ref.empty:
                ref_closest = ref["distance_to_ball"].min() < cand["distance_to_ball"].min()
            elif cand.empty and not ref.empty:
                ref_closest = True

            if cand.empty or not ball_ok:
                # arbitro/unknown cerca del balon NO genera posesion
                out[fid] = PossessionFrame(fid, None, 0.0, None, False, ref_closest,
                                           source="none")
                continue

            cand = cand.sort_values("distance_to_ball")
            d0 = float(cand["distance_to_ball"].iloc[0])
            t0 = int(cand["track_id"].iloc[0])
            ambiguous = (len(cand) > 1 and
                         float(cand["distance_to_ball"].iloc[1]) - d0 < self.ambiguous_margin)

            # 1) attached_player_id: solo si es candidato, existe en el frame,
            #    esta dentro de la distancia y NO hay ambiguedad fuerte.
            att = attached_by_frame.get(fid)
            if self.use_attached and att is not None:
                row = cand[cand["track_id"] == int(att)]
                if not row.empty:
                    d_att = float(row["distance_to_ball"].iloc[0])
                    if d_att <= self.attached_max_dist and not ambiguous:
                        out[fid] = PossessionFrame(fid, int(att), 1.0, d_att,
                                                   False, ref_closest, source="attached")
                        continue
                    # attached presente pero no valido -> se marca y se cae a distancia
                    out[fid] = self._distance_frame(fid, d0, t0, ambiguous, ref_closest,
                                                    invalid_attached=True)
                    continue
                else:
                    # attached apunta a un track no-candidato (arbitro/unknown/ausente)
                    out[fid] = self._distance_frame(fid, d0, t0, ambiguous, ref_closest,
                                                    invalid_attached=True)
                    continue

            out[fid] = self._distance_frame(fid, d0, t0, ambiguous, ref_closest)
        return out

    def _distance_frame(self, fid, d0, t0, ambiguous, ref_closest,
                        invalid_attached: bool = False) -> PossessionFrame:
        if d0 > self.max_dist:
            return PossessionFrame(fid, None, 0.0, d0, ambiguous, ref_closest,
                                   source="distance", invalid_attached=invalid_attached)
        conf = float(np.clip(1.0 - d0 / self.max_dist, 0.1, 1.0))
        return PossessionFrame(fid, t0, conf, d0, ambiguous, ref_closest,
                               source="distance", invalid_attached=invalid_attached)
