# core/scanning_v2/head_pose_smoother.py
"""
FASE 6 — Suavizado temporal circular del yaw por evento/track.

El yaw es circular: se suaviza sobre el vector unitario [cos,sin] (EMA circular,
reusa core.scanning.circular.CircularEMA), nunca con promedios lineales. En
frames invalidos se hace "coast" hasta `max_gap_frames`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from core.scanning.circular import CircularEMA


class HeadYawSmoother:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.alpha = float(c.get("alpha", 0.35))
        self.min_conf = float(c.get("min_confidence", 0.35))
        self.max_gap = int(c.get("max_gap_frames", 5))

    def smooth(self, series: Sequence[Tuple[Optional[float], float]]
               ) -> List[Tuple[Optional[float], float]]:
        """series: [(yaw_deg|None, conf)] -> [(yaw_smooth_deg|None, conf_smooth)]."""
        ema = CircularEMA(alpha=self.alpha)
        out: List[Tuple[Optional[float], float]] = []
        gap = 0
        for yaw, conf in series:
            if yaw is None or conf < self.min_conf:
                gap += 1
                if ema.initialized and gap <= self.max_gap:
                    out.append((math.degrees(ema.coast()), ema.magnitude * 0.5))
                else:
                    out.append((None, 0.0))
            else:
                gap = 0
                s = ema.update(math.radians(float(yaw)))
                out.append((math.degrees(s), ema.magnitude * conf))
        return out
