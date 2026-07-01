# core/scanning/orientation_smoother.py
"""
Suavizado temporal de la orientacion por track_id.

Reduce el jitter frame-a-frame con una media exponencial CIRCULAR (sobre el
vector unitario del angulo). El alpha es adaptativo a la confianza: observaciones
mas confiables pesan mas. En frames invalidos el estado se atenua (coast) para
que la confianza del suavizado decaiga sin saltos en +-pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .circular import CircularEMA


@dataclass
class SmoothResult:
    track_id: int
    frame_id: int
    theta_visual_smooth: Optional[float]
    smooth_confidence: float


class OrientationSmoother:
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.alpha = float(config.get("smoothing_alpha", 0.35))
        self.alpha_min = float(config.get("smoothing_alpha_min", 0.10))
        self.decay = float(config.get("smoothing_decay", 0.85))
        self._ema: Dict[int, CircularEMA] = {}

    def _adaptive_alpha(self, confidence: float) -> float:
        """Interpola alpha_min..alpha en funcion de la confianza [0,1]."""
        c = max(0.0, min(1.0, confidence))
        return self.alpha_min + (self.alpha - self.alpha_min) * c

    def update(
        self,
        track_id: int,
        frame_id: int,
        theta: Optional[float],
        confidence: float = 0.0,
    ) -> SmoothResult:
        ema = self._ema.get(track_id)
        if ema is None:
            ema = CircularEMA(alpha=self.alpha, decay=self.decay)
            self._ema[track_id] = ema

        if theta is None or confidence <= 0.0:
            smoothed = ema.coast()
            # confianza del suavizado proporcional a la magnitud del vector EMA
            conf = ema.magnitude * 0.5
        else:
            smoothed = ema.update(theta, alpha=self._adaptive_alpha(confidence))
            conf = ema.magnitude * confidence

        return SmoothResult(
            track_id=track_id,
            frame_id=frame_id,
            theta_visual_smooth=smoothed,
            smooth_confidence=float(max(0.0, min(1.0, conf))),
        )

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._ema.clear()
        else:
            self._ema.pop(track_id, None)
