# core/scanning/scanning_detector.py
"""
Deteccion de eventos de scanning antes de una recepcion.

Definicion operacional: un jugador realiza scanning si, en la ventana previa a
recibir el balon, cambia de forma clara su orientacion para observar zonas del
campo distintas a la direccion inmediata del balon/carrera.

Para cada recepcion se toma la serie de orientacion suavizada del receptor en la
ventana [t_rec - before, t_rec - after] y se computan metricas de barrido.

Un "scan" es un giro sostenido: un tramo de orientacion en su mayoria monotono
cuyo cambio angular acumulado supera `scan_turn_threshold_deg`, dura
>= `min_scan_duration_frames` y tiene confianza media >= `min_scan_confidence`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .circular import circular_delta, orientation_entropy


@dataclass
class FrameOrientation:
    frame_id: int
    theta_smooth: Optional[float]
    confidence: float
    ball_relative_angle: Optional[float] = None


@dataclass
class ScanningMetrics:
    video_id: str
    event_id: str
    receiver_track_id: int
    frame_reception: int
    timestamp_reception: float
    window_start: int
    window_end: int
    scan_count: int
    max_orientation_change_deg: float
    mean_orientation_change_deg: float
    orientation_entropy: float
    looked_away_from_ball_count: int
    looked_towards_ball_count: int
    quality_score: float
    scan_label: int            # 1 si scan_count > 0 (etiqueta heuristica)

    def to_dict(self) -> dict:
        return asdict(self)


class ScanningDetector:
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.fps = float(config.get("fps", 25.0))
        self.before = float(config.get("scan_window_seconds_before", 3.0))
        self.after = float(config.get("scan_window_seconds_after", 0.2))
        self.turn_threshold = math.radians(
            float(config.get("scan_turn_threshold_deg", 45.0)))
        self.min_duration = int(config.get("min_scan_duration_frames", 4))
        self.min_conf = float(config.get("min_scan_confidence", 0.40))
        self.ball_cone = math.radians(float(config.get("ball_cone_deg", 35.0)))

    # -- segmentacion de giros sostenidos --
    def _count_scans(self, frames: Sequence[FrameOrientation]) -> Tuple[int, float]:
        """Devuelve (scan_count, max_accumulated_turn_deg)."""
        # solo frames validos y confiables, en orden
        seq = [f for f in frames
               if f.theta_smooth is not None and f.confidence >= self.min_conf]
        if len(seq) < 2:
            return 0, 0.0
        scan_count = 0
        max_turn = 0.0
        accum = 0.0          # giro acumulado con signo en el tramo actual
        length = 1
        confs = [seq[0].confidence]
        run_sign = 0
        for prev, cur in zip(seq[:-1], seq[1:]):
            d = circular_delta(cur.theta_smooth, prev.theta_smooth)
            s = 0 if abs(d) < 1e-6 else (1 if d > 0 else -1)
            if run_sign == 0 or s == 0 or s == run_sign:
                accum += d
                length += 1
                confs.append(cur.confidence)
                if s != 0:
                    run_sign = s
            else:
                # cambio de direccion: cerrar tramo
                if (abs(accum) >= self.turn_threshold
                        and length >= self.min_duration
                        and np.mean(confs) >= self.min_conf):
                    scan_count += 1
                    max_turn = max(max_turn, abs(accum))
                accum = d
                length = 2
                confs = [prev.confidence, cur.confidence]
                run_sign = s
        # cerrar ultimo tramo
        if (abs(accum) >= self.turn_threshold and length >= self.min_duration
                and np.mean(confs) >= self.min_conf):
            scan_count += 1
            max_turn = max(max_turn, abs(accum))
        return scan_count, math.degrees(max_turn)

    def evaluate_window(
        self,
        video_id: str,
        event_id: str,
        receiver_track_id: int,
        frame_reception: int,
        frames: Sequence[FrameOrientation],
    ) -> ScanningMetrics:
        window_start = int(frame_reception - self.before * self.fps)
        window_end = int(frame_reception - self.after * self.fps)
        win = [f for f in frames if window_start <= f.frame_id <= window_end]
        valid = [f for f in win if f.theta_smooth is not None]

        scan_count, max_turn = self._count_scans(win)

        # cambio medio frame-a-frame
        deltas = []
        seq = [f for f in valid if f.confidence >= self.min_conf]
        for prev, cur in zip(seq[:-1], seq[1:]):
            deltas.append(abs(circular_delta(cur.theta_smooth, prev.theta_smooth)))
        mean_change = math.degrees(float(np.mean(deltas))) if deltas else 0.0

        # entropia de orientacion
        angles = [f.theta_smooth for f in seq]
        entropy = orientation_entropy(angles) if angles else 0.0

        # mirando hacia / lejos del balon
        towards = away = 0
        for f in seq:
            if f.ball_relative_angle is None:
                continue
            if abs(circular_delta(f.theta_smooth, f.ball_relative_angle)) <= self.ball_cone:
                towards += 1
            else:
                away += 1

        # calidad: cobertura * confianza media
        win_frames = max(1, window_end - window_start + 1)
        coverage = len(valid) / win_frames
        mean_conf = float(np.mean([f.confidence for f in valid])) if valid else 0.0
        quality = float(np.clip(coverage * mean_conf, 0.0, 1.0))

        return ScanningMetrics(
            video_id=video_id,
            event_id=event_id,
            receiver_track_id=receiver_track_id,
            frame_reception=frame_reception,
            timestamp_reception=float(frame_reception / self.fps),
            window_start=window_start,
            window_end=window_end,
            scan_count=scan_count,
            max_orientation_change_deg=float(max_turn),
            mean_orientation_change_deg=float(mean_change),
            orientation_entropy=float(entropy),
            looked_away_from_ball_count=int(away),
            looked_towards_ball_count=int(towards),
            quality_score=quality,
            scan_label=int(scan_count > 0),
        )
