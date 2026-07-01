# core/scanning/vision_map.py
"""
Vision map probabilistico (opcional).

Dada la posicion en cancha de un jugador y su orientacion visual estimada,
genera un mapa de "cuanto puede ver" cada zona del campo. No es un cono binario:
combina decaimiento periferico (angular) y decaimiento por distancia, con
oclusion opcional por otros jugadores.

Convencion de coordenadas
-------------------------
Posiciones en cancha CENTRADAS (origen en el centro), en metros, con Y hacia
abajo (igual que data_io / el minimapa). `theta_field` viene en convencion
matematica (+y arriba), asi que el angulo de cada celda se calcula negando la
componente Y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VisionMapResult:
    vision_grid: np.ndarray                 # (rows, cols) pesos [0,1]
    extent: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax) centrado
    visible_teammates: List[int] = dc_field(default_factory=list)
    visible_opponents: List[int] = dc_field(default_factory=list)
    visible_ball: bool = False
    observed_space_score: float = 0.0


class VisionMap:
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.fov = math.radians(float(config.get("fov_deg", 120.0)))
        self.central_weight = float(config.get("central_weight", 1.0))
        self.peripheral_decay = float(config.get("peripheral_decay", 2.0))
        self.distance_decay = float(config.get("distance_decay", 0.04))
        self.max_view = float(config.get("max_view_distance_m", 40.0))
        self.occlusion_enabled = bool(config.get("occlusion_enabled", True))
        self.res = float(config.get("vision_grid_resolution_m", 1.0))
        self.pitch_length = float(config.get("pitch_length", 105.0))
        self.pitch_width = float(config.get("pitch_width", 68.0))
        self.visible_threshold = 0.1

    def _weight(self, dx: float, dy: float) -> float:
        """Peso de visibilidad de un punto relativo (ya en marco math, +y arriba)."""
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return self.central_weight
        if d > self.max_view:
            return 0.0
        cell_angle = math.atan2(dy, dx)
        return self._weight_polar(cell_angle, d)

    def _weight_polar(self, cell_angle: float, d: float, theta: float = 0.0) -> float:
        offset = abs((cell_angle - theta + math.pi) % (2 * math.pi) - math.pi)
        half = self.fov / 2.0
        if offset > half:
            return 0.0
        ang_term = math.exp(-self.peripheral_decay * (offset / half) ** 2)
        dist_term = math.exp(-self.distance_decay * d)
        return self.central_weight * ang_term * dist_term

    def _entity_visibility(
        self, px: float, py: float, theta: float,
        ex: float, ey: float,
        blockers: Sequence[Tuple[float, float]] = (),
    ) -> float:
        dx, dy = ex - px, -(ey - py)        # negar Y -> marco math
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return self.central_weight
        if d > self.max_view:
            return 0.0
        cell_angle = math.atan2(dy, dx)
        w = self._weight_polar(cell_angle, d, theta)
        if w <= 0.0:
            return 0.0
        if self.occlusion_enabled and blockers:
            # oclusion simple: si un blocker esta mas cerca y casi alineado
            for bx, by in blockers:
                bdx, bdy = bx - px, -(by - py)
                bd = math.hypot(bdx, bdy)
                if bd < 1e-6 or bd >= d:
                    continue
                bang = math.atan2(bdy, bdx)
                if abs((bang - cell_angle + math.pi) % (2 * math.pi) - math.pi) < math.radians(4.0):
                    return 0.0
        return w

    def compute(
        self,
        player_xy: Tuple[float, float],
        theta_field: float,
        other_players: Sequence[Tuple[int, float, float, Optional[int]]] = (),
        ball_xy: Optional[Tuple[float, float]] = None,
        player_team: Optional[int] = None,
    ) -> VisionMapResult:
        px, py = player_xy
        L, W = self.pitch_length, self.pitch_width
        xs = np.arange(-L / 2, L / 2 + self.res, self.res)
        ys = np.arange(-W / 2, W / 2 + self.res, self.res)
        grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
        for iy, cy in enumerate(ys):
            for ix, cx in enumerate(xs):
                dx, dy = cx - px, -(cy - py)   # negar Y -> marco math
                grid[iy, ix] = self._weight_polar(
                    math.atan2(dy, dx), math.hypot(dx, dy), theta_field) \
                    if math.hypot(dx, dy) <= self.max_view else 0.0

        blockers = [(x, y) for (_tid, x, y, _t) in other_players]
        visible_teammates, visible_opponents = [], []
        for tid, x, y, team in other_players:
            others = [(bx, by) for (btid, bx, by, _bt) in other_players if btid != tid]
            v = self._entity_visibility(px, py, theta_field, x, y, others)
            if v >= self.visible_threshold:
                if player_team is not None and team is not None:
                    (visible_teammates if team == player_team
                     else visible_opponents).append(int(tid))
                else:
                    visible_teammates.append(int(tid))

        visible_ball = False
        if ball_xy is not None:
            visible_ball = self._entity_visibility(
                px, py, theta_field, ball_xy[0], ball_xy[1], blockers) >= self.visible_threshold

        observed = float(grid.sum() / grid.size) if grid.size else 0.0
        return VisionMapResult(
            vision_grid=grid,
            extent=(-L / 2, L / 2, -W / 2, W / 2),
            visible_teammates=visible_teammates,
            visible_opponents=visible_opponents,
            visible_ball=visible_ball,
            observed_space_score=observed,
        )
