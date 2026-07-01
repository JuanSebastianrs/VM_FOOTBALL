# core/scanning_v2/vision_map_wog.py
"""
FASE 8 — Capa de "espacio observado" inspirada en Wide Open Gazes.

NO usa pesos de WOG; es una capa metodologica posterior: dado el receptor, su
orientacion en cancha y el resto de jugadores, estima un campo de visibilidad
probabilistico (FOV con decaimiento angular y por distancia).

Usa `theta_head_field` si es fiable; si no, cae a `theta_body_field` y marca
`vision_map_confidence` baja. Si no hay NINGUNA orientacion en cancha, devuelve
`available=False` (no se inventa una orientacion).

Reusa la implementacion probabilistica ya existente en core.scanning.vision_map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from core.scanning.vision_map import VisionMap


@dataclass
class VisionResult:
    available: bool
    used_orientation: str            # "head" | "body" | "none"
    vision_map_confidence: float
    visible_teammates: list
    visible_opponents: list
    visible_ball: bool
    observed_space_score: float
    grid = None
    extent = None


class VisionMapWOG:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.enabled = bool(c.get("enabled", True))
        self._vm = VisionMap({
            "fov_deg": c.get("fov_deg", 120),
            "max_view_distance_m": c.get("max_distance_m", 35),
            "vision_grid_resolution_m": c.get("grid_resolution_m", 1.0),
            "peripheral_decay": c.get("angular_decay", 0.7) * 3.0,
            "distance_decay": c.get("distance_decay", 0.05),
            "occlusion_enabled": c.get("occlusion_enabled", False),
        })

    def compute(self, player_xy, theta_head_field, theta_body_field,
                other_players=(), ball_xy=None, player_team=None) -> VisionResult:
        if not self.enabled or player_xy is None:
            return VisionResult(False, "none", 0.0, [], [], False, 0.0)
        if theta_head_field is not None:
            theta, used, conf = theta_head_field, "head", 0.6
        elif theta_body_field is not None:
            theta, used, conf = theta_body_field, "body", 0.3
        else:
            return VisionResult(False, "none", 0.0, [], [], False, 0.0)

        r = self._vm.compute(player_xy, theta, other_players=other_players,
                             ball_xy=ball_xy, player_team=player_team)
        out = VisionResult(True, used, conf, r.visible_teammates, r.visible_opponents,
                           r.visible_ball, r.observed_space_score)
        out.grid = r.vision_grid
        out.extent = r.extent
        return out
