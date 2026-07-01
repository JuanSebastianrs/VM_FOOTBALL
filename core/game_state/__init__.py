# core/game_state/__init__.py
"""Game state limpio (V2): unifica tracking, balon, roles, equipos y homografia."""
from .game_state_builder import build_game_state
from .schema import CANDIDATE_ROLES, GAME_STATE_COLUMNS, REFEREE_ROLE

__all__ = ["build_game_state", "GAME_STATE_COLUMNS", "CANDIDATE_ROLES", "REFEREE_ROLE"]
