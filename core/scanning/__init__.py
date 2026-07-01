# core/scanning/__init__.py
"""
Modulo de estimacion de *visual scanning* en futbol broadcast.

Estima la orientacion visual APROXIMADA de cada jugador (a partir de cabeza,
hombros, torso, direccion de carrera y contexto temporal) y detecta eventos de
scanning antes de recibir el balon. NO realiza eye-tracking exacto: en video
broadcast los ojos rara vez son visibles con suficiente resolucion.

Pipeline:
  frame/crop -> keypoints -> orientacion -> suavizado temporal
             -> recepciones -> scanning -> export -> visualizacion

Las clases pesadas (que dependen de torch/ultralytics) se importan de forma
perezosa via `load_config` / los submodulos para mantener el import barato.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .circular import (
    CircularEMA,
    angle_to_bin,
    bin_to_angle,
    circular_delta,
    circular_mean,
    orientation_entropy,
    wrap_angle,
)
from .data_io import (
    BallObservation,
    HomographyLookup,
    PlayerDetection,
    SequenceData,
    load_sequence,
)

__all__ = [
    "load_config",
    "CircularEMA",
    "angle_to_bin",
    "bin_to_angle",
    "circular_delta",
    "circular_mean",
    "orientation_entropy",
    "wrap_angle",
    "BallObservation",
    "HomographyLookup",
    "PlayerDetection",
    "SequenceData",
    "load_sequence",
]

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "scanning.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Carga configs/scanning.yaml (o un path explicito) como dict."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
