# core/scanning_v2/paths.py
"""
Layout canonico de outputs del proyecto.

Todo lo que pertenece a UNA secuencia vive bajo `outputs/<video_id>/`:

    outputs/<video_id>/
        <video_id>_detections.json, _trajectory.json, ...   (pipeline tactico)
        calibration_hinv.json
        scanning/                                            (scanning V2)
            pass_reception_events.parquet, head_pose.parquet,
            scanning_events.parquet/csv, vision_map_metrics.csv,
            game_state.parquet, windows/, event_clips/, ...

Los artefactos GLOBALES (no por secuencia) del entrenamiento del clasificador
de scanning viven en `outputs/scanning_training/` (dataset/, models/,
reports_gt/, predictions/).

Se mantiene compatibilidad de LECTURA con el layout antiguo
(`outputs/scanning_v2/<video_id>/...`): `resolve_scanning_dir` devuelve el
directorio nuevo si existe y si no cae al antiguo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

SCANNING_SUBDIR = "scanning"
SCANNING_MARKER = "pass_reception_events.parquet"

PathLike = Union[str, os.PathLike]


def scanning_dir(outputs_root: PathLike, video_id: str) -> Path:
    """Directorio canonico (layout nuevo) de scanning de una secuencia."""
    return Path(outputs_root) / str(video_id) / SCANNING_SUBDIR


def resolve_scanning_dir(outputs_root: PathLike, video_id: str) -> Path:
    """Directorio de scanning para LECTURA, con fallback al layout antiguo.

    Orden: `<root>/<vid>/scanning` -> `<root>/<vid>` (layout antiguo, cuando
    outputs_root apuntaba a `outputs/scanning_v2`). Si ninguno tiene outputs,
    devuelve el canonico (para mensajes de error coherentes)."""
    new = scanning_dir(outputs_root, video_id)
    if (new / SCANNING_MARKER).exists():
        return new
    legacy = Path(outputs_root) / str(video_id)
    if (legacy / SCANNING_MARKER).exists():
        return legacy
    return new


def discover_scanning_videos(outputs_root: PathLike) -> List[str]:
    """video_ids con outputs de scanning no vacios bajo `outputs_root`."""
    import pandas as pd

    root = Path(outputs_root)
    vids: List[str] = []
    if not root.is_dir():
        return vids
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        marker = resolve_scanning_dir(root, d.name) / SCANNING_MARKER
        if marker.is_file():
            try:
                if len(pd.read_parquet(marker)):
                    vids.append(d.name)
            except Exception:
                pass
    return vids
