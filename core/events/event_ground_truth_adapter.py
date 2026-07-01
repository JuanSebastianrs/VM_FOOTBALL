# core/events/event_ground_truth_adapter.py
"""
Adaptador de eventos de ground truth externos (SoccerTrack v2, SoccerNet
Player-Centric Ball-Action Spotting, FOOTPASS, anotacion propia, ...).

No asume un formato unico: mapea columnas frecuentes al esquema canonico de
eventos del modulo.

REGLA CLAVE (corrige un bug de la V2): un `player_id` NO es necesariamente el
receptor (puede ser el ejecutor del pase, el dueno de la accion, etc.). Por eso
`player_id` ya NO se mapea automaticamente a `receiver_track_id`. Si el dataset
no trae un receptor explicito (`receiver`/`receiver_track_id`/`to_track_id`), el
evento se marca `rejection_reason="missing_receiver_in_gt"` y NO podra usarse
para extraer ventanas de scanning (el detector lo descarta y lo registra). La
validacion final (rol, existencia en game_state, no-arbitro) la hace el detector.

Uso:
    df = load_event_ground_truth(path, video_id)   # None si no hay archivo
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import EVENT_COLUMNS, normalize_source

# alias frecuentes -> columna canonica.
# NOTA: "player_id" NO esta aqui a proposito: no es seguro que sea el receptor.
_ALIASES = {
    "frame_event": "frame_reception",
    "frame": "frame_reception",
    "reception_frame": "frame_reception",
    "pass_frame": "frame_pass",
    "frame_pass_start": "frame_pass",
    "passer": "passer_track_id",
    "passer_id": "passer_track_id",
    "from_track_id": "passer_track_id",
    "receiver": "receiver_track_id",
    "receiver_id": "receiver_track_id",
    "receiver_track": "receiver_track_id",
    "to_track_id": "receiver_track_id",
    "team": "receiver_team_id",
    "action": "event_type",
    "label": "event_type",
}


def load_event_ground_truth(path: Optional[str], video_id: str) -> Optional[pd.DataFrame]:
    if not path or not Path(path).exists():
        return None
    raw = (pd.read_parquet(path) if str(path).endswith("parquet")
           else pd.read_csv(path))
    df = raw.rename(columns={k: v for k, v in _ALIASES.items() if k in raw.columns})
    if "video_id" in df.columns:
        df = df[df["video_id"].astype(str) == str(video_id)].copy()
    out = pd.DataFrame()
    for c in EVENT_COLUMNS:
        out[c] = df[c] if c in df.columns else None
    out["video_id"] = video_id

    # source validado contra SOURCE_PRIORITY (default ground_truth para datasets)
    if "source" in df.columns:
        out["source"] = df["source"].map(lambda v: normalize_source(v) if pd.notna(v)
                                          else "ground_truth")
    else:
        out["source"] = "ground_truth"

    if "event_type" not in df.columns or out["event_type"].isna().all():
        out["event_type"] = "reception"

    # receptor faltante: NO se inventa. Se marca para que el detector lo descarte.
    recv = pd.to_numeric(out["receiver_track_id"], errors="coerce")
    out["receiver_track_id"] = recv
    missing_recv = recv.isna()
    out["event_confidence"] = 1.0
    out.loc[missing_recv, "event_confidence"] = 0.0
    out.loc[missing_recv, "rejection_reason"] = "missing_receiver_in_gt"
    out["event_id"] = [f"{video_id}_gt_{i:04d}" for i in range(len(out))]
    return out.reset_index(drop=True)
