# core/scanning_v2/supervised/schema.py
"""
Esquema del dataset supervisado de scanning V2.

Distingue tres grupos de columnas:
  - IDENTITY/META : identifican el sample (event_id, video, rol, fuente). NO son
                    features numericas de entrenamiento por defecto.
  - LABEL         : etiquetas/anotacion humana (scan_label_gt, ...). NUNCA features.
  - FEATURES      : columnas numericas derivadas de las senales aproximadas.

Reglas duras (para no contaminar el entrenamiento):
  - `scan_label_gt`, `confidence`, `visibility`, `notes`, `turn_direction_gt`,
    `head_turn_count_gt` son ETIQUETAS/eval -> nunca features.
  - `heuristic_scan_label_pred` (la prediccion heuristica V2) solo entra como
    feature si la config lo permite explicitamente.
"""

from __future__ import annotations

from typing import List

import pandas as pd

FEATURE_VERSION = "scanning_v2_features_v1"

# columnas que identifican el sample (no se entrenan como numeros)
ID_COLUMNS = ["event_id", "video_id", "receiver_track_id"]
META_COLUMNS = ["receiver_role", "event_source", "feature_version"]

# etiquetas / anotacion humana
LABEL_COLUMNS = [
    "event_id", "video_id", "scan_label_gt", "head_turn_count_gt",
    "turn_direction_gt", "visibility", "confidence", "notes",
]

# columnas que JAMAS pueden ser feature de entrenamiento
FORBIDDEN_FEATURES = {
    "scan_label_gt", "head_turn_count_gt", "turn_direction_gt", "visibility",
    "confidence", "notes", "scan_label_pred", "scan_label_model",
    "scan_probability_model",
    # derivadas de la etiqueta/gates (fuga de informacion) o de claves de merge
    "training_eligible", "sample_id", "frame_start", "frame_end", "team_id",
}

# la prediccion heuristica V2, opcional como feature (default: NO)
HEURISTIC_PRED_COLUMN = "heuristic_scan_label_pred"

LABEL_COLUMN = "scan_label_gt"


def training_feature_columns(df: pd.DataFrame,
                             include_heuristic_pred: bool = False) -> List[str]:
    """Columnas numericas usables como features de entrenamiento.

    Excluye identidad, meta, etiquetas y columnas prohibidas. Incluye la
    prediccion heuristica SOLO si `include_heuristic_pred=True`.
    """
    excluded = set(ID_COLUMNS) | set(META_COLUMNS) | set(LABEL_COLUMNS) | FORBIDDEN_FEATURES
    if not include_heuristic_pred:
        excluded.add(HEURISTIC_PRED_COLUMN)
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if c.endswith("_lab") or c.endswith("_feat"):   # columnas de merge
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


PREDICTION_COLUMNS = [
    "event_id", "video_id", "receiver_track_id",
    "scan_label_model", "scan_probability_model", "model_confidence",
    "model_version", "feature_version", "prediction_source",
]
