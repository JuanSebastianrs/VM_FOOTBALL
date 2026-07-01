# core/scanning_v2/supervised/weak_labeler.py
"""
Etiquetado DEBIL (weak supervision) para scanning V2.

Genera pseudo-etiquetas por reglas EXPLICITAS sobre las features por evento,
para poder entrenar los clasificadores cuando aun no existe GT humano a escala.

REGLAS DE HONESTIDAD (no negociables):
  - Las pseudo-etiquetas se escriben en un archivo SEPARADO
    (`scanning_windows_weak_*.csv`), NUNCA en el GT humano.
  - Cada fila lleva `label_source="weak_rules_v1"` y `notes` con la regla que
    disparo. El GT humano, cuando exista, SIEMPRE tiene prioridad.
  - Solo se etiquetan eventos INEQUIVOCOS segun las reglas (zona gris -> sin
    etiqueta). Un modelo entrenado asi es una DESTILACION suavizada de las
    reglas: sus metricas contra estas etiquetas miden consistencia, NO validez
    contra percepcion humana. Documentado en el reporte de entrenamiento.

Umbrales elegidos deliberadamente MAS estrictos que la heuristica V2 de runtime
(40 grados / min_valid_pose_ratio 0.4), para que la zona gris quede fuera.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

WEAK_LABEL_SOURCE = "weak_rules_v1"

# gates de calidad: sin señal de pose suficiente NO se pseudo-etiqueta
MIN_VALID_POSE_RATIO = 0.5
MIN_YAW_VALID_COUNT = 15
MIN_WINDOW_FRAMES = 30

# positivo inequivoco: movimiento de cabeza repetido y amplio
POS_MIN_TURNS_30 = 2
POS_ALT_MIN_TURNS_40 = 1
POS_ALT_MIN_DIR_CHANGES = 2
POS_MIN_YAW_RANGE = 60.0

# negativo inequivoco: cabeza esencialmente quieta
NEG_MAX_TURNS_20 = 0
NEG_MAX_YAW_RANGE = 25.0
NEG_MAX_MEAN_DELTA = 3.0


def weak_label_row(r: pd.Series) -> Tuple[Optional[int], float, str]:
    """(label 0/1/None, confidence 0..1, reason)."""
    vpr = float(r.get("valid_pose_ratio") or 0.0)
    nvalid = float(r.get("yaw_valid_count") or 0.0)
    nframes = float(r.get("n_frames") or 0.0)
    if vpr < MIN_VALID_POSE_RATIO or nvalid < MIN_YAW_VALID_COUNT \
            or nframes < MIN_WINDOW_FRAMES:
        return None, 0.0, "low_quality_window"

    t20 = float(r.get("sustained_turn_count_20deg") or 0.0)
    t30 = float(r.get("sustained_turn_count_30deg") or 0.0)
    t40 = float(r.get("sustained_turn_count_40deg") or 0.0)
    yaw_range = float(r.get("yaw_range_deg") or 0.0)
    mean_delta = float(r.get("yaw_mean_abs_delta_deg") or 0.0)
    dir_changes = float(r.get("yaw_num_direction_changes") or 0.0)

    pos_main = t30 >= POS_MIN_TURNS_30 and yaw_range >= POS_MIN_YAW_RANGE
    pos_alt = (t40 >= POS_ALT_MIN_TURNS_40 and dir_changes >= POS_ALT_MIN_DIR_CHANGES
               and yaw_range >= POS_MIN_YAW_RANGE)
    if pos_main or pos_alt:
        conf = float(np.clip(0.6 + 0.1 * t30 + 0.002 * (yaw_range - POS_MIN_YAW_RANGE),
                             0.6, 0.95))
        return 1, conf, ("pos_sustained_30x2" if pos_main else "pos_40_plus_altern")

    if t20 <= NEG_MAX_TURNS_20 and yaw_range <= NEG_MAX_YAW_RANGE \
            and mean_delta <= NEG_MAX_MEAN_DELTA:
        conf = float(np.clip(0.6 + 0.01 * (NEG_MAX_YAW_RANGE - yaw_range), 0.6, 0.9))
        return 0, conf, "neg_static_head"

    return None, 0.0, "gray_zone"


def generate_weak_labels(features: pd.DataFrame,
                         human_gt: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Features por evento -> DataFrame en formato de anotacion.

    Si `human_gt` trae etiquetas humanas validas (0/1) para un (video_id,
    event_id), ese evento se EXCLUYE del archivo debil (el humano manda y no se
    mezclan fuentes en un mismo archivo)."""
    human_keys = set()
    if human_gt is not None and "scan_label_gt" in human_gt.columns:
        lab = pd.to_numeric(human_gt["scan_label_gt"], errors="coerce")
        ok = human_gt[lab.isin([0, 1])]
        human_keys = set(zip(ok["video_id"].astype(str), ok["event_id"].astype(str)))

    rows, stats = [], {"pos": 0, "neg": 0, "gray_zone": 0,
                       "low_quality_window": 0, "human_override": 0}
    for _, r in features.iterrows():
        key = (str(r.get("video_id")), str(r.get("event_id")))
        if key in human_keys:
            stats["human_override"] += 1
            continue
        label, conf, reason = weak_label_row(r)
        if label is None:
            stats[reason] += 1
            continue
        stats["pos" if label == 1 else "neg"] += 1
        rows.append({
            "sample_id": len(rows),
            "event_id": r.get("event_id"),
            "video_id": r.get("video_id"),
            "receiver_track_id": r.get("receiver_track_id"),
            "scan_label_gt": int(label),
            "head_turn_count_gt": int(r.get("sustained_turn_count_30deg") or 0),
            "turn_direction_gt": "",
            "visibility": "high" if float(r.get("valid_pose_ratio") or 0) >= 0.75
                          else "medium",
            "confidence": round(conf, 3),
            "notes": f"{WEAK_LABEL_SOURCE}: {reason}",
            "label_source": WEAK_LABEL_SOURCE,
        })
    out = pd.DataFrame(rows)
    out.attrs["stats"] = stats
    return out


def write_weak_labels(df: pd.DataFrame, path: str) -> dict:
    p = Path(path)
    if p.exists():
        existing = pd.read_csv(p)
        if "label_source" in existing.columns and \
                (existing["label_source"] != WEAK_LABEL_SOURCE).any():
            raise ValueError(
                f"{path} contiene filas con otro label_source; no se sobreescribe "
                "un archivo que no es 100% debil.")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    stats = dict(df.attrs.get("stats", {}))
    stats.update({"path": str(p), "n_rows": int(len(df)),
                  "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                  "label_source": WEAK_LABEL_SOURCE})
    return stats
