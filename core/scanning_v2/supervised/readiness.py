# core/scanning_v2/supervised/readiness.py
"""
Reporte de "readiness" para entrenamiento supervisado de scanning V2.

Resume, sin entrenar ni inventar nada, si hay evidencia suficiente para entrenar:
cuantas etiquetas/positivos faltan, si los clips existen, si las anotaciones estan
sincronizadas con los eventos V2, y el estado del modelo (entrenado o skipped).
Termina con la accion recomendada.

(head-turn / visual scanning APROXIMADO antes de recepcion; NO gaze real.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .annotation_validator import validate_annotations


def compute_readiness(video_id: str, v2_outputs: str, supervised_outputs: str,
                      annotations_path: Optional[str],
                      min_labeled: int = 20, min_positive: int = 3) -> dict:
    v2 = Path(v2_outputs)
    sup = Path(supervised_outputs)

    # eventos V2 actuales
    current_ids = set()
    ev_path = v2 / "pass_reception_events.parquet"
    if ev_path.exists():
        current_ids = set(pd.read_parquet(ev_path)["event_id"].astype(str).tolist())

    # validacion de anotaciones (clip_path relativo al annotation_pack)
    ann = (pd.read_csv(annotations_path)
           if annotations_path and Path(annotations_path).exists() else None)
    val = validate_annotations(ann, current_ids, video_id,
                               annotation_pack_dir=str(v2 / "annotation_pack"))

    # dataset manifest (si existe)
    manifest = _load_json(sup / "dataset" / "dataset_manifest.json")
    # estado del modelo
    model_meta = _load_json(sup / "models" / "model_metadata.json")
    model_pkl = (sup / "models" / "scanning_classifier.pkl").exists()
    training_report = (sup / "models" / "training_report.md").exists()

    # totales (del CSV) vs ELEGIBLES para entrenar (del manifest si existe)
    labeled_total = val["labeled_rows"]
    positive_total = val["positives"]
    negative_total = val["negatives"]
    if manifest:
        labeled_trainable = int(manifest.get("n_trainable", labeled_total))
        positive_trainable = int(manifest.get("n_trainable_positive", positive_total))
        negative_trainable = int(manifest.get("n_trainable_negative", negative_total))
        excl_vis = int(manifest.get("n_excluded_low_visibility", 0))
        excl_pose = int(manifest.get("n_excluded_low_pose_ratio", 0))
    else:
        labeled_trainable, positive_trainable, negative_trainable = (
            labeled_total, positive_total, negative_total)
        excl_vis = excl_pose = 0

    labels_needed = max(0, min_labeled - labeled_trainable)
    positives_needed = max(0, min_positive - positive_trainable)
    clips_exist = (len(val["missing_clip_file"]) == 0 and len(current_ids) > 0)

    model_status = "absent"
    if model_meta:
        model_status = model_meta.get("status") or ("trained" if model_pkl else "metadata_only")
    elif model_pkl:
        model_status = "trained"

    # can_train usa ELEGIBLES reales y exige sincronizacion del GT
    can_train = (labels_needed == 0 and positives_needed == 0
                 and negative_trainable >= 1 and val["synchronized"])

    if not val["synchronized"]:
        action = ("Anotaciones desincronizadas (duplicados/obsoletos/faltantes): "
                  "re-genera el annotation_pack y corrige el GT antes de entrenar.")
    elif can_train and not model_pkl:
        action = "Hay etiquetas elegibles suficientes: ejecuta train_scanning_model.py."
    elif can_train and model_pkl:
        action = "Modelo entrenado: ejecuta predict + evaluate_scanning_model.py."
    else:
        faltan = []
        if labels_needed:
            faltan.append(f"{labels_needed} etiquetas elegibles mas")
        if positives_needed:
            faltan.append(f"{positives_needed} positivos elegibles mas")
        if negative_trainable < 1:
            faltan.append("al menos 1 negativo")
        action = ("Anota mas ventanas en el annotation_pack ("
                  + ", ".join(faltan or ["mas evidencia"]) + ") y reintenta.")

    return {
        "video_id": video_id,
        "annotation": val,
        "min_labeled_samples": min_labeled,
        "min_positive_samples": min_positive,
        "labeled_total": labeled_total,
        "labeled_trainable": labeled_trainable,
        "positive_total": positive_total,
        "positive_trainable": positive_trainable,
        "negative_trainable": negative_trainable,
        "excluded_by_low_visibility": excl_vis,
        "excluded_by_pose_quality": excl_pose,
        # alias para compatibilidad de reporte
        "labeled_effective": labeled_trainable,
        "positives": positive_trainable,
        "negatives": negative_trainable,
        "labels_needed": labels_needed,
        "positives_needed": positives_needed,
        "clips_exist": clips_exist,
        "annotations_synchronized": val["synchronized"],
        "dataset_built": manifest is not None,
        "model_status": model_status,
        "training_report_exists": training_report,
        "can_train": bool(can_train),
        "recommended_action": action,
    }


def render_markdown(r: dict) -> str:
    a = r["annotation"]
    L = ["# Readiness supervisado — scanning V2", "",
         "> head-turn / visual scanning **APROXIMADO**. **No es gaze real.**", "",
         f"- video: `{r['video_id']}`",
         f"- eventos V2 sin anotar: {a['v2_event_ids_missing_in_annotations'] or '-'}",
         f"- anotaciones obsoletas: {a['stale_event_ids_not_in_v2'] or '-'}",
         f"- sincronizado: {'sí' if r['annotations_synchronized'] else 'NO'}",
         f"- clips presentes: {'sí' if r['clips_exist'] else 'NO'}",
         "",
         "## Etiquetas (totales vs elegibles para entrenar)",
         f"- labeled total: {r['labeled_total']}  | **trainable: {r['labeled_trainable']}** "
         f"/ min {r['min_labeled_samples']}",
         f"- positivos total: {r['positive_total']}  | **trainable: {r['positive_trainable']}** "
         f"/ min {r['min_positive_samples']}",
         f"- negativos trainable: {r['negative_trainable']}",
         f"- excluidos por baja visibilidad: {r['excluded_by_low_visibility']}  | "
         f"por baja calidad de pose: {r['excluded_by_pose_quality']}",
         f"- **faltan**: {r['labels_needed']} etiquetas elegibles, "
         f"{r['positives_needed']} positivos elegibles",
         "",
         "## Modelo",
         f"- dataset construido: {'sí' if r['dataset_built'] else 'no'}",
         f"- estado del modelo: **{r['model_status']}**",
         f"- ¿se puede entrenar ya?: {'sí' if r['can_train'] else 'NO'}",
         "",
         f"## Accion recomendada\n\n{r['recommended_action']}", ""]
    return "\n".join(L)


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
