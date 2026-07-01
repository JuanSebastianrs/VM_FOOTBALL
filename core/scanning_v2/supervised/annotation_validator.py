# core/scanning_v2/supervised/annotation_validator.py
"""
Validacion de la anotacion humana de ventanas de scanning.

Comprueba que las etiquetas humanas son consistentes y que el CSV de anotacion
esta SINCRONIZADO con los eventos V2 actuales. NO inventa nada: solo reporta.

(head-turn / visual scanning APROXIMADO antes de recepcion; NO gaze real.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import pandas as pd

# valores permitidos (vacio = sin anotar, siempre permitido)
ALLOWED_SCAN_LABEL = {"", "0", "1"}
ALLOWED_TURN_DIRECTION = {"", "left", "right", "both", "unclear"}
ALLOWED_VISIBILITY = {"", "high", "medium", "low"}
ALLOWED_CONFIDENCE = {"", "high", "medium", "low"}


def _norm(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    # "1.0" -> "1"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s.lower() if not s.isdigit() else s


def validate_annotations(annotations: Optional[pd.DataFrame],
                         current_event_ids: Set[str],
                         video_id: str,
                         annotation_pack_dir: Optional[str] = None) -> dict:
    """Reporte de validacion. La clave logica es (video_id, event_id). Detecta
    duplicados, duplicados CONFLICTIVOS (mismo evento, label distinto), etiquetas
    invalidas (con indice de fila), desincronizacion con los eventos V2 y clips
    faltantes (via annotation_pack/clip_path con fallback)."""
    if annotations is None:
        annotations = pd.DataFrame(columns=["event_id", "video_id", "scan_label_gt"])
    df = annotations.copy()
    if "video_id" in df.columns:
        df = df[df["video_id"].astype(str) == str(video_id)]
    df = df.reset_index(drop=False)   # conserva el indice original en 'index'

    current = set(str(e) for e in current_event_ids)
    ann_ids = [str(e) for e in df["event_id"].tolist()] if "event_id" in df else []

    # ---- duplicados por (video_id, event_id) ----
    seen, dups = set(), set()
    for e in ann_ids:
        (dups.add(e) if e in seen else seen.add(e))
    duplicated = sorted(dups)
    # duplicados CONFLICTIVOS: misma clave, scan_label_gt distinto
    conflicting = []
    if duplicated and "scan_label_gt" in df.columns:
        for e in duplicated:
            labs = set(_norm(v) for v in df[df["event_id"].astype(str) == e]["scan_label_gt"])
            labs.discard("")          # vacio no cuenta como conflicto
            if len(labs) > 1:
                conflicting.append({"event_id": e, "labels": sorted(labs)})

    # ---- etiquetas invalidas (con indice de fila) + frame_start ----
    invalid, bad_frame_start = [], []
    for _, r in df.iterrows():
        eid = str(r.get("event_id"))
        idx = int(r.get("index"))
        problems = []
        if _norm(r.get("scan_label_gt")) not in ALLOWED_SCAN_LABEL:
            problems.append(f"scan_label_gt='{_norm(r.get('scan_label_gt'))}'")
        if _norm(r.get("turn_direction_gt")) not in ALLOWED_TURN_DIRECTION:
            problems.append(f"turn_direction_gt='{_norm(r.get('turn_direction_gt'))}'")
        if _norm(r.get("visibility")) not in ALLOWED_VISIBILITY:
            problems.append(f"visibility='{_norm(r.get('visibility'))}'")
        if _norm(r.get("confidence")) not in ALLOWED_CONFIDENCE:
            problems.append(f"confidence='{_norm(r.get('confidence'))}'")
        if problems:
            invalid.append({"row": idx, "event_id": eid, "problems": problems})
        fs = r.get("frame_start")
        try:
            if fs is not None and not pd.isna(fs) and float(fs) < 0:
                bad_frame_start.append({"row": idx, "event_id": eid, "frame_start": float(fs)})
        except (TypeError, ValueError):
            pass

    # ---- conteos (solo etiquetas validas 0/1) ----
    sl_norm = df["scan_label_gt"].map(_norm) if "scan_label_gt" in df else pd.Series([], dtype=str)
    labeled = int((sl_norm.isin(["0", "1"])).sum())
    positives = int((sl_norm == "1").sum())
    negatives = int((sl_norm == "0").sum())
    total = int(len(df))
    unlabeled = int(total - labeled)

    # ---- clips via annotation_pack/clip_path (+fallback clips/{event_id}_video.mp4) ----
    missing_clip_path, missing_clip_file = [], []
    pack = Path(annotation_pack_dir) if annotation_pack_dir else None
    for _, r in df.iterrows():
        eid = str(r.get("event_id"))
        cp = _norm(r.get("clip_path")) if "clip_path" in df.columns else ""
        rel = cp if cp else f"clips/{eid}_video.mp4"   # fallback
        if not cp:
            missing_clip_path.append(eid)
        if pack is not None and not (pack / rel).exists():
            missing_clip_file.append(eid)

    # ---- sincronizacion con eventos V2 ----
    stale = sorted(set(ann_ids) - current)
    missing_in_ann = sorted(current - set(ann_ids))
    synchronized = (not stale and not missing_in_ann and not duplicated
                    and not conflicting)

    return {
        "video_id": video_id,
        "key_columns": ["video_id", "event_id"],
        "total_rows": total,
        "labeled_rows": labeled,
        "unlabeled_rows": unlabeled,
        "positives": positives,
        "negatives": negatives,
        "invalid_labels": invalid,
        "n_invalid_labels": len(invalid),
        "invalid_frame_start": bad_frame_start,
        "duplicated_event_ids": duplicated,
        "conflicting_duplicates": conflicting,
        "missing_clip_path": sorted(set(missing_clip_path)),
        "missing_clip_file": sorted(set(missing_clip_file)),
        "missing_clips": sorted(set(missing_clip_file)),   # alias compat
        "stale_event_ids_not_in_v2": stale,
        "v2_event_ids_missing_in_annotations": missing_in_ann,
        "synchronized": synchronized,
    }


def render_markdown(rep: dict) -> str:
    L = ["# Validacion de anotaciones — scanning V2", "",
         "> head-turn / visual scanning **APROXIMADO**. **No es gaze real.**", "",
         f"- video: `{rep['video_id']}`",
         f"- total: {rep['total_rows']}  | labeled: {rep['labeled_rows']}  | "
         f"unlabeled: {rep['unlabeled_rows']}",
         f"- positivos: {rep['positives']}  | negativos: {rep['negatives']}",
         f"- etiquetas invalidas: {rep['n_invalid_labels']}",
         f"- frame_start invalido (<0): {len(rep.get('invalid_frame_start', []))}",
         f"- event_id duplicados: {rep['duplicated_event_ids'] or '-'}",
         f"- duplicados CONFLICTIVOS: {rep['conflicting_duplicates'] or '-'}",
         f"- clip_path vacio: {rep['missing_clip_path'] or '-'}",
         f"- archivo de clip faltante: {rep['missing_clip_file'] or '-'}",
         f"- anotaciones obsoletas (no en V2): {rep['stale_event_ids_not_in_v2'] or '-'}",
         f"- eventos V2 sin anotar: {rep['v2_event_ids_missing_in_annotations'] or '-'}",
         f"- **sincronizado**: {'sí' if rep['synchronized'] else 'NO'}", ""]
    if rep["invalid_labels"]:
        L.append("## Etiquetas invalidas")
        for it in rep["invalid_labels"]:
            L.append(f"- fila {it['row']} `{it['event_id']}`: {', '.join(it['problems'])}")
    return "\n".join(L)
