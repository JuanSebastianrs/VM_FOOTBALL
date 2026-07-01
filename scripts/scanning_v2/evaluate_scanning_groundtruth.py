# scripts/scanning_v2/evaluate_scanning_groundtruth.py
"""
FASE 11 — Evaluacion contra ground truth humano.

Compara scanning_events.parquet (scan_label_pred) con
data/annotations/scanning_windows_gt.csv (scan_label_gt). Si NO hay GT humano
(columna vacia o archivo ausente), NO inventa metricas: reporta que falta.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true))}


def _fp_fn(m):
    """event_id de falsos positivos / falsos negativos."""
    fp = m[(m["scan_label_gt"] == 0) & (m["scan_label_pred"] == 1)]["event_id"].tolist()
    fn = m[(m["scan_label_gt"] == 1) & (m["scan_label_pred"] == 0)]["event_id"].tolist()
    return [str(x) for x in fp], [str(x) for x in fn]


def _by(m, col):
    if col not in m.columns:
        return {}
    return {str(v): _metrics(g["scan_label_gt"], g["scan_label_pred"])
            for v, g in m.groupby(col)}


def _attach_head_pose(m, head_pose_path):
    """Agrega por evento el backend dominante de head pose y la calidad media de
    crop, para poder reportar metricas por backend y por crop_quality."""
    if not head_pose_path or not Path(head_pose_path).exists():
        return m
    hp = (pd.read_parquet(head_pose_path) if head_pose_path.endswith("parquet")
          else pd.read_csv(head_pose_path))
    if "event_id" not in hp.columns:
        return m
    agg = []
    for eid, g in hp.groupby("event_id"):
        backend = (g["head_pose_backend_used"].mode().iloc[0]
                   if "head_pose_backend_used" in g and not g["head_pose_backend_used"].mode().empty
                   else "unknown")
        q = float(g["head_crop_quality"].mean()) if "head_crop_quality" in g else 0.0
        bucket = "high" if q >= 0.5 else ("medium" if q >= 0.2 else "low")
        agg.append({"event_id": eid, "head_pose_backend_used": backend,
                    "crop_quality_mean": round(q, 3), "crop_quality_bucket": bucket})
    if not agg:
        return m
    return m.merge(pd.DataFrame(agg), on="event_id", how="left")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scanning", required=True)
    p.add_argument("--ground_truth", default="data/annotations/scanning_windows_gt.csv")
    p.add_argument("--head_pose", default=None,
                   help="head_pose.parquet para desglose por backend/crop_quality")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    a = get_args()
    out = Path(a.output_dir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    pred = (pd.read_parquet(a.scanning) if a.scanning.endswith("parquet")
            else pd.read_csv(a.scanning))

    gt_path = Path(a.ground_truth)
    have_gt = False
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        if "scan_label_gt" in gt.columns:
            gt = gt[pd.to_numeric(gt["scan_label_gt"], errors="coerce").notna()]
            have_gt = len(gt) > 0

    if not have_gt:
        msg = ("# Evaluacion scanning V2\n\n"
               "**Ground truth not available. Only heuristic outputs generated.**\n\n"
               "Para evaluar precision/recall/F1, completa `scan_label_gt` en "
               f"`{gt_path}` (ver annotation_pack/README_annotation_guidelines.md).\n")
        (out / "evaluation_gt.md").write_text(msg, encoding="utf-8")
        (out / "evaluation_gt.json").write_text(
            json.dumps({"ground_truth_available": False}, indent=2), encoding="utf-8")
        print("[eval] " + "Ground truth not available. Only heuristic outputs generated.")
        return

    gt["scan_label_gt"] = gt["scan_label_gt"].astype(int)
    gt_cols = ["event_id", "scan_label_gt"] + [c for c in ("visibility", "event_source")
                                               if c in gt.columns]
    m = pred.merge(gt[gt_cols], on="event_id", how="inner")
    m["scan_label_pred"] = pd.to_numeric(m["scan_label_pred"], errors="coerce").fillna(0).astype(int)
    m = _attach_head_pose(m, a.head_pose)

    overall = _metrics(m["scan_label_gt"], m["scan_label_pred"])
    fp_ids, fn_ids = _fp_fn(m)
    result = {
        "ground_truth_available": True, "overall": overall,
        "false_positive_event_ids": fp_ids, "false_negative_event_ids": fn_ids,
        "by_visibility": _by(m, "visibility"),
        "by_event_source": _by(m, "event_source" if "event_source" in m else "source"),
        "by_head_pose_backend_used": _by(m, "head_pose_backend_used"),
        "by_crop_quality": _by(m, "crop_quality_bucket"),
    }
    (out / "evaluation_gt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Evaluacion scanning V2 (contra GT humano)", "",
             f"- muestras evaluadas: {overall['n']}",
             f"- precision: {overall['precision']:.3f}  recall: {overall['recall']:.3f}  "
             f"F1: {overall['f1']:.3f}  accuracy: {overall['accuracy']:.3f}",
             f"- confusion: TP={overall['tp']} FP={overall['fp']} "
             f"FN={overall['fn']} TN={overall['tn']}",
             f"- falsos positivos (event_id): {fp_ids or '-'}",
             f"- falsos negativos (event_id): {fn_ids or '-'}", "",
             "## Desglose", "",
             f"- por visibility: {json.dumps(result['by_visibility'])}",
             f"- por event_source: {json.dumps(result['by_event_source'])}",
             f"- por head_pose_backend_used: {json.dumps(result['by_head_pose_backend_used'])}",
             f"- por crop_quality: {json.dumps(result['by_crop_quality'])}", ""]
    (out / "evaluation_gt.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
