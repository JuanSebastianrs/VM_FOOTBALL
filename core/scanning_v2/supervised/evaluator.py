# core/scanning_v2/supervised/evaluator.py
"""
Evaluacion del modelo supervisado contra GT humano.

Si NO hay GT (labels vacios), NO inventa metricas: reporta que falta. Con GT
calcula accuracy/precision/recall/F1, ROC-AUC y PR-AUC (solo si hay ambas clases),
matriz de confusion, FP/FN por event_id y desgloses por subgrupo. Compara contra
la baseline heuristica V2 (`scan_label_pred`) cuando esta disponible.

NO es gaze real: mide head-turn / visual scanning APROXIMADO antes de recepcion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema import LABEL_COLUMN


def _metrics(y_true, y_pred, proba=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    out = {"n": int(len(y_true)), "accuracy": acc, "precision": prec,
           "recall": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
           "roc_auc": None, "pr_auc": None}
    # AUCs solo si hay ambas clases en y_true y hay probabilidades
    if proba is not None and len(set(y_true.tolist())) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            out["roc_auc"] = float(roc_auc_score(y_true, proba))
            out["pr_auc"] = float(average_precision_score(y_true, proba))
        except Exception:
            pass
    return out


class ScanningEvaluator:
    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or {}

    # ------------------------------------------------------------------
    def evaluate(self, predictions: pd.DataFrame, labels: pd.DataFrame,
                 heuristic: Optional[pd.DataFrame] = None,
                 features: Optional[pd.DataFrame] = None) -> dict:
        lab = labels.copy()
        lab[LABEL_COLUMN] = pd.to_numeric(lab[LABEL_COLUMN], errors="coerce")
        lab = lab[lab[LABEL_COLUMN].isin([0, 1])]
        if not len(lab):
            return {"ground_truth_available": False,
                    "message": "Ground truth not available. No supervised metrics computed."}

        on = self._merge_keys(predictions, lab)
        m = predictions.merge(lab, on=on, how="inner", suffixes=("", "_lab"))
        if not len(m):
            # reportar claves faltantes para diagnosticar
            pk = self._keys(predictions, on)
            lk = self._keys(lab, on)
            return {"ground_truth_available": False,
                    "message": "No overlap between predictions and labeled GT.",
                    "key_columns": on,
                    "prediction_keys_not_in_gt": sorted(pk - lk)[:50],
                    "gt_keys_not_in_predictions": sorted(lk - pk)[:50]}
        if features is not None and len(features):
            m = self._attach_features(m, features, on)

        y = m[LABEL_COLUMN].astype(int)
        yhat = m["scan_label_model"].astype(int)
        proba = m["scan_probability_model"] if "scan_probability_model" in m else None
        overall = _metrics(y, yhat, proba)

        fp = self._error_rows(m, (y == 0) & (yhat == 1))
        fn = self._error_rows(m, (y == 1) & (yhat == 0))

        subgroups = {}
        for col in ("visibility", "confidence", "event_source",
                    "head_pose_backend_dominant", "head_crop_quality_bin", "video_id"):
            if col in m.columns:
                subgroups[col] = {str(v): _metrics(g[LABEL_COLUMN].astype(int),
                                                   g["scan_label_model"].astype(int))
                                  for v, g in m.groupby(col)}

        result = {"ground_truth_available": True, "key_columns": on, "model": overall,
                  "false_positives": fp, "false_negatives": fn,
                  "false_positive_event_ids": [r["event_id"] for r in fp],
                  "false_negative_event_ids": [r["event_id"] for r in fn],
                  "subgroups": subgroups, "n_labeled": int(len(m)),
                  "note": "head-turn / visual scanning APROXIMADO; NO gaze real."}

        # baseline heuristica V2 (misma clave)
        if heuristic is not None and len(heuristic):
            h = heuristic.rename(columns={"scan_label_pred": "scan_label_heur"})
            hk = [c for c in on if c in h.columns] or ["event_id"]
            hcols = hk + ["scan_label_heur"]
            mh = m.merge(h[hcols].drop_duplicates(hk), on=hk, how="inner")
            if len(mh):
                result["heuristic_baseline"] = _metrics(
                    mh[LABEL_COLUMN].astype(int),
                    pd.to_numeric(mh["scan_label_heur"], errors="coerce").fillna(0).astype(int))
        return result

    @staticmethod
    def _merge_keys(a, b):
        return (["event_id", "video_id"]
                if "video_id" in a.columns and "video_id" in b.columns
                else ["event_id"])

    @staticmethod
    def _keys(df, on):
        return set(tuple(str(v) for v in row) for row in df[on].itertuples(index=False))

    @staticmethod
    def _error_rows(m, mask):
        cols = [c for c in ("video_id", "event_id", "receiver_track_id",
                            "scan_probability_model", LABEL_COLUMN) if c in m.columns]
        return [{k: (None if pd.isna(v) else v) for k, v in r.items()}
                for r in m.loc[mask, cols].to_dict("records")]

    def _attach_features(self, m, features, on):
        cols = list(on)
        f = features.copy()
        if "event_source" in f.columns:
            cols.append("event_source")
        # backend dominante
        bcols = [c for c in f.columns if c.startswith("backend_") and c.endswith("_ratio")]
        if bcols:
            f["head_pose_backend_dominant"] = (f[bcols].idxmax(axis=1)
                                               .str.replace("backend_", "", regex=False)
                                               .str.replace("_ratio", "", regex=False))
            cols.append("head_pose_backend_dominant")
        if "mean_head_crop_quality" in f.columns:
            q = pd.to_numeric(f["mean_head_crop_quality"], errors="coerce").fillna(0.0)
            f["head_crop_quality_bin"] = np.where(q >= 0.5, "high",
                                                  np.where(q >= 0.2, "medium", "low"))
            cols.append("head_crop_quality_bin")
        keep = [c for c in cols if c in f.columns]
        key = [c for c in on if c in f.columns] or ["event_id"]
        fsub = f[keep].drop_duplicates(key)
        return m.merge(fsub, on=key, how="left", suffixes=("", "_feat"))

    # ------------------------------------------------------------------
    @staticmethod
    def write(result: dict, out_dir: str) -> dict:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "evaluation_report.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        md = d / "evaluation_report.md"
        if not result.get("ground_truth_available"):
            md.write_text("# Evaluacion modelo scanning V2 (supervisado)\n\n"
                          f"**{result.get('message')}**\n\n"
                          "No se inventan metricas. Completa `scan_label_gt` en el GT.\n",
                          encoding="utf-8")
            return {"report": str(md), "json": str(d / "evaluation_report.json")}
        o = result["model"]
        lines = ["# Evaluacion modelo scanning V2 (supervisado)", "",
                 "> head-turn / visual scanning **APROXIMADO** antes de recepcion. "
                 "**No es gaze real.**", "",
                 f"- muestras con GT: {o['n']}",
                 f"- accuracy {o['accuracy']:.3f}  precision {o['precision']:.3f}  "
                 f"recall {o['recall']:.3f}  F1 {o['f1']:.3f}",
                 f"- ROC-AUC {o['roc_auc']}  PR-AUC {o['pr_auc']}",
                 f"- confusion: TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']}",
                 f"- falsos positivos: {result['false_positive_event_ids'] or '-'}",
                 f"- falsos negativos: {result['false_negative_event_ids'] or '-'}", ""]
        if "heuristic_baseline" in result:
            b = result["heuristic_baseline"]
            lines += ["## Baseline heuristica V2",
                      f"- accuracy {b['accuracy']:.3f}  precision {b['precision']:.3f}  "
                      f"recall {b['recall']:.3f}  F1 {b['f1']:.3f}", ""]
        if result.get("subgroups"):
            lines.append("## Subgrupos")
            for k, v in result["subgroups"].items():
                lines.append(f"- **{k}**: {json.dumps(v, default=str)}")
        md.write_text("\n".join(lines), encoding="utf-8")
        return {"report": str(md), "json": str(d / "evaluation_report.json")}
