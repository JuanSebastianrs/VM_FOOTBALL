# core/scanning_v2/supervised/trainer.py
"""
Entrenamiento del clasificador supervisado de scanning V2.

Solo entrena si hay EVIDENCIA suficiente (min_labeled_samples / min_positive_samples
/ ambas clases presentes). Si no, NO entrena y escribe un reporte explicando por
que; la heuristica V2 sigue siendo el fallback.

Splits reproducibles (`random_state`); opcionalmente agrupados por video para no
filtrar el mismo partido entre train y test.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .models import build_model
from .schema import FEATURE_VERSION, LABEL_COLUMN, training_feature_columns


class ScanningTrainer:
    def __init__(self, config: Optional[dict] = None):
        c = (config or {}).get("model", {}) if "model" in (config or {}) else (config or {})
        self.model_cfg = dict(c)
        self.min_labeled = int(c.get("min_labeled_samples", 20))
        self.min_positive = int(c.get("min_positive_samples", 3))
        self.test_size = float(c.get("test_size", 0.2))
        self.val_size = float(c.get("val_size", 0.2))
        self.group_split_by_video = bool(c.get("group_split_by_video", True))
        self.random_state = int(c.get("random_state", 42))
        self.include_heuristic_pred = bool(
            c.get("use_heuristic_pred_as_feature", False))
        self.feature_version = str(c.get("feature_version", FEATURE_VERSION))
        self.merge_warning = None

    # ------------------------------------------------------------------
    def train(self, features: pd.DataFrame, labels: pd.DataFrame) -> dict:
        merged = self._merge(features, labels)
        y_all = pd.to_numeric(merged[LABEL_COLUMN], errors="coerce")
        labeled = merged[y_all.isin([0, 1])].copy()
        labeled[LABEL_COLUMN] = pd.to_numeric(labeled[LABEL_COLUMN]).astype(int)
        # respetar elegibilidad calculada por el DatasetBuilder (gates de config)
        if "training_eligible" in labeled.columns:
            labeled = labeled[labeled["training_eligible"].fillna(False).astype(bool)]

        n = int(len(labeled))
        n_pos = int((labeled[LABEL_COLUMN] == 1).sum())
        n_neg = int((labeled[LABEL_COLUMN] == 0).sum())

        gate = self._check_gates(n, n_pos, n_neg)
        if gate is not None:
            return {"status": "skipped", "reason": gate,
                    "n_labeled": n, "n_positive": n_pos, "n_negative": n_neg,
                    "min_labeled_samples": self.min_labeled,
                    "min_positive_samples": self.min_positive}

        feat_cols = training_feature_columns(labeled, self.include_heuristic_pred)
        X, y = labeled[feat_cols], labeled[LABEL_COLUMN]
        groups = labeled["video_id"] if "video_id" in labeled else None

        splits = self._splits(labeled, y, groups)
        model, params = build_model({**self.model_cfg,
                                     "feature_version": self.feature_version})
        tr_idx = splits["train"]
        model.fit(X.loc[tr_idx], y.loc[tr_idx])

        meta = {
            "model_type": params["type"], "model_params": params,
            "feature_version": self.feature_version,
            "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "n_samples": n, "n_positive": n_pos, "n_negative": n_neg,
            "feature_columns": feat_cols, "label_column": LABEL_COLUMN,
            "split_method": splits["method"],
            "split_sizes": {k: int(len(v)) for k, v in splits.items()
                            if k in ("train", "val", "test")},
            "use_heuristic_pred_as_feature": self.include_heuristic_pred,
            "repo_version": _repo_version(),
            "key_columns": ["video_id", "event_id"],
            "merge_warning": self.merge_warning,
            "config": self.model_cfg,
            "limitations": ("Orientacion APROXIMADA de cabeza/cuerpo, NO gaze real. "
                            "Validez limitada por el tamano del GT humano."),
        }
        return {"status": "trained", "model": model, "metadata": meta,
                "labeled": labeled, "feature_columns": feat_cols, "splits": splits}

    # ------------------------------------------------------------------
    def _check_gates(self, n, n_pos, n_neg) -> Optional[str]:
        if n < self.min_labeled:
            return (f"Not enough labeled samples to train supervised scanning model "
                    f"({n} < {self.min_labeled}).")
        if n_pos < self.min_positive:
            return (f"Not enough positive samples ({n_pos} < {self.min_positive}).")
        if n_neg < 1:
            return "Only one class present in labels; cannot train a classifier."
        return None

    def _merge(self, features, labels):
        keep = [c for c in labels.columns if c == LABEL_COLUMN
                or c in ("event_id", "video_id", "training_eligible")]
        lab = labels[keep].copy()
        # clave logica (video_id, event_id); fallback a event_id con warning
        if "video_id" in features.columns and "video_id" in lab.columns:
            on = ["event_id", "video_id"]
        else:
            on = ["event_id"]
            self.merge_warning = ("video_id ausente; merge features/labels solo por "
                                  "event_id (riesgo en multi-video).")
        return features.merge(lab, on=on, how="left", suffixes=("", "_lab"))

    def _splits(self, labeled, y, groups):
        idx = labeled.index.to_numpy()
        n_groups = groups.nunique() if groups is not None else 1
        if self.group_split_by_video and n_groups > 1:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size,
                                    random_state=self.random_state)
            tr, te = next(gss.split(idx, y, groups))
            method = "group_by_video"
        else:
            from sklearn.model_selection import train_test_split
            strat = y if y.nunique() > 1 else None
            try:
                tr, te = train_test_split(np.arange(len(idx)), test_size=self.test_size,
                                          random_state=self.random_state, stratify=strat)
            except ValueError:
                tr, te = train_test_split(np.arange(len(idx)), test_size=self.test_size,
                                          random_state=self.random_state)
            base = "stratified" if strat is not None else "random"
            # si se pidio group split pero solo hay 1 video, documentar el fallback
            method = (f"{base} (group_split fallback: solo {n_groups} video)"
                      if self.group_split_by_video else base)
        # val desde train
        val = np.array([], dtype=int)
        if self.val_size > 0 and len(tr) > 2:
            from sklearn.model_selection import train_test_split
            y_tr = y.iloc[tr]
            strat = y_tr if y_tr.nunique() > 1 else None
            try:
                tr2, val = train_test_split(tr, test_size=self.val_size,
                                            random_state=self.random_state, stratify=strat)
                tr = tr2
            except ValueError:
                pass
        to_label = lambda pos: pd.Index(idx[pos])
        return {"method": method, "train": to_label(tr), "val": to_label(val),
                "test": to_label(te)}

    # ------------------------------------------------------------------
    @staticmethod
    def write(result: dict, out_dir: str, overwrite: bool = False) -> dict:
        import joblib
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        report = d / "training_report.md"
        if result["status"] != "trained":
            _write_skip_report(report, result)
            (d / "model_metadata.json").write_text(
                json.dumps({"status": "skipped", "reason": result["reason"]},
                           indent=2), encoding="utf-8")
            return {"status": "skipped", "report": str(report)}

        model_path = d / "scanning_classifier.pkl"
        if model_path.exists() and not overwrite:
            raise FileExistsError(
                f"{model_path} ya existe; usa --overwrite para reemplazar.")
        joblib.dump(result["model"], model_path)
        (d / "model_metadata.json").write_text(
            json.dumps(result["metadata"], indent=2, default=str), encoding="utf-8")
        # splits
        for name in ("train", "val", "test"):
            sub = result["labeled"].loc[result["splits"][name]]
            sub[["event_id", "video_id", LABEL_COLUMN]].to_csv(
                d / f"{name}_split.csv", index=False)
        _write_train_report(report, result)
        return {"status": "trained", "model": str(model_path),
                "metadata": str(d / "model_metadata.json"), "report": str(report)}


def _repo_version():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _write_skip_report(path: Path, r: dict):
    path.write_text(
        "# Entrenamiento scanning V2 (supervisado)\n\n"
        f"**No se entreno.** {r['reason']}\n\n"
        f"- labeled: {r.get('n_labeled')}  positive: {r.get('n_positive')}  "
        f"negative: {r.get('n_negative')}\n"
        f"- minimos requeridos: labeled>={r.get('min_labeled_samples')}, "
        f"positive>={r.get('min_positive_samples')}\n\n"
        "Se mantiene la heuristica V2 como fallback. Completa mas etiquetas en "
        "`data/annotations/scanning_windows_gt.csv` y reintenta.\n",
        encoding="utf-8")


def _write_train_report(path: Path, r: dict):
    m = r["metadata"]
    lines = ["# Entrenamiento scanning V2 (supervisado)", "",
             f"- modelo: **{m['model_type']}**",
             f"- feature_version: {m['feature_version']}",
             f"- samples: {m['n_samples']} (pos {m['n_positive']} / neg {m['n_negative']})",
             f"- split: {m['split_method']} -> {m['split_sizes']}",
             f"- n features: {len(m['feature_columns'])}",
             f"- entrenado: {m['trained_at']}",
             "- splits guardados junto al modelo: `train_split.csv`, `val_split.csv`, "
             "`test_split.csv` (en el `output_dir` del modelo).", "",
             "Orientacion APROXIMADA de cabeza/cuerpo, **NO gaze real**.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
