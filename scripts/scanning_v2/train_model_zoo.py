# scripts/scanning_v2/train_model_zoo.py
"""
Entrena y compara TODAS las arquitecturas soportadas del clasificador de
scanning V2 sobre el mismo dataset y el mismo split (agrupado por video),
evalua en el split de test held-out y guarda cada modelo + un reporte
comparativo. El mejor modelo (por F1 en test; desempate PR-AUC) se copia a
`<output_dir>/best`.

Ejemplo:
  python scripts/scanning_v2/train_model_zoo.py \
    --config configs/scanning_v2_supervised_weak.yaml \
    --features outputs/scanning_training/dataset/features.parquet \
    --labels   outputs/scanning_training/dataset/labels.parquet \
    --output_dir outputs/scanning_training/models
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised import ScanningTrainer, SUPPORTED_MODELS  # noqa: E402
from core.scanning_v2.supervised.evaluator import _metrics                 # noqa: E402
from core.scanning_v2.supervised.schema import (                           # noqa: E402
    HEURISTIC_PRED_COLUMN, LABEL_COLUMN)


def cv_evaluate(result: dict, model_cfg: dict, n_splits: int = 4) -> dict:
    """Metricas out-of-fold con CV estratificada AGRUPADA por video.

    Con datasets pequenos un unico split de test puede quedar sin positivos;
    la CV agrupada usa TODAS las muestras etiquetadas manteniendo videos
    completos fuera del fold de entrenamiento."""
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    from core.scanning_v2.supervised.models import build_model

    labeled = result["labeled"]
    cols = result["feature_columns"]
    X = labeled[cols]
    y = labeled[LABEL_COLUMN].astype(int).to_numpy()
    groups = (labeled["video_id"].to_numpy() if "video_id" in labeled else None)
    n_splits = int(min(n_splits, max(2, int((y == 1).sum()))))
    if groups is not None and len(set(groups)) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        folds = splitter.split(X, y, groups)
        method = f"stratified_group_kfold(k={n_splits})"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=42)
        folds = splitter.split(X, y)
        method = f"stratified_kfold(k={n_splits})"

    proba = np.full(len(y), np.nan)
    for tr, te in folds:
        m, _ = build_model(model_cfg)
        m.fit(X.iloc[tr], y[tr])
        p = (m.predict_proba(X.iloc[te])[:, 1] if hasattr(m, "predict_proba")
             else m.predict(X.iloc[te]).astype(float))
        proba[te] = p
    ok = ~np.isnan(proba)
    met = _metrics(y[ok], (proba[ok] >= 0.5).astype(int), proba[ok])
    met["cv_method"] = method
    # umbral alternativo: mejor F1 sobre las probas out-of-fold (diagnostico)
    best_thr, best_f1 = 0.5, met["f1"]
    for thr in np.unique(np.round(proba[ok], 3)):
        mm = _metrics(y[ok], (proba[ok] >= thr).astype(int))
        if mm["f1"] > best_f1:
            best_thr, best_f1 = float(thr), mm["f1"]
    met["best_f1_threshold"] = best_thr
    met["best_f1_at_threshold"] = best_f1
    return met


def evaluate_on_split(result: dict, split: str) -> dict:
    labeled = result["labeled"]
    idx = result["splits"][split]
    if not len(idx):
        return {"n": 0}
    X = labeled.loc[idx, result["feature_columns"]]
    y = labeled.loc[idx, LABEL_COLUMN].astype(int)
    model = result["model"]
    proba = (model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba")
             else None)
    pred = model.predict(X)
    return _metrics(y, pred, proba)


def heuristic_on_all(result: dict) -> dict:
    labeled = result["labeled"]
    if HEURISTIC_PRED_COLUMN not in labeled.columns:
        return {"n": 0}
    y = labeled[LABEL_COLUMN].astype(int)
    pred = pd.to_numeric(labeled[HEURISTIC_PRED_COLUMN],
                         errors="coerce").fillna(0).astype(int)
    return _metrics(y, pred)


def heuristic_on_split(result: dict, split: str) -> dict:
    labeled = result["labeled"]
    idx = result["splits"][split]
    if not len(idx) or HEURISTIC_PRED_COLUMN not in labeled.columns:
        return {"n": 0}
    y = labeled.loc[idx, LABEL_COLUMN].astype(int)
    pred = pd.to_numeric(labeled.loc[idx, HEURISTIC_PRED_COLUMN],
                         errors="coerce").fillna(0).astype(int)
    return _metrics(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scanning_v2_supervised_weak.yaml")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help="subconjunto de arquitecturas (default: todas)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cv", type=int, default=None,
                    help="folds de CV agrupada; default: automatico si n<40")
    ap.add_argument("--refit_all", action="store_true",
                    help="re-entrena el modelo final con TODOS los labeled "
                         "(recomendado junto con --cv)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_cfg = dict(config.get("model", {}))
    base_cfg["feature_version"] = config.get("dataset", {}).get(
        "feature_version", "scanning_v2_features_v1")

    features = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    model_types = args.models or list(SUPPORTED_MODELS)

    rows, saved = [], {}
    heur_test = None
    for mtype in model_types:
        cfg = dict(base_cfg)
        cfg["type"] = mtype
        trainer = ScanningTrainer(cfg)
        result = trainer.train(features, labels)
        if result["status"] != "trained":
            print(f"[zoo] {mtype}: NO entrenado -> {result['reason']}")
            rows.append({"model": mtype, "status": result["reason"]})
            continue

        n_labeled = len(result["labeled"])
        use_cv = args.cv if args.cv is not None else (4 if n_labeled < 40 else 0)
        if use_cv:
            te = cv_evaluate(result, cfg, n_splits=use_cv)
            eval_scope = te["cv_method"]
        else:
            te = evaluate_on_split(result, "test")
            eval_scope = f"holdout ({result['metadata']['split_method']})"
        if heur_test is None:
            heur_test = (heuristic_on_all(result) if use_cv
                         else heuristic_on_split(result, "test"))
        if args.refit_all or use_cv:
            # artefacto final entrenado con TODOS los labeled (metricas = CV)
            X_all = result["labeled"][result["feature_columns"]]
            y_all = result["labeled"][LABEL_COLUMN].astype(int)
            result["model"].fit(X_all, y_all)
            result["metadata"]["final_fit"] = "all_labeled"
        result["metadata"]["evaluation"] = {"scope": eval_scope, **te}
        out_dir = os.path.join(args.output_dir, mtype)
        saved[mtype] = ScanningTrainer.write(result, out_dir,
                                             overwrite=args.overwrite)
        rows.append({"model": mtype, "status": "trained",
                     "test_f1": te.get("f1"), "test_precision": te.get("precision"),
                     "test_recall": te.get("recall"), "test_accuracy": te.get("accuracy"),
                     "test_pr_auc": te.get("pr_auc"), "test_roc_auc": te.get("roc_auc"),
                     "best_f1_threshold": te.get("best_f1_threshold"),
                     "best_f1_at_threshold": te.get("best_f1_at_threshold"),
                     "n_test": te.get("n"), "split_method": eval_scope})
        print(f"[zoo] {mtype}: {eval_scope} F1={te.get('f1'):.3f} "
              f"P={te.get('precision'):.3f} R={te.get('recall'):.3f} "
              f"PR-AUC={te.get('pr_auc')} (n={te.get('n')})")

    trained = [r for r in rows if r.get("status") == "trained"]
    if not trained:
        print("[zoo] ninguna arquitectura pudo entrenarse.")
        sys.exit(1)

    best = max(trained, key=lambda r: (r["test_f1"] or 0, r["test_pr_auc"] or 0))
    best_dir = os.path.join(args.output_dir, "best")
    if os.path.isdir(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(os.path.join(args.output_dir, best["model"]), best_dir)

    report = {
        "models": rows,
        "best_model": best["model"],
        "heuristic_baseline_test": heur_test,
        "label_source": base_cfg.get("label_source", "human_gt"),
        "note": ("Con etiquetas debiles, las metricas miden CONSISTENCIA con las "
                 "reglas, no validez contra percepcion humana. NO es gaze real."),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "model_zoo_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    md = ["# Comparativa de arquitecturas — scanning V2 supervisado", "",
          f"- etiquetas: **{report['label_source']}**",
          f"- mejor modelo: **{best['model']}** (copiado a `best/`)", "",
          f"- evaluacion: **{trained[0]['split_method']}**", "",
          "| modelo | F1@0.5 | precision | recall | accuracy | PR-AUC | ROC-AUC "
          "| F1@thr* | thr* | n |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in trained:
        fmt = lambda v: "-" if v is None else round(v, 3)
        md.append(f"| {r['model']} | {r['test_f1']:.3f} | {r['test_precision']:.3f} "
                  f"| {r['test_recall']:.3f} | {r['test_accuracy']:.3f} "
                  f"| {fmt(r['test_pr_auc'])} | {fmt(r['test_roc_auc'])} "
                  f"| {fmt(r.get('best_f1_at_threshold'))} "
                  f"| {fmt(r.get('best_f1_threshold'))} | {r['n_test']} |")
    md += ["", "\\* mejor umbral F1 sobre probas out-of-fold (diagnostico, "
           "no seleccionado en datos de test independientes)."]
    skipped = [r for r in rows if r.get("status") != "trained"]
    if skipped:
        md += ["", "No entrenados: " + ", ".join(
            f"{r['model']} ({r['status']})" for r in skipped)]
    if heur_test and heur_test.get("n"):
        md += ["", "## Baseline heuristica V2 (mismo test split)",
               f"- F1 {heur_test['f1']:.3f}  precision {heur_test['precision']:.3f}  "
               f"recall {heur_test['recall']:.3f}  accuracy {heur_test['accuracy']:.3f}"]
    md += ["", "> " + report["note"], ""]
    with open(os.path.join(args.output_dir, "model_zoo_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[zoo] mejor: {best['model']} -> {best_dir}")
    print(f"[zoo] reporte -> {os.path.join(args.output_dir, 'model_zoo_report.md')}")


if __name__ == "__main__":
    main()
