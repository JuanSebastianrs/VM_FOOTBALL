# scripts/scanning_v2/train_scanning_model.py
"""
Entrena el clasificador supervisado de scanning V2 (si hay GT suficiente).

Si no hay etiquetas suficientes, NO entrena y escribe un reporte explicandolo;
la heuristica V2 sigue como fallback.

Ejemplo:
  python scripts/scanning_v2/train_scanning_model.py \
    --config configs/scanning_v2_supervised.yaml \
    --features outputs/scanning_training_gt/dataset/features.parquet \
    --labels   outputs/scanning_training_gt/dataset/labels.parquet \
    --output_dir outputs/scanning_training_gt/models \
    --model-type logistic_regression --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised import ScanningTrainer   # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/scanning_v2_supervised.yaml")
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model-type", dest="model_type", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = get_args()
    with open(a.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_cfg = dict(config.get("model", {}))
    model_cfg["feature_version"] = config.get("dataset", {}).get(
        "feature_version", "scanning_v2_features_v1")
    if a.model_type:
        model_cfg["type"] = a.model_type

    features = pd.read_parquet(a.features)
    labels = pd.read_parquet(a.labels)

    trainer = ScanningTrainer(model_cfg)
    result = trainer.train(features, labels)
    out = ScanningTrainer.write(result, a.output_dir, overwrite=a.overwrite)

    if result["status"] == "trained":
        m = result["metadata"]
        print(f"[train] modelo={m['model_type']} samples={m['n_samples']} "
              f"(pos {m['n_positive']}/neg {m['n_negative']}) -> {out['model']}")
    else:
        print(f"[train] NO se entreno: {result['reason']}")
        print(f"[train] reporte -> {out['report']}")


if __name__ == "__main__":
    main()
