# scripts/scanning_v2/evaluate_scanning_model.py
"""
Evalua el modelo supervisado contra GT humano y lo compara con la heuristica V2.

Sin GT -> no inventa metricas. Intenta enriquecer subgrupos con features.parquet
(hermano del dataset) si esta disponible.

Ejemplo:
  python scripts/scanning_v2/evaluate_scanning_model.py \
    --config configs/scanning_v2_supervised.yaml \
    --predictions outputs/scanning_v2_supervised/SNMOT-148/predictions/scanning_model_predictions.parquet \
    --labels      outputs/scanning_v2_supervised/SNMOT-148/dataset/labels.parquet \
    --heuristic_scanning outputs/scanning_v2/SNMOT-148/scanning_events.parquet \
    --output_dir  outputs/scanning_v2_supervised/SNMOT-148/reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised import ScanningEvaluator   # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/scanning_v2_supervised.yaml")
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--heuristic_scanning", default=None)
    p.add_argument("--features", default=None,
                   help="opcional; si no se da, se busca features.parquet hermano")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _maybe(path):
    return pd.read_parquet(path) if path and Path(path).exists() else None


def main():
    a = get_args()
    predictions = pd.read_parquet(a.predictions)
    labels = pd.read_parquet(a.labels)
    heuristic = _maybe(a.heuristic_scanning)

    feats_path = a.features
    if not feats_path:
        guess = Path(a.labels).parent / "features.parquet"
        feats_path = str(guess) if guess.exists() else None
    features = _maybe(feats_path)

    ev = ScanningEvaluator()
    result = ev.evaluate(predictions, labels, heuristic, features)
    out = ScanningEvaluator.write(result, a.output_dir)
    if result.get("ground_truth_available"):
        print(json.dumps(result["model"], indent=2, default=str))
    else:
        print("[eval] " + result.get("message", "sin GT"))
    print(f"[eval] reporte -> {out['report']}")


if __name__ == "__main__":
    main()
