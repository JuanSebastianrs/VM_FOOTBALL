# scripts/scanning_v2/predict_scanning_model.py
"""
Predice scan_label/probabilidad por evento con el modelo supervisado entrenado.

Si no existe el modelo, falla con error claro (no silenciosamente).

Ejemplo:
  python scripts/scanning_v2/predict_scanning_model.py \
    --config configs/scanning_v2_supervised.yaml \
    --features outputs/scanning_training_gt/dataset/features.parquet \
    --model   outputs/scanning_training_gt/models/scanning_classifier.pkl \
    --output_dir outputs/scanning_training_gt/predictions
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised import ScanningPredictor   # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/scanning_v2_supervised.yaml")
    p.add_argument("--features", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    a = get_args()
    features = pd.read_parquet(a.features)
    predictor = ScanningPredictor.load(a.model)   # FileNotFoundError si no existe
    preds = predictor.predict(features)
    path = ScanningPredictor.write(preds, a.output_dir)
    n_pos = int((preds["scan_label_model"] == 1).sum())
    print(f"[predict] {len(preds)} eventos ({n_pos} scan=1) -> {path}")


if __name__ == "__main__":
    main()
