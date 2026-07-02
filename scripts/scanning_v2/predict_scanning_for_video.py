# scripts/scanning_v2/predict_scanning_for_video.py
"""
Predice scanning por evento para UNA secuencia con el clasificador entrenado
y guarda el resultado junto a los outputs de la secuencia:

    outputs/<video_id>/scanning/model_predictions.parquet

Construye las features en memoria desde los outputs V2 de la secuencia (mismo
FeatureExtractor del entrenamiento) y aplica el modelo `best`.

  python scripts/scanning_v2/predict_scanning_for_video.py \
      --video_id SNMOT-148 \
      --model outputs/scanning_training/models/best/scanning_classifier.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.paths import resolve_scanning_dir            # noqa: E402
from core.scanning_v2.supervised import DatasetBuilder, ScanningPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scanning_v2_supervised_weak.yaml",
                    help="config del dataset (mismas features que el modelo)")
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--model",
                    default="outputs/scanning_training/models/best/scanning_classifier.pkl")
    ap.add_argument("--output", default=None,
                    help="default: <scanning_dir>/model_predictions.parquet")
    a = ap.parse_args()

    with open(a.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out = a.output or str(resolve_scanning_dir(a.outputs_root, a.video_id)
                          / "model_predictions.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    builder = DatasetBuilder(config)
    result = builder.build([a.video_id], a.outputs_root, annotations_path=None)
    features = result["features"]
    if not len(features):
        # parquet vacio con schema: marca la secuencia como procesada (cache)
        import pandas as pd
        from core.scanning_v2.supervised.schema import PREDICTION_COLUMNS
        pd.DataFrame(columns=PREDICTION_COLUMNS).to_parquet(out, index=False)
        print(f"[predict] {a.video_id}: sin eventos de recepcion -> {out} (vacio)")
        return

    predictor = ScanningPredictor.load(a.model)   # error claro si no existe
    preds = predictor.predict(features)
    preds.to_parquet(out, index=False)
    n_pos = int((preds["scan_label_model"] == 1).sum())
    print(f"[predict] {a.video_id}: {len(preds)} eventos ({n_pos} scan=1) -> {out}")


if __name__ == "__main__":
    main()
