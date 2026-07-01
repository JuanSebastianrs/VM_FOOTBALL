# scripts/scanning/train_orientation_model.py
"""
Entrena el modelo temporal de orientacion (TCN/BiLSTM).

Ejemplo:
  python scripts/scanning/train_orientation_model.py \
      --features data/annotations/SNMOT-148_orientation_features.parquet \
      --labels data/annotations/orientation_labels.csv \
      --epochs 50 --model tcn
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning import load_config                              # noqa: E402
from core.scanning.models.train_orientation_model import Trainer   # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--labels", default=None, help="override data.annotation_csv")
    p.add_argument("--features", default=None, help="features parquet store")
    p.add_argument("--model", default=None, choices=["tcn", "bilstm"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--sequence_length", type=int, default=None)
    p.add_argument("--balance_bins", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    tr = config.setdefault("training", {})
    if args.labels:
        config["data"]["annotation_csv"] = args.labels
    if args.features:
        config["features_parquet"] = args.features
    if args.model:
        tr["model"] = args.model
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if args.lr is not None:
        tr["learning_rate"] = args.lr
    if args.sequence_length is not None:
        tr["sequence_length"] = args.sequence_length
    if args.balance_bins:
        tr["balance_bins"] = True
    if args.device:
        tr["device"] = args.device

    trainer = Trainer(config)
    print(f"train={len(trainer.train_ds)}  val={len(trainer.val_ds)}  "
          f"model={tr.get('model', 'tcn')}  device={trainer.device}")
    trainer.fit()


if __name__ == "__main__":
    main()
