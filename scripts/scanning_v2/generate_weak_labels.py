# scripts/scanning_v2/generate_weak_labels.py
"""
Genera pseudo-etiquetas DEBILES (weak supervision) desde features por evento.

Escribe un CSV separado del GT humano (`label_source=weak_rules_v1`). Ver
core/scanning_v2/supervised/weak_labeler.py para reglas y advertencias.

Ejemplo:
  python scripts/scanning_v2/generate_weak_labels.py \
    --features outputs/scanning_v2_supervised_weak/dataset/features.parquet \
    --human_gt data/annotations/scanning_windows_gt.csv \
    --output data/annotations/scanning_windows_weak_v1.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised import generate_weak_labels, write_weak_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.parquet por evento")
    ap.add_argument("--human_gt", default="data/annotations/scanning_windows_gt.csv")
    ap.add_argument("--output", default="data/annotations/scanning_windows_weak_v1.csv")
    args = ap.parse_args()

    features = pd.read_parquet(args.features)
    human = pd.read_csv(args.human_gt) if os.path.exists(args.human_gt) else None
    weak = generate_weak_labels(features, human)
    stats = write_weak_labels(weak, args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
