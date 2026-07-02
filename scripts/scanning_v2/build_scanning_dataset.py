# scripts/scanning_v2/build_scanning_dataset.py
"""
Construye el dataset supervisado (features + labels) desde los outputs V2.

Ejemplo:
  python scripts/scanning_v2/build_scanning_dataset.py \
    --config configs/scanning_v2_supervised.yaml --video_ids SNMOT-148 \
    --outputs_root outputs \
    --annotations data/annotations/scanning_windows_gt.csv \
    --output_dir outputs/scanning_training_gt/dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised import DatasetBuilder   # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/scanning_v2_supervised.yaml")
    p.add_argument("--video_ids", nargs="+", default=None)
    p.add_argument("--outputs_root", default=None)
    p.add_argument("--annotations", default=None)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    a = get_args()
    with open(a.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    inp = config.get("inputs", {})
    video_ids = a.video_ids or inp.get("video_ids", [])
    outputs_root = a.outputs_root or inp.get("outputs_root", "outputs")
    annotations = a.annotations or inp.get("annotations_path")

    builder = DatasetBuilder(config)
    result = builder.build(video_ids, outputs_root, annotations)
    paths = DatasetBuilder.write(result, a.output_dir)

    print("=== DATASET SUPERVISADO ===")
    print(json.dumps(result["manifest"], indent=2))
    print("paths:", json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
