"""
Score every crop of a jersey tracklet dataset with the trained LegibilityClassifier.

Produces a reusable cache (legibility_scores.json) mapping crop relative path ->
legibility score in [0, 1]. The cache is consumed by:
  - training/identification/train_jersey_perframe.py (filter training crops)
  - scripts/evaluate_jersey_dataset_temporal.py (legibility-weighted fusion)

Scoring is pure inference: the legibility model was trained with the test split
and SNMOT-148 excluded, so scoring test crops here does NOT leak labels.

Usage:
    python scripts/score_crop_legibility.py \
        --dataset_dir datasets/jersey_tracking_v1 \
        --legibility_model runs/jersey_legibility_v1/best.pt \
        --device cuda:0
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.identity.jersey_model import TRANSFORM_INFERENCE, load_legibility_model


@torch.no_grad()
def score_paths(model, crop_paths, root, device, batch_size=128):
    """Score a list of relative crop paths. Returns dict path -> float score."""
    scores = {}
    batch_tensors = []
    batch_keys = []

    def _flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        for key, p in zip(batch_keys, np.atleast_1d(probs)):
            scores[key] = float(p)
        batch_tensors.clear()
        batch_keys.clear()

    for rel in crop_paths:
        img_path = root / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Missing crop: {img_path}")
        img = np.array(Image.open(img_path).convert("RGB"))
        batch_tensors.append(TRANSFORM_INFERENCE(img))
        batch_keys.append(rel)
        if len(batch_tensors) >= batch_size:
            _flush()
    _flush()
    return scores


def main():
    parser = argparse.ArgumentParser(description="Score dataset crops with legibility model")
    parser.add_argument("--dataset_dir", type=str, default="datasets/jersey_tracking_v1")
    parser.add_argument("--legibility_model", type=str, default="runs/jersey_legibility_v1/best.pt")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Default: <dataset_dir>/legibility_scores.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_json = Path(args.output_json) if args.output_json else dataset_dir / "legibility_scores.json"

    with open(dataset_dir / "tracklets.json") as f:
        records = json.load(f)

    # Normalize to posix-style relative paths so the cache is OS-independent
    all_paths = []
    seen = set()
    for rec in records:
        for cp in rec.get("crop_paths", []):
            key = Path(cp).as_posix()
            if key not in seen:
                seen.add(key)
                all_paths.append(key)
    print(f"Scoring {len(all_paths)} unique crops from {len(records)} tracklets...")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_legibility_model(args.legibility_model, device)

    scores = score_paths(model, all_paths, dataset_dir, device, batch_size=args.batch_size)

    with open(output_json, "w") as f:
        json.dump(scores, f)

    vals = np.array(list(scores.values()))
    print(f"Saved {len(scores)} scores -> {output_json}")
    print(f"  mean={vals.mean():.3f} median={np.median(vals):.3f} "
          f">=0.5: {(vals >= 0.5).mean():.1%}  >=0.7: {(vals >= 0.7).mean():.1%}")


if __name__ == "__main__":
    main()
