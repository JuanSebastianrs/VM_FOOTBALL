"""
Build a per-frame training index from SoccerNet Jersey 2023 (train split).

For each VISIBLE tracklet (jersey != -1):
  1. Deterministically subsample up to --frames_per_tracklet frames.
  2. Extract the standard torso crop (same fractions as core/identity/crops.py:
     rows 10-70%, cols 10-90% — SoccerNet images are tight player crops, so the
     image itself plays the role of the bbox).
  3. Score each torso crop with the LegibilityClassifier.
  4. Keep the top --max_keep most legible frames with score >= --min_legibility.

Output: JSON index consumed by train_jersey_perframe.py --soccernet_index:
  [{"image_path": "<relative to soccernet root>", "jersey": 25, "legibility": 0.93}, ...]

Only the train split is ever indexed — test/challenge are never touched.

Usage:
    python scripts/build_soccernet_perframe_index.py \
        --soccernet_root datasets/soccernet/jersey-2023 \
        --legibility_model runs/jersey_legibility_v1/best.pt \
        --output_json datasets/soccernet/jersey-2023/perframe_index_train.json
"""

import sys
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.identity.jersey_model import TRANSFORM_INFERENCE, load_legibility_model


def torso_crop_array(img):
    """Standard torso fractions applied to a tight player crop."""
    h, w = img.shape[:2]
    top, bottom = int(h * 0.10), int(h * 0.70)
    left, right = int(w * 0.10), int(w * 0.90)
    if top >= bottom or left >= right:
        return None
    crop = img[top:bottom, left:right]
    return crop if crop.size else None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Build SoccerNet jersey per-frame index")
    parser.add_argument("--soccernet_root", type=str, default="datasets/soccernet/jersey-2023")
    parser.add_argument("--legibility_model", type=str, default="runs/jersey_legibility_v1/best.pt")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="test SOLO como datos de entrenamiento extra (renuncia a ese benchmark)")
    parser.add_argument("--frames_per_tracklet", type=int, default=32)
    parser.add_argument("--max_keep", type=int, default=16)
    parser.add_argument("--min_legibility", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.soccernet_root)
    sp = args.split
    gt_path = root / sp / f"{sp}_gt.json"
    if not gt_path.exists():
        gt_path = root / sp / sp / f"{sp}_gt.json"
    with open(gt_path) as f:
        gt = json.load(f)
    visible = {k: v for k, v in gt.items() if v != -1}

    images_dir = root / sp / "images"
    if not images_dir.exists():
        images_dir = root / sp / sp / "images"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_legibility_model(args.legibility_model, device)

    index = []
    skipped = 0
    for n_done, (pid, jersey) in enumerate(sorted(visible.items())):
        jersey = int(jersey)
        if jersey < 1 or jersey > 99:
            continue
        player_dir = images_dir / pid
        if not player_dir.exists():
            skipped += 1
            continue
        files = sorted(player_dir.glob("*.jpg"))
        if not files:
            skipped += 1
            continue
        import zlib
        rng = random.Random(args.seed + zlib.crc32(pid.encode()))  # stable across runs
        if len(files) > args.frames_per_tracklet:
            files = rng.sample(files, args.frames_per_tracklet)

        crops, keys = [], []
        for fp in files:
            img = np.array(Image.open(fp).convert("RGB"))
            crop = torso_crop_array(img)
            if crop is None:
                continue
            crops.append(TRANSFORM_INFERENCE(crop))
            keys.append(fp.relative_to(root).as_posix())
        if not crops:
            skipped += 1
            continue

        scores = []
        for j in range(0, len(crops), args.batch_size):
            x = torch.stack(crops[j:j + args.batch_size]).to(device)
            scores.extend(torch.sigmoid(model(x)).cpu().numpy().tolist())

        ranked = sorted(zip(keys, scores), key=lambda kv: kv[1], reverse=True)
        kept = [(k, s) for k, s in ranked if s >= args.min_legibility][:args.max_keep]
        for k, s in kept:
            index.append({"image_path": k, "jersey": jersey, "legibility": float(s),
                          "tracklet_id": pid})

        if (n_done + 1) % 100 == 0:
            print(f"  {n_done + 1}/{len(visible)} tracklets, {len(index)} legible frames so far")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(index, f)
    n_tracklets = len({e['tracklet_id'] for e in index})
    print(f"Saved {len(index)} legible frames from {n_tracklets} tracklets "
          f"({skipped} skipped) -> {out}")


if __name__ == "__main__":
    main()
