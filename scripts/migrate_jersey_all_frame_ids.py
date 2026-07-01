"""
Migrate existing jersey dataset tracklets.json to add all_frame_ids.
Reads GT detections per sequence and stores the full temporal range per tracklet.
Does NOT regenerate crops.

Usage:
    python scripts/migrate_jersey_all_frame_ids.py \
        --dataset_dir datasets/jersey_tracking_v1 \
        --gt_root data/tracking/SoccerNet/tracking
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_gt_for_frames(gt_path, min_area=400):
    """Return {track_id: [frame_ids]} from gt.txt."""
    df = pd.read_csv(
        gt_path, header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis", "_extra"],
    )
    df = df[df["conf"] > 0]
    df["area"] = df["w"] * df["h"]
    df = df[df["area"] >= min_area]
    frames_by_tid = defaultdict(list)
    for _, row in df.iterrows():
        tid = int(row["track_id"])
        frames_by_tid[tid].append(int(row["frame"]))
    return frames_by_tid


def main():
    parser = argparse.ArgumentParser(description="Migrate dataset to add all_frame_ids")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--gt_root", type=str, required=True,
                        help="Root dir with splits containing sequences")
    parser.add_argument("--min_bbox_area", type=int, default=400)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    tracklets_path = dataset_dir / "tracklets.json"
    metadata_path = dataset_dir / "metadata.csv"

    with open(tracklets_path) as f:
        records = json.load(f)

    gt_root = Path(args.gt_root)

    # Determine unique sequences and locate their GT paths
    seq_to_gt = {}
    for rec in records:
        seq = rec["sequence"]
        if seq in seq_to_gt:
            continue
        # Search under gt_root/train, test, challenge
        found = False
        for split in ["train", "test", "challenge"]:
            split_dir = gt_root / split
            if not split_dir.exists():
                continue
            for game_dir in split_dir.iterdir():
                if not game_dir.is_dir():
                    continue
                candidate = game_dir / seq / "gt" / "gt.txt"
                if candidate.exists():
                    seq_to_gt[seq] = candidate
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"  [WARNING] GT not found for sequence {seq}")

    # Process each unique sequence once and cache frames_by_tid
    seq_cache = {}
    for seq, gt_path in seq_to_gt.items():
        seq_cache[seq] = parse_gt_for_frames(str(gt_path), min_area=args.min_bbox_area)

    migrated = 0
    for rec in records:
        seq = rec["sequence"]
        tid = rec["track_id"]
        frames_by_tid = seq_cache.get(seq)
        if frames_by_tid is None:
            continue
        all_frames = sorted(set(frames_by_tid.get(tid, [])))
        if all_frames:
            rec["all_frame_ids"] = all_frames
            migrated += 1

    print(f"Migrated {migrated}/{len(records)} tracklets with all_frame_ids")

    # Rewrite tracklets.json
    with open(tracklets_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {tracklets_path}")

    # Regenerate metadata.csv flat
    flat_records = []
    for rec in records:
        flat = {k: v for k, v in rec.items() if k not in ("frame_ids", "all_frame_ids", "crop_paths", "quality_scores")}
        flat["num_top_frames"] = len(rec["frame_ids"])
        flat["num_all_frames"] = len(rec.get("all_frame_ids", []))
        flat_records.append(flat)
    df = pd.DataFrame(flat_records)
    df.to_csv(metadata_path, index=False)
    print(f"Saved metadata: {metadata_path} ({len(df)} tracklets)")


if __name__ == "__main__":
    main()
