"""
Build jersey-tracking dataset from SoccerNet Tracking sequences.
Extracts crops per tracklet, computes quality scores, builds top-K frame bags.

Usage:
    python training/identification/build_tracking_jersey_dataset.py \
        --root_dir data/tracking/SoccerNet/tracking \
        --output_dir datasets/jersey_tracking_v1 \
        --img_size 128 \
        --max_frames_per_tracklet 24 \
        --min_frames_per_tracklet 8 \
        --min_bbox_area 400 \
        --train_game_ids 4,6,9 \
        --val_game_ids 4 \
        --test_game_ids 7,8,11 \
        --num_workers 4
"""

import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


CLASS_LABEL_MAP = {
    "player team left": 0,
    "player team right": 1,
    "goalkeepers team left": 0,
    "goalkeepers team right": 1,
    "goalkeeper team left": 0,
    "goalkeeper team right": 1,
}


def parse_gameinfo(gameinfo_path):
    tracklet_info = {}
    game_id = None
    with open(gameinfo_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("gameID="):
            try:
                game_id = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        if not line.startswith("trackletID_"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        tid_str = key.split("_")[1]
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        if ";" not in val:
            continue
        team_desc, jersey_raw = val.rsplit(";", 1)
        team_desc = team_desc.strip().lower()
        jersey_raw = jersey_raw.strip()

        # Only players and goalkeepers
        if not ("player" in team_desc or "goalkeeper" in team_desc):
            continue

        try:
            jersey_number = int(jersey_raw)
        except ValueError:
            jersey_number = -1
        is_goalkeeper = "goalkeeper" in team_desc
        tracklet_info[tid] = {
            "team": CLASS_LABEL_MAP.get(team_desc, -1),
            "jersey_number": jersey_number,
            "is_goalkeeper": is_goalkeeper,
        }
    return tracklet_info, game_id


def parse_gt(gt_path, min_area=400):
    detections = defaultdict(list)
    df = pd.read_csv(
        gt_path,
        header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis", "_extra"],
    )
    df = df[df["conf"] > 0]
    df["area"] = df["w"] * df["h"]
    df = df[df["area"] >= min_area]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    for _, row in df.iterrows():
        detections[int(row["frame"])].append(
            {
                "track_id": int(row["track_id"]),
                "bbox": [float(row["x"]), float(row["y"]), float(row["x2"]), float(row["y2"])],
                "area": float(row["area"]),
            }
        )
    return detections


def extract_torso_crop(image, bbox, pad=0.05):
    x1, y1, x2, y2 = bbox
    x1 -= (x2 - x1) * pad
    y1 -= (y2 - y1) * pad
    x2 += (x2 - x1) * pad
    y2 += (y2 - y1) * pad
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None, None, None
    top = int(y1 + (y2 - y1) * 0.10)
    bottom = int(y1 + (y2 - y1) * 0.70)
    left = int(x1 + (x2 - x1) * 0.10)
    right = int(x1 + (x2 - x1) * 0.90)
    if top >= bottom or left >= right:
        return None, None, None
    crop = image[top:bottom, left:right]
    return crop, (x1, y1, x2, y2), (left, top, right, bottom)


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.identity.crops import compute_crop_quality

def compute_quality_score(crop):
    return compute_crop_quality(crop)


def process_sequence(seq_dir, tracklet_info, img_size, max_k, min_frames, min_bbox_area, crops_dir):
    img1_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists() or not img1_dir.exists():
        return [], {}

    detections_by_frame = parse_gt(str(gt_path), min_area=min_bbox_area)
    if not detections_by_frame:
        return [], {}

    seq_name = seq_dir.name
    seq_crop_dir = crops_dir / seq_name
    seq_crop_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: accumulate all frame candidates with quality scores (in-memory, no disk I/O)
    tracklets = defaultdict(lambda: {"frames": [], "bboxes": [], "quality_scores": [], "crops": []})
    # All detected frames per track (before crop quality filters) for accurate temporal range
    all_detected_frames = defaultdict(list)

    for frame_num in sorted(detections_by_frame.keys()):
        frame_path = img1_dir / f"{frame_num:06d}.jpg"
        if not frame_path.exists():
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        for det in detections_by_frame[frame_num]:
            tid = det["track_id"]
            if tid not in tracklet_info:
                continue
            ti = tracklet_info[tid]
            if ti["jersey_number"] < 0:
                continue
            # Register frame for full temporal range (before quality filters)
            all_detected_frames[tid].append(frame_num)

            crop, _, _ = extract_torso_crop(image, det["bbox"])
            if crop is None:
                continue
            quality = compute_quality_score(crop)
            if quality <= 0:
                continue

            tracklets[tid]["frames"].append(frame_num)
            tracklets[tid]["bboxes"].append(det["bbox"])
            tracklets[tid]["quality_scores"].append(quality)
            tracklets[tid]["crops"].append(crop)

    # Phase 2: per tracklet, pick top-K, save only those crops to disk
    results = []
    for tid, tdata in tracklets.items():
        if len(tdata["frames"]) < min_frames:
            continue
        ti = tracklet_info[tid]
        sorted_indices = sorted(
            range(len(tdata["quality_scores"])),
            key=lambda i: tdata["quality_scores"][i],
            reverse=True,
        )
        top_indices = sorted_indices[:max_k]

        crop_paths = []
        frame_ids = []
        quality_scores = []
        for idx in top_indices:
            frame_num = tdata["frames"][idx]
            crop = tdata["crops"][idx]
            crop_resized = cv2.resize(crop, (img_size, img_size))
            crop_filename = f"{tid}_{frame_num:06d}.jpg"
            crop_path = seq_crop_dir / crop_filename
            cv2.imwrite(str(crop_path), crop_resized)
            crop_paths.append(str(crop_path.relative_to(crops_dir.parent)))
            frame_ids.append(int(frame_num))
            quality_scores.append(float(tdata["quality_scores"][idx]))

        results.append(
            {
                "track_id": tid,
                "sequence": seq_name,
                "team_id": ti["team"],
                "jersey_number": ti["jersey_number"],
                "is_goalkeeper": ti["is_goalkeeper"],
                "num_frames": len(tdata["frames"]),
                "all_frame_ids": sorted(set(int(f) for f in all_detected_frames.get(tid, []))),
                "frame_ids": frame_ids,
                "crop_paths": crop_paths,
                "quality_scores": quality_scores,
                "state": "unknown",
                "predicted_number": None,
                "confidence": None,
            }
        )
    return results, detections_by_frame


def build_dataset(root_dir, output_dir, img_size, max_k, min_frames, min_bbox_area, game_splits, num_workers, sequences=None):
    root = Path(root_dir)
    output = Path(output_dir)
    crops_dir = output / "crops"
    metadata_path = output / "metadata.csv"
    tracklets_path = output / "tracklets.json"
    splits_path = output / "splits.json"

    all_records = []
    seq_dirs = []

    for split in ["train", "test", "challenge"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for game_dir in split_dir.iterdir():
            if not game_dir.is_dir():
                continue
            for seq_dir in game_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                if sequences and seq_dir.name not in sequences:
                    continue
                seq_dirs.append(seq_dir)

    print(f"Processing {len(seq_dirs)} sequences...")
    for seq_dir in tqdm(seq_dirs):
        gameinfo = seq_dir / "gameinfo.ini"
        if not gameinfo.exists():
            continue
        tracklet_info, game_id = parse_gameinfo(gameinfo)
        records, _ = process_sequence(
            seq_dir, tracklet_info, img_size, max_k, min_frames, min_bbox_area, crops_dir
        )
        for rec in records:
            rec["game_id"] = game_id
            all_records.append(rec)

    if not all_records:
        print("No tracklets found. Check paths and min_frames/min_bbox_area.")
        return

    # Save tracklets.json (source of truth)
    with open(tracklets_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved tracklets: {tracklets_path}")

    # Save metadata.csv (flat, without nested lists)
    flat_records = []
    for rec in all_records:
        flat = {k: v for k, v in rec.items() if k not in ("frame_ids", "all_frame_ids", "crop_paths", "quality_scores")}
        flat["num_top_frames"] = len(rec["frame_ids"])
        flat["num_all_frames"] = len(rec.get("all_frame_ids", []))
        flat_records.append(flat)
    df = pd.DataFrame(flat_records)
    df.to_csv(metadata_path, index=False)
    print(f"Saved metadata: {metadata_path} ({len(df)} tracklets)")

    # Build splits
    split_assignments = {}
    for split_name, game_ids in game_splits.items():
        split_assignments[split_name] = {
            rec["sequence"]: rec["game_id"] in game_ids
            for rec in all_records
            if rec["game_id"] is not None
        }

    with open(splits_path, "w") as f:
        json.dump(split_assignments, f, indent=2)
    print(f"Saved splits: {splits_path}")

    print(f"\nDataset summary:")
    print(f"  Total tracklets: {len(all_records)}")
    print(f"  Numeric jersey numbers: {sum(1 for r in all_records if r['jersey_number'] >= 0)}")
    print(f"  Unknown jerseys: {sum(1 for r in all_records if r['jersey_number'] < 0)}")
    print(f"  By team 0: {sum(1 for r in all_records if r['team_id'] == 0)}")
    print(f"  By team 1: {sum(1 for r in all_records if r['team_id'] == 1)}")
    print(f"  Goalkeepers: {sum(1 for r in all_records if r['is_goalkeeper'])}")
    for split, games in game_splits.items():
        count = sum(1 for r in all_records if r["game_id"] in games)
        print(f"  {split} (games {games}): {count} tracklets")
    print(f"  Crops saved to: {crops_dir}")
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build jersey-tracking dataset from SoccerNet")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--max_frames_per_tracklet", type=int, default=24)
    parser.add_argument("--min_frames_per_tracklet", type=int, default=8)
    parser.add_argument("--min_bbox_area", type=int, default=400)
    parser.add_argument("--train_game_ids", type=str, default="4,6,9")
    parser.add_argument("--val_game_ids", type=str, default="4")
    parser.add_argument("--test_game_ids", type=str, default="7,8,11")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sequences", type=str, default=None, help="Comma-separated sequence names to process (default: all)")
    args = parser.parse_args()

    seq_filter = None
    if args.sequences:
        seq_filter = set(args.sequences.split(","))
    game_splits = {
        "train": set(int(x) for x in args.train_game_ids.split(",")),
        "val": set(int(x) for x in args.val_game_ids.split(",")),
        "test": set(int(x) for x in args.test_game_ids.split(",")),
    }

    build_dataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        max_k=args.max_frames_per_tracklet,
        min_frames=args.min_frames_per_tracklet,
        min_bbox_area=args.min_bbox_area,
        game_splits=game_splits,
        num_workers=args.num_workers,
        sequences=seq_filter,
    )