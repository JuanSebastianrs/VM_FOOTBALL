"""
Run inference on the built dataset (tracklets.json) for evaluation.
Outputs a prediction JSON that can be directly compared with ground truth
using the same track IDs, avoiding pipeline↔GT ID mismatch issues.

Usage:
    python scripts/infer_jersey_on_dataset.py \
        --dataset_dir datasets/jersey_tracking_v1 \
        --model_path runs/jersey_digit_mil/best.pt \
        --split val \
        --output_json outputs/jersey_eval/val_predictions.json \
        --K 16 \
        --device cuda:0
"""

import sys
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.identity.jersey_model import (
    DigitCompositionalMIL,
    TRANSFORM_INFERENCE,
    get_model_transform,
    load_jersey_model,
)
from core.identity.jersey_assignment import TrackletInfo, assign_per_team


def load_model(model_path, device):
    """Load jersey model using centralized loader (strict=True)."""
    return load_jersey_model(model_path, device, strict=True)


def build_splits(tracklets_path, splits_json_path, val_ratio=0.15):
    with open(tracklets_path) as f:
        records = json.load(f)
    with open(splits_json_path) as f:
        splits_data = json.load(f)
    train_seqs = {s for s, v in splits_data.get("train", {}).items() if v}
    val_seqs = {s for s, v in splits_data.get("val", {}).items() if v}
    test_seqs = {s for s, v in splits_data.get("test", {}).items() if v}

    import hashlib
    def _is_val(seq):
        h = int(hashlib.md5(seq.encode()).hexdigest(), 16)
        return (h % 100) < int(val_ratio * 100)

    for rec in records:
        seq = rec["sequence"]
        if seq in test_seqs:
            rec["split"] = "test"
        elif seq in val_seqs and _is_val(seq):
            rec["split"] = "val"
        elif seq in train_seqs:
            rec["split"] = "train"
        else:
            rec["split"] = "unknown"
    return [r for r in records if r["split"] != "unknown"]


@torch.no_grad()
def infer_on_records(model, records, dataset_dir, device, K=16, seed=42):
    model.eval()
    digit_tf = get_model_transform(model)
    root = Path(dataset_dir)
    results = {}
    for rec in records:
        tid = rec["track_id"]
        seq = rec["sequence"]
        crop_paths = rec.get("crop_paths", [])[:K]
        crops = []
        for cp in crop_paths:
            img_path = root / cp
            if not img_path.exists():
                raise FileNotFoundError(f"Missing crop: {img_path}")
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            crop_tensor = digit_tf(img)
            crops.append(crop_tensor)
        if len(crops) == 0:
            raise RuntimeError(f"No crops for {seq}/{tid}")
        rng = random.Random(seed + tid)
        while len(crops) < K:
            crops.append(crops[rng.randrange(len(crops))])
        x = torch.stack(crops[:K]).unsqueeze(0).to(device)
        out_len, out_tens, out_ones = model(x)

        len_probs = F.softmax(out_len, dim=1).squeeze(0).cpu().numpy()
        tens_probs = F.softmax(out_tens, dim=1).squeeze(0).cpu().numpy()
        ones_probs = F.softmax(out_ones, dim=1).squeeze(0).cpu().numpy()

        jersey_probs = np.zeros(99, dtype=np.float64)
        for t_digit in range(10):
            for o_digit in range(10):
                num = t_digit * 10 + o_digit
                if num == 0 or num >= 100:
                    continue
                idx = num - 1
                if num <= 9:
                    jersey_probs[idx] = len_probs[0] * ones_probs[o_digit]
                else:
                    jersey_probs[idx] = len_probs[1] * tens_probs[t_digit] * ones_probs[o_digit]

        valid_mass = jersey_probs.sum()
        if valid_mass > 1e-8:
            jersey_probs = jersey_probs / valid_mass
        else:
            jersey_probs = np.zeros(99, dtype=np.float64)

        topk_idx = np.argsort(jersey_probs)[::-1][:5]
        alternatives = [(int(idx + 1), float(jersey_probs[idx])) for idx in topk_idx]
        results[(seq, tid)] = {
            "sequence": seq,
            "track_id": tid,
            "team_id": rec.get("team_id", -1),
            "jersey_probs": jersey_probs,
            "alternatives": alternatives,
            "gt_jersey_number": rec.get("jersey_number"),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Infer jersey numbers on dataset tracklets")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--p1_threshold", type=float, default=0.85)
    parser.add_argument("--margin_threshold", type=float, default=0.20)
    parser.add_argument("--roster_json", type=str, default=None,
                        help="Roster JSON from extract_rosters.py")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic crop padding")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    tracklets_path = Path(args.dataset_dir) / "tracklets.json"
    splits_path = Path(args.dataset_dir) / "splits.json"
    records = build_splits(tracklets_path, splits_path)
    split_records = [r for r in records if r.get("split") == args.split]
    print(f"Running inference on {len(split_records)} {args.split} tracklets...")

    inferred = infer_on_records(model, split_records, args.dataset_dir, device, K=args.K, seed=args.seed)

    # Build TrackletInfo objects and assign per (sequence, team)
    # Key fix: group by SEQUENCE first, then by team within sequence
    by_seq = defaultdict(list)
    ti_lookup = {}  # (seq, tid) -> TrackletInfo
    for key, data in inferred.items():
        seq, tid = key
        # Find the original record to get frame_ids
        orig_rec = next((r for r in split_records
                         if r["sequence"] == seq and r["track_id"] == tid), {})
        ti = TrackletInfo(
            track_id=data["track_id"],
            team_id=data["team_id"],
            jersey_probs=data["jersey_probs"],
            num_frames=orig_rec.get("num_frames", len(orig_rec.get("crop_paths", []))),
            state="unknown",
            frame_ids=orig_rec.get("all_frame_ids", orig_rec.get("frame_ids", [])),
        )
        ti.alternatives = data["alternatives"]
        ti_lookup[key] = ti
        by_seq[seq].append(ti)

    # Load rosters if provided
    all_rosters = None
    if args.roster_json:
        with open(args.roster_json) as f:
            all_rosters = json.load(f)
        print(f"  Rosters loaded for {len(all_rosters)} sequences")

    # Assign per-sequence (NOT globally, to avoid track_id collision)
    assignments = {}  # (seq, tid) -> assignment dict
    for seq, seq_tracklets in by_seq.items():
        # Build per-sequence roster if available
        team_rosters = None
        if all_rosters and seq in all_rosters:
            seq_roster = all_rosters[seq]
            # Dataset-native mapping (verified by majority vote):
            #   team_id=0 → left,  team_id=1 → right
            dataset_team_to_side = {0: "left", 1: "right"}
            team_rosters = {}
            for tid, side_name in dataset_team_to_side.items():
                side_data = seq_roster.get(side_name, {})
                nums = set(side_data.get("field_players", []) + side_data.get("goalkeepers", []))
                if nums:
                    team_rosters[tid] = nums

        seq_assignments = assign_per_team(
            seq_tracklets,
            team_rosters=team_rosters,
            p1_threshold=args.p1_threshold,
            margin_threshold=args.margin_threshold,
        )
        for tid, asn in seq_assignments.items():
            assignments[(seq, tid)] = asn

    output_tracklets = []
    for key, data in inferred.items():
        seq, tid = key
        asn = assignments.get((seq, tid), {})
        output_tracklets.append({
            "sequence": seq,
            "track_id": int(tid),
            "team_id": int(data["team_id"]),
            "predicted_number": int(asn["predicted_number"]) if asn.get("predicted_number") is not None else None,
            "confidence": float(asn.get("confidence", 0.0)),
            "state": str(asn.get("state", "unknown")),
            "p1": float(asn.get("p1", 0.0)),
            "margin": float(asn.get("margin", 0.0)),
            "alternatives": [(int(n), float(p)) for n, p in data["alternatives"]],
            "gt_jersey_number": int(data["gt_jersey_number"]) if data["gt_jersey_number"] is not None else None,
        })

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"tracklets": output_tracklets}, f, indent=2)
    print(f"Saved predictions: {args.output_json}")


if __name__ == "__main__":
    main()
