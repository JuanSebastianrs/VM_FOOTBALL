"""
Production-style (temporal) jersey evaluation over dataset splits.

This is the multi-sequence generalization gate: unlike infer_jersey_on_dataset.py
(MIL bags of K=16), this script replicates the PRODUCTION inference path of
core/identity/jersey_identity_phase.py — per-frame digit inference + legibility
filtering + temporal fusion + per-sequence assignment — over every sequence of a
split (val: calibration, test: 49-sequence generalization report).

Per-frame probabilities are cached to disk (.npz) so fusion/threshold sweeps are
near-instant and never re-run GPU inference.

Usage:
    # Single config
    python scripts/evaluate_jersey_dataset_temporal.py \
        --dataset_dir datasets/jersey_tracking_v1 \
        --model_path runs/jersey_perframe_v1/best.pt \
        --split test --output_dir outputs/jersey_eval/test_perframe_v1

    # Calibration sweep on val (uses cache after first run)
    python scripts/evaluate_jersey_dataset_temporal.py ... --split val --sweep
"""

import sys
import argparse
import json
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.identity.jersey_model import (
    get_model_transform,
    load_jersey_model,
    compute_jersey_probs_from_logits,
)
from core.identity.jersey_identity_phase import temporal_fusion
from core.identity.jersey_assignment import TrackletInfo, assign_per_team
from scripts.evaluate_jersey_tracklets import compute_tracklet_accuracy
from scripts.infer_jersey_on_dataset import build_splits


@torch.no_grad()
def compute_frame_probs(model, records, dataset_dir, device, batch_size=128):
    """
    Per-frame digit inference on ALL crops of each tracklet.
    Returns dict: "seq|tid" -> np.ndarray (T, 99) aligned with crop_paths.
    """
    model.eval()
    digit_tf = get_model_transform(model)
    root = Path(dataset_dir)
    out = {}
    for i, rec in enumerate(records):
        key = f"{rec['sequence']}|{rec['track_id']}"
        tensors = []
        for cp in rec.get("crop_paths", []):
            img_path = root / cp
            if not img_path.exists():
                raise FileNotFoundError(f"Missing crop: {img_path}")
            img = np.array(Image.open(img_path).convert("RGB"))
            tensors.append(digit_tf(img))
        probs_list = []
        for j in range(0, len(tensors), batch_size):
            x = torch.stack(tensors[j:j + batch_size]).unsqueeze(1).to(device)  # (B,1,3,H,W)
            out_len, out_tens, out_ones = model(x)
            p = np.atleast_2d(compute_jersey_probs_from_logits(out_len, out_tens, out_ones))
            probs_list.append(p)
        out[key] = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 99))
        if (i + 1) % 100 == 0:
            print(f"  inference {i + 1}/{len(records)} tracklets")
    return out


def build_frame_results(rec, probs, legibility_scores):
    """Assemble per-frame result dicts in the format temporal_fusion expects."""
    quality = rec.get("quality_scores", [])
    frame_results = []
    for idx, cp in enumerate(rec.get("crop_paths", [])):
        if idx >= len(probs):
            break
        rel = Path(cp).as_posix()
        frame_results.append({
            "frame_id": idx,
            "jersey_probs": probs[idx],
            "quality_score": float(quality[idx]) if idx < len(quality) else 1.0,
            "legibility_score": float(legibility_scores.get(rel, 1.0)),
        })
    return frame_results


def evaluate_config(records, frame_probs, legibility_scores, config, team_rosters_by_seq=None,
                    reassign_conflicts=False):
    """Fuse cached per-frame probs with one config and compute split metrics."""
    by_seq = defaultdict(list)
    fused_lookup = {}

    for rec in records:
        key = f"{rec['sequence']}|{rec['track_id']}"
        probs = frame_probs.get(key)
        if probs is None or len(probs) == 0:
            continue
        frame_results = build_frame_results(rec, probs, legibility_scores)

        # Production legibility gate: filter if enough legible frames remain
        filtered = [fr for fr in frame_results
                    if fr["legibility_score"] >= config["legibility_threshold"]]
        if len(filtered) >= config["min_legible_frames"]:
            frame_results = filtered

        fused = temporal_fusion(
            frame_results,
            p1_threshold=config["p1_threshold"],
            margin_threshold=config["margin_threshold"],
            roster_set=None,  # roster applied in assign_per_team
            min_peak_quality=0.3,
            fusion_mode=config["fusion_mode"],
            temperature=config["temperature"],
        )
        ti = TrackletInfo(
            track_id=int(rec["track_id"]),
            team_id=int(rec.get("team_id", -1)),
            jersey_probs=fused["probs"],
            num_frames=int(rec.get("num_frames", len(rec.get("crop_paths", [])))),
            state="unknown",
            frame_ids=rec.get("all_frame_ids", rec.get("frame_ids", [])),
        )
        ti.alternatives = fused["alternatives"]
        by_seq[rec["sequence"]].append(ti)
        fused_lookup[(rec["sequence"], int(rec["track_id"]))] = fused

    pred_records = {}
    for seq, seq_tracklets in by_seq.items():
        team_rosters = team_rosters_by_seq.get(seq) if team_rosters_by_seq else None
        assignments = assign_per_team(
            seq_tracklets,
            team_rosters=team_rosters,
            p1_threshold=config["p1_threshold"],
            margin_threshold=config["margin_threshold"],
            reassign_conflicts=reassign_conflicts,
        )
        ti_lookup = {ti.track_id: ti for ti in seq_tracklets}
        for tid, asn in assignments.items():
            pred_records[(seq, tid)] = {
                "predicted_number": asn.get("predicted_number"),
                "confidence": asn.get("confidence", 0.0),
                "state": asn.get("state", "unknown"),
                "team_id": asn.get("team_id", -1),
                "alternatives": getattr(ti_lookup.get(tid), "alternatives", []),
                "raw_top1": asn.get("raw_top1"),
            }

    gt_records = {}
    for rec in records:
        jersey = int(rec.get("jersey_number", -1))
        if jersey < 1:
            continue
        gt_records[(rec["sequence"], int(rec["track_id"]))] = {
            "jersey_number": jersey,
            "team_id": int(rec.get("team_id", -1)),
            "sequence": rec["sequence"],
        }

    return compute_tracklet_accuracy(pred_records, gt_records)


def main():
    parser = argparse.ArgumentParser(description="Temporal (production-style) jersey eval on dataset splits")
    parser.add_argument("--dataset_dir", type=str, default="datasets/jersey_tracking_v1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--legibility_scores", type=str, default=None,
                        help="Default: <dataset_dir>/legibility_scores.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--roster_json", type=str, default=None)
    # Fusion config (single-run mode)
    parser.add_argument("--fusion_mode", type=str, default="geometric",
                        choices=["geometric", "arithmetic", "topk_geometric", "confidence_topk"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--legibility_threshold", type=float, default=0.70)
    parser.add_argument("--min_legible_frames", type=int, default=4)
    parser.add_argument("--p1_threshold", type=float, default=0.75)
    parser.add_argument("--margin_threshold", type=float, default=0.15)
    parser.add_argument("--reassign_conflicts", action="store_true",
                        help="Conflict losers fall back to best non-conflicting alternative")
    parser.add_argument("--parseq_cache", type=str, default=None,
                        help="npz de build_parseq_cache.py para ensamble")
    parser.add_argument("--parseq_weight", type=float, default=0.35)
    parser.add_argument("--sweep", action="store_true",
                        help="Grid-sweep fusion configs from the cached probs (calibration; use on val)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    leg_path = Path(args.legibility_scores) if args.legibility_scores else dataset_dir / "legibility_scores.json"
    with open(leg_path) as f:
        legibility_scores = json.load(f)

    records = build_splits(dataset_dir / "tracklets.json", dataset_dir / "splits.json")
    split_records = [r for r in records if r.get("split") == args.split]
    print(f"{args.split} split: {len(split_records)} tracklets, "
          f"{len({r['sequence'] for r in split_records})} sequences")

    # --- Per-frame probs: load cache or run inference once ---
    model_tag = Path(args.model_path).parent.name
    cache_path = output_dir / f"frame_probs_{args.split}_{model_tag}.npz"
    if cache_path.exists():
        print(f"Loading cached per-frame probs: {cache_path}")
        npz = np.load(cache_path)
        frame_probs = {k: npz[k] for k in npz.files}
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Loading model {args.model_path} on {device}...")
        model = load_jersey_model(args.model_path, device, strict=True)
        print("Running per-frame inference (cached afterwards)...")
        frame_probs = compute_frame_probs(model, split_records, dataset_dir, device)
        np.savez_compressed(cache_path, **frame_probs)
        print(f"Cached: {cache_path}")

    # --- Optional PARSeq ensemble: p ∝ p_model^(1-w) * p_parseq^w por frame ---
    if args.parseq_cache and Path(args.parseq_cache).exists():
        w = float(args.parseq_weight)
        pz = np.load(args.parseq_cache)
        mixed = 0
        for k in list(frame_probs.keys()):
            if k not in pz.files:
                continue
            pm, pp = frame_probs[k].copy(), pz[k]
            n = min(len(pm), len(pp))
            if n == 0:
                continue
            # solo mezclar frames donde PARSeq LEYO algo (fila no-uniforme);
            # mezclar contra uniforme solo aplana el posterior del modelo
            read = pp[:n].max(axis=1) > (1.5 / 99.0)
            if read.any():
                mix = (np.power(pm[:n][read] + 1e-12, 1.0 - w)
                       * np.power(pp[:n][read] + 1e-12, w))
                pm[:n][read] = mix / mix.sum(axis=1, keepdims=True)
                frame_probs[k] = pm
                mixed += 1
        print(f"Ensamble PARSeq (w={w}): {mixed} tracklets mezclados")

    # --- Optional rosters (external_roster mode; primary metric is no-roster) ---
    team_rosters_by_seq = None
    if args.roster_json:
        with open(args.roster_json) as f:
            all_rosters = json.load(f)
        team_rosters_by_seq = {}
        for seq in {r["sequence"] for r in split_records}:
            if seq not in all_rosters:
                continue
            seq_roster = all_rosters[seq]
            rosters = {}
            for tid, side in {0: "left", 1: "right"}.items():  # dataset-native mapping
                side_data = seq_roster.get(side, {})
                nums = set(side_data.get("field_players", []) + side_data.get("goalkeepers", []))
                if nums:
                    rosters[tid] = nums
            if rosters:
                team_rosters_by_seq[seq] = rosters

    if not args.sweep:
        config = {
            "fusion_mode": args.fusion_mode,
            "temperature": args.temperature,
            "legibility_threshold": args.legibility_threshold,
            "min_legible_frames": args.min_legible_frames,
            "p1_threshold": args.p1_threshold,
            "margin_threshold": args.margin_threshold,
        }
        metrics = evaluate_config(split_records, frame_probs, legibility_scores, config,
                                  team_rosters_by_seq=team_rosters_by_seq,
                                  reassign_conflicts=args.reassign_conflicts)
        summary = {k: v for k, v in metrics.items() if k != "errors"}
        print(f"\n=== {args.split} | {config['fusion_mode']} T={config['temperature']} "
              f"leg>={config['legibility_threshold']} p1={config['p1_threshold']} "
              f"m={config['margin_threshold']} ===")
        print(f"  raw top-1 (matched): {summary['raw_top1_matched_acc']:.1%}")
        print(f"  raw top-3 (matched): {summary['raw_top3_matched_acc']:.1%}")
        print(f"  assigned (matched):  {summary['assigned_matched_acc']:.1%}")
        print(f"  locked: {summary['locked_total']} | locked acc: {summary['locked_accuracy']:.1%} "
              f"| false lock: {summary['false_lock_rate']:.1%}")
        out_json = output_dir / f"temporal_eval_{args.split}.json"
        with open(out_json, "w") as f:
            json.dump({"config": config, "metrics": summary}, f, indent=2)
        with open(output_dir / f"temporal_eval_{args.split}_errors.json", "w") as f:
            json.dump(metrics["errors"], f, indent=2)
        print(f"Saved: {out_json}")
        return

    # --- Sweep mode (calibration) ---
    import pandas as pd
    grid = {
        "fusion_mode": ["geometric", "arithmetic", "topk_geometric", "confidence_topk"],
        "temperature": [1.0, 1.5, 2.0, 3.0],
        "legibility_threshold": [0.40, 0.50, 0.60, 0.70, 0.80],
        "p1_threshold": [0.60, 0.75, 0.85],
        "margin_threshold": [0.15],
        "min_legible_frames": [4],
    }
    keys = list(grid.keys())
    rows = []
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Sweeping {len(combos)} configs on {args.split}...")
    for vals in combos:
        config = dict(zip(keys, vals))
        m = evaluate_config(split_records, frame_probs, legibility_scores, config,
                            team_rosters_by_seq=team_rosters_by_seq)
        rows.append({
            **config,
            "raw_top1": m["raw_top1_matched_acc"],
            "raw_top3": m["raw_top3_matched_acc"],
            "assigned": m["assigned_matched_acc"],
            "locked_total": m["locked_total"],
            "locked_acc": m["locked_accuracy"],
            "false_lock_rate": m["false_lock_rate"],
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"fusion_sweep_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    ranked = df.sort_values(by=["raw_top1", "assigned"], ascending=False)
    print(ranked.head(20).to_string(index=False))
    print(f"\nSaved sweep: {csv_path}")


if __name__ == "__main__":
    main()
