"""
Sweep Jersey Thresholds: Grid search over jersey confidence, margin, and legibility thresholds
using in-memory caching of predictions for extreme speed, strict holdout guard, and subprocess safety.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import pipeline operations
from core.identity.jersey_model import load_jersey_model, load_legibility_model
from core.identity.jersey_identity_phase import build_tracklet_frames, predict_per_frame, temporal_fusion
from core.identity.jersey_assignment import TrackletInfo, assign_per_team
from scripts.evaluate_jersey_e2e import (
    _safe_rate,
    parse_gameinfo,
    parse_gt_txt,
    load_pred_detections,
    match_per_frame,
)


def main():
    parser = argparse.ArgumentParser(description="Sweep Jersey Thresholds (In-Memory Fast)")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--detections_json", type=str, required=True)
    parser.add_argument("--team_assignments_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--legibility_model", type=str, required=True)
    parser.add_argument("--roster_json", type=str, default=None)
    parser.add_argument("--team_mapping", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_csv", type=str, default="outputs/SNMOT-148/sweep_results.csv")
    parser.add_argument("--allow_holdout_sweep", action="store_true", help="Explicitly allow sweep on holdout SNMOT-148")
    parser.add_argument("--multi_crop", action="store_true", help="Enable multi-crop candidates in sweep")
    args = parser.parse_args()

    # 1. Strict Holdout Guard
    seq_name = Path(args.sequence_dir).name
    if "SNMOT-148" in seq_name and not args.allow_holdout_sweep:
        print(f"\n=======================================================")
        print(f"CRITICAL ERROR: Sweeping thresholds on the holdout sequence")
        print(f"SNMOT-148 is NOT allowed without --allow_holdout_sweep flag.")
        print(f"=======================================================\n")
        sys.exit(1)

    # 2. Resolve team mapping
    tm_arg = args.team_mapping
    if not tm_arg:
        output_dir = Path(args.output_csv).parent
        audit_path = output_dir / "team_audit.json"
        if audit_path.exists():
            tm_arg = str(audit_path)
        else:
            tm_arg = "0:right,1:left"  # fallback
            
    print(f"Resolved team mapping for sweep: {tm_arg}")

    # Load spatial matching and GT data exactly once
    gi_path = Path(args.sequence_dir) / "gameinfo.ini"
    gt_path = Path(args.sequence_dir) / "gt" / "gt.txt"
    if not gi_path.exists():
        raise FileNotFoundError(f"gameinfo.ini not found: {gi_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    print("Loading GT and spatial mapping...")
    gt_tracklet_info = parse_gameinfo(gi_path)
    gt_detections = parse_gt_txt(gt_path, gt_tracklet_info=gt_tracklet_info)
    pred_detections = load_pred_detections(args.detections_json)

    # Resolve team mapping dictionary
    team_mapping = {}
    mapping_path = Path(tm_arg)
    if mapping_path.exists() and mapping_path.suffix == ".json":
        with open(mapping_path) as f:
            audit = json.load(f)
        team_mapping = {int(k): v for k, v in audit.get("team_mapping", {}).items()}
    else:
        for pair in tm_arg.split(","):
            k, v = pair.strip().split(":")
            team_mapping[int(k)] = v.strip()

    # Pre-compute spatial matching matches exactly once
    pred_to_gt_votes = {}
    all_frames = sorted(set(pred_detections.keys()) | set(gt_detections.keys()))
    for frame in all_frames:
        p_dets = pred_detections.get(frame, [])
        g_dets = gt_detections.get(frame, [])
        matches = match_per_frame(p_dets, g_dets, iou_threshold=0.5)
        for pred_tid, gt_tid, iou in matches:
            if pred_tid not in pred_to_gt_votes:
                pred_to_gt_votes[pred_tid] = {}
            pred_to_gt_votes[pred_tid][gt_tid] = pred_to_gt_votes[pred_tid].get(gt_tid, 0) + 1

    # Resolve each pred track -> best GT track
    pred_to_gt = {}
    for pred_tid, votes in pred_to_gt_votes.items():
        best_gt_tid = max(votes, key=votes.get)
        pred_to_gt[pred_tid] = best_gt_tid

    # 3. Load deep models and run frame prediction exactly ONCE
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading digit model on {device}...")
    model = load_jersey_model(args.model_path, device, strict=True)
    print(f"Loading legibility model from {args.legibility_model}...")
    legibility_model = load_legibility_model(args.legibility_model, device)

    print("Building tracklet frames...")
    tracklets_data = build_tracklet_frames(
        args.detections_json,
        args.team_assignments_json,
        args.sequence_dir,
    )
    print(f"  {len(tracklets_data)} tracklets built.")

    # Load rosters if any
    team_rosters = None
    if args.roster_json:
        with open(args.roster_json) as f:
            all_rosters = json.load(f)
        if seq_name in all_rosters:
            seq_roster = all_rosters[seq_name]
            team_rosters = {}
            for tid, side in team_mapping.items():
                side_data = seq_roster.get(side, {})
                numbers = set(side_data.get("field_players", []) + side_data.get("goalkeepers", []))
                if numbers:
                    team_rosters[tid] = numbers

    # Cache per-frame predictions in memory
    print("Caching per-frame predictions (running E2E model inference once)...")
    cached_frame_results = {}
    for idx, t in enumerate(tracklets_data):
        tid = t["track_id"]
        print(f"  [{idx + 1}/{len(tracklets_data)}] Predicting tracklet {tid} ...")
        # Run prediction with legibility_threshold=0.0 to capture ALL frames
        frs = predict_per_frame(
            model, t, args.sequence_dir, device,
            legibility_model=legibility_model,
            legibility_threshold=0.0,
            multi_crop=args.multi_crop,
        )
        cached_frame_results[tid] = frs

    # 4. Sweep Grids
    p1_grid = [0.75, 0.80, 0.85, 0.90]
    margin_grid = [0.10, 0.15, 0.20, 0.25]
    legibility_grid = [0.40, 0.50, 0.60, 0.70, 0.80]

    results = []
    total_combinations = len(p1_grid) * len(margin_grid) * len(legibility_grid)
    print(f"\nStarting in-memory sweep over {total_combinations} combinations...")

    comb_idx = 0
    for legibility in legibility_grid:
        for p1 in p1_grid:
            for margin in margin_grid:
                comb_idx += 1
                
                # Perform in-memory temporal fusion
                tracklet_infos = []
                for t in tracklets_data:
                    tid = t["track_id"]
                    frame_results = cached_frame_results[tid]
                    
                    # Filter frame results by legibility threshold (min_legible_frames = 4)
                    filtered_frame_results = [fr for fr in frame_results if fr.get("legibility_score", 1.0) >= legibility]
                    if len(filtered_frame_results) >= 4:
                        active_frs = filtered_frame_results
                    else:
                        active_frs = frame_results

                    roster = team_rosters.get(t["team_id"]) if team_rosters else None
                    fused = temporal_fusion(
                        active_frs,
                        p1_threshold=p1,
                        margin_threshold=margin,
                        roster_set=roster,
                        min_peak_quality=0.3,
                    )
                    
                    ti = TrackletInfo(
                        track_id=tid,
                        team_id=t["team_id"],
                        jersey_probs=fused["probs"],
                        num_frames=t["num_frames"],
                        quality_scores=t.get("all_quality_scores", t.get("quality_scores", [])),
                        state="unknown",
                        frame_ids=t.get("all_frame_ids", t.get("frame_ids", [])),
                    )
                    ti.alternatives = fused["alternatives"]
                    tracklet_infos.append(ti)

                # Assign per team
                assignments = assign_per_team(
                    tracklet_infos,
                    team_rosters=team_rosters,
                    p1_threshold=p1,
                    margin_threshold=margin,
                )

                # Compute E2E metrics in memory
                fragment_results = []
                for tid, asn in assignments.items():
                    gt_tid = pred_to_gt.get(tid)
                    if gt_tid is None:
                        continue
                    gt_info = gt_tracklet_info.get(gt_tid)
                    if gt_info is None:
                        continue
                    gt_jersey = gt_info.get("jersey_number", -1)
                    if gt_jersey < 1:
                        continue

                    alts = getattr(next((ti for ti in tracklet_infos if ti.track_id == tid), None), "alternatives", [])
                    raw_top1 = alts[0][0] if len(alts) > 0 else None

                    fragment_results.append({
                        "pred_track_id": tid,
                        "gt_track_id": gt_tid,
                        "gt_jersey": gt_jersey,
                        "predicted_number": asn["predicted_number"],
                        "raw_top1": raw_top1,
                        "state": asn["state"],
                        "confidence": asn["confidence"],
                        "match_frames": int(pred_to_gt_votes[tid][gt_tid]),
                    })

                if not fragment_results:
                    continue

                # Aggregate per-GT tracklet
                gt_to_fragments = {}
                for fr in fragment_results:
                    if fr["gt_track_id"] not in gt_to_fragments:
                        gt_to_fragments[fr["gt_track_id"]] = []
                    gt_to_fragments[fr["gt_track_id"]].append(fr)

                raw_top1_correct_gt = 0
                assigned_correct_gt = 0
                locked_total_gt = 0
                locked_correct_gt = 0
                false_locks_gt = 0
                best_frag_locked_total = 0
                best_frag_locked_correct = 0

                for gt_tid, frags in gt_to_fragments.items():
                    rep = max(frags, key=lambda x: x["match_frames"])
                    gt_jersey = rep["gt_jersey"]
                    pred_num = rep["predicted_number"]
                    raw_top1 = rep["raw_top1"]
                    state = rep["state"]

                    is_raw_correct = (raw_top1 is not None and raw_top1 == gt_jersey)
                    is_assigned_correct = (pred_num is not None and pred_num == gt_jersey)

                    if is_raw_correct:
                        raw_top1_correct_gt += 1
                    if is_assigned_correct:
                        assigned_correct_gt += 1
                    if state == "locked":
                        locked_total_gt += 1
                        if is_assigned_correct:
                            locked_correct_gt += 1
                        elif pred_num is not None:
                            false_locks_gt += 1

                    # Best fragment
                    best_frag = rep
                    for frag in frags:
                        f_pred = frag["predicted_number"]
                        f_raw = frag["raw_top1"]
                        if f_pred is not None and f_pred == gt_jersey:
                            if best_frag["predicted_number"] != gt_jersey or frag["confidence"] > best_frag["confidence"]:
                                best_frag = frag
                        elif f_raw is not None and f_raw == gt_jersey and best_frag["predicted_number"] != gt_jersey:
                            if best_frag["raw_top1"] != gt_jersey or frag["confidence"] > best_frag["confidence"]:
                                best_frag = frag

                    if best_frag["state"] == "locked":
                        best_frag_locked_total += 1
                        if best_frag["predicted_number"] is not None and best_frag["predicted_number"] == gt_jersey:
                            best_frag_locked_correct += 1

                # Calculate final rates
                numeric_gt_total = len([k for k, v in gt_tracklet_info.items() if v.get("jersey_number", -1) >= 1 and v.get("team_side") in ("left", "right")])
                unique_gt_matched = len(gt_to_fragments)
                n_gt = max(unique_gt_matched, 1)
                n_gt_total = max(numeric_gt_total, 1)

                gt_raw_top1_matched_acc = _safe_rate(raw_top1_correct_gt, n_gt)
                gt_assigned_matched_acc = _safe_rate(assigned_correct_gt, n_gt)
                gt_locked_accuracy = _safe_rate(locked_correct_gt, locked_total_gt) if locked_total_gt > 0 else 0.0
                gt_locked_coverage = _safe_rate(locked_total_gt, n_gt)
                gt_false_lock_rate = _safe_rate(false_locks_gt, locked_total_gt) if locked_total_gt > 0 else 0.0
                gt_locked_total_coverage = _safe_rate(locked_total_gt, n_gt_total)
                gt_best_frag_locked_acc = _safe_rate(best_frag_locked_correct, best_frag_locked_total) if best_frag_locked_total > 0 else 0.0

                results.append({
                    "p1_threshold": p1,
                    "margin_threshold": margin,
                    "legibility_threshold": legibility,
                    "false_lock_rate": gt_false_lock_rate,
                    "locked_accuracy": gt_locked_accuracy,
                    "locked_coverage": gt_locked_coverage,
                    "locked_total_coverage": gt_locked_total_coverage,
                    "assigned_accuracy": gt_assigned_matched_acc,
                    "raw_top1_accuracy": gt_raw_top1_matched_acc,
                    "best_frag_locked_accuracy": gt_best_frag_locked_acc,
                })

    # Save to CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nAll sweep results saved to: {output_path}")

    # Sort results using strict gate criteria
    sorted_df = df.sort_values(
        by=["false_lock_rate", "locked_accuracy", "locked_total_coverage", "assigned_accuracy"],
        ascending=[True, False, False, False]
    )

    print("\nSweep Rank Results (Top 15):")
    print(sorted_df.head(15).to_string(index=False))

    best = sorted_df.iloc[0]
    print(f"\n[BEST COMBINATION RESOLVED]")
    print(f"  - p1_threshold:         {best['p1_threshold']:.2f}")
    print(f"  - margin_threshold:     {best['margin_threshold']:.2f}")
    print(f"  - legibility_threshold: {best['legibility_threshold']:.2f}")
    print(f"  - False Lock Rate:      {best['false_lock_rate']*100:.1f}%")
    print(f"  - Locked Accuracy:      {best['locked_accuracy']*100:.1f}%")
    print(f"  - Locked Total Coverage:{best['locked_total_coverage']*100:.1f}%")
    print(f"  - Assigned Accuracy:    {best['assigned_accuracy']*100:.1f}%")
    print(f"  - Raw Top-1 Accuracy:   {best['raw_top1_accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
