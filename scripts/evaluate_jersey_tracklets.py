"""
Tracklet-level jersey number evaluation.
Computes separated metrics: raw model accuracy vs assigned accuracy,
coverage, false-lock rate, per-team breakdown.

Key improvement over v1: uses GT denominator (not just intersection),
reports raw_top1 independently of Hungarian/locking.

Usage:
    python scripts/evaluate_jersey_tracklets.py \
        --pred_json outputs/jersey_eval/val_predictions.json \
        --gt_json datasets/jersey_tracking_v1/tracklets.json \
        --output_dir outputs/jersey_eval/val_results \
        --sequence SNMOT-148
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_predictions(pred_json):
    """Load prediction JSON -> dict keyed by (sequence, track_id)."""
    with open(pred_json) as f:
        data = json.load(f)
    tracklets = data.get("tracklets", [])
    records = {}
    for t in tracklets:
        key = (t.get("sequence"), int(t.get("track_id", -1)))
        records[key] = {
            "predicted_number": t.get("predicted_number"),
            "confidence": t.get("confidence", 0.0),
            "state": t.get("state", "unknown"),
            "team_id": t.get("team_id", -1),
            "alternatives": t.get("alternatives", []),
            "raw_top1": t.get("raw_top1"),  # from new assignment
        }
    return records


def load_ground_truth(gt_json, sequence=None, allowed_sequences=None):
    """Load GT tracklets -> dict keyed by (sequence, track_id).

    Args:
        gt_json: Path to tracklets.json.
        sequence: Optional single sequence name to filter.
        allowed_sequences: Optional set of sequence names to include.
    """
    with open(gt_json) as f:
        records = json.load(f)
    if sequence:
        records = [r for r in records if r.get("sequence") == sequence]
    elif allowed_sequences:
        records = [r for r in records if r.get("sequence") in allowed_sequences]
    gt = {}
    for r in records:
        jersey = r.get("jersey_number", -1)
        try:
            jersey = int(jersey)
        except (ValueError, TypeError):
            jersey = -1
        if jersey < 1:
            continue  # skip non-numeric GT
        key = (r.get("sequence"), int(r.get("track_id", -1)))
        gt[key] = {
            "jersey_number": jersey,
            "team_id": int(r.get("team_id", -1)),
            "sequence": r.get("sequence"),
        }
    return gt


def compute_tracklet_accuracy(pred_records, gt_records):
    """
    Compute metrics using GT as the denominator.

    Metrics reported:
    - raw_top1_accuracy: model's top-1 prediction vs GT (no roster/lock)
    - raw_top3_accuracy: GT in model's top-3
    - assigned_accuracy: final assigned number (after lock/roster) vs GT
    - coverage: % of GT tracklets that have a prediction
    - locked_accuracy: accuracy only among locked tracks
    - false_lock_rate: % of locked tracks that are wrong
    """
    # GT-based denominator
    gt_keys = set(gt_records.keys())
    pred_keys = set(pred_records.keys())
    matched_keys = gt_keys & pred_keys
    missing_keys = gt_keys - pred_keys  # GT without prediction
    extra_keys = pred_keys - gt_keys    # predictions without GT

    gt_total = len(gt_keys)
    coverage = len(matched_keys) / max(gt_total, 1)

    # Counters
    raw_top1_correct = 0
    raw_top3_correct = 0
    assigned_correct = 0
    locked_total = 0
    locked_correct = 0
    false_locks = 0
    by_team_raw = defaultdict(lambda: {"total": 0, "correct": 0})
    by_team_assigned = defaultdict(lambda: {"total": 0, "correct": 0})
    errors = []

    for key in matched_keys:
        pred = pred_records[key]
        gt = gt_records[key]
        gt_num = gt["jersey_number"]
        p_num = pred.get("predicted_number")
        state = pred.get("state", "unknown")
        conf = pred.get("confidence", 0.0)

        # Raw model accuracy from alternatives
        alts = pred.get("alternatives", [])
        if isinstance(alts, list) and len(alts) > 0:
            raw_top1 = alts[0][0] if isinstance(alts[0], (tuple, list)) else alts[0]
            raw_top3 = [a[0] if isinstance(a, (tuple, list)) else a for a in alts[:3]]
        else:
            raw_top1 = pred.get("raw_top1")
            raw_top3 = [raw_top1] if raw_top1 else []

        is_raw_top1 = (raw_top1 is not None and raw_top1 == gt_num)
        is_raw_top3 = (gt_num in raw_top3) if raw_top3 else False

        if is_raw_top1:
            raw_top1_correct += 1
        if is_raw_top1 or is_raw_top3:
            raw_top3_correct += 1

        # Assigned accuracy (after lock/roster)
        is_assigned_correct = (p_num is not None and p_num == gt_num)
        if is_assigned_correct:
            assigned_correct += 1

        # Lock stats
        if state == "locked":
            locked_total += 1
            if is_assigned_correct:
                locked_correct += 1
            elif p_num is not None:
                false_locks += 1

        # Per-team stats
        team_key = f"team_{gt['team_id']}"
        by_team_raw[team_key]["total"] += 1
        if is_raw_top1:
            by_team_raw[team_key]["correct"] += 1
        by_team_assigned[team_key]["total"] += 1
        if is_assigned_correct:
            by_team_assigned[team_key]["correct"] += 1

        errors.append({
            "key": list(key),
            "gt": gt_num,
            "raw_top1": raw_top1,
            "assigned": p_num,
            "state": state,
            "confidence": conf,
            "raw_correct": is_raw_top1,
            "assigned_correct": is_assigned_correct,
        })

    # Add missing GT keys as errors
    for key in missing_keys:
        gt = gt_records[key]
        errors.append({
            "key": list(key),
            "gt": gt["jersey_number"],
            "raw_top1": None,
            "assigned": None,
            "state": "missing",
            "confidence": 0.0,
            "raw_correct": False,
            "assigned_correct": False,
        })

    n_matched = max(len(matched_keys), 1)
    n_gt = max(gt_total, 1)

    return {
        "gt_total": gt_total,
        "pred_total": len(pred_keys),
        "matched": len(matched_keys),
        "missing_from_pred": len(missing_keys),
        "extra_in_pred": len(extra_keys),
        "coverage": round(coverage, 4),
        # Raw model metrics — matched denominator (accuracy on seen tracklets)
        "raw_top1_matched_acc": round(raw_top1_correct / n_matched, 4),
        "raw_top3_matched_acc": round(raw_top3_correct / n_matched, 4),
        # Raw model metrics — GT denominator (penalizes missing predictions)
        "raw_top1_gt_acc": round(raw_top1_correct / n_gt, 4),
        "raw_top3_gt_acc": round(raw_top3_correct / n_gt, 4),
        "raw_top1_correct": raw_top1_correct,
        "raw_top3_correct": raw_top3_correct,
        # Assigned metrics — both denominators
        "assigned_matched_acc": round(assigned_correct / n_matched, 4),
        "assigned_gt_acc": round(assigned_correct / n_gt, 4),
        "assigned_correct": assigned_correct,
        # Lock metrics
        "locked_total": locked_total,
        "locked_accuracy": round(locked_correct / max(locked_total, 1), 4) if locked_total > 0 else 0.0,
        "locked_coverage": round(locked_total / n_matched, 4),
        "false_lock_rate": round(false_locks / max(locked_total, 1), 4) if locked_total > 0 else 0.0,
        "false_locks": false_locks,
        # Per-team
        "raw_by_team": {k: {"accuracy": round(v["correct"]/max(v["total"],1), 4), **v}
                        for k, v in by_team_raw.items()},
        "assigned_by_team": {k: {"accuracy": round(v["correct"]/max(v["total"],1), 4), **v}
                             for k, v in by_team_assigned.items()},
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Tracklet-level jersey evaluation")
    parser.add_argument("--pred_json", type=str, required=True, help="Predictions JSON")
    parser.add_argument("--gt_json", type=str, required=True, help="Ground-truth tracklets JSON")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence", type=str, default=None, help="Filter to single sequence")
    parser.add_argument("--splits_json", type=str, default=None,
                        help="splits.json to filter GT by split")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"],
                        help="Split name (requires --splits_json)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_records = load_predictions(args.pred_json)

    # Determine which GT sequences to include
    allowed_sequences = None
    if args.splits_json and args.split:
        with open(args.splits_json) as f:
            splits = json.load(f)
        allowed_sequences = {s for s, v in splits.get(args.split, {}).items() if v}
        print(f"  Filtering GT to {args.split} split: {len(allowed_sequences)} sequences")
    else:
        # Auto-filter: restrict GT to sequences that appear in predictions
        pred_sequences = {key[0] for key in pred_records.keys() if key[0]}
        if pred_sequences:
            allowed_sequences = pred_sequences
            print(f"  Auto-filtering GT to {len(allowed_sequences)} sequences from predictions")

    gt_records = load_ground_truth(args.gt_json, sequence=args.sequence,
                                   allowed_sequences=allowed_sequences)

    metrics = compute_tracklet_accuracy(pred_records, gt_records)

    summary = {k: v for k, v in metrics.items() if k != "errors"}
    print("\n=== Jersey Identification Evaluation ===")
    print(f"  GT tracklets:        {summary['gt_total']}")
    print(f"  Pred tracklets:      {summary['pred_total']}")
    print(f"  Matched:             {summary['matched']}")
    print(f"  Coverage:            {summary['coverage']:.1%}")
    print()
    print(f"  Raw top-1 (matched): {summary['raw_top1_matched_acc']:.1%} ({summary['raw_top1_correct']}/{summary['matched']})")
    print(f"  Raw top-1 (GT):      {summary['raw_top1_gt_acc']:.1%} ({summary['raw_top1_correct']}/{summary['gt_total']})")
    print(f"  Raw top-3 (matched): {summary['raw_top3_matched_acc']:.1%} ({summary['raw_top3_correct']}/{summary['matched']})")
    print()
    print(f"  Assigned (matched):  {summary['assigned_matched_acc']:.1%} ({summary['assigned_correct']}/{summary['matched']})")
    print(f"  Assigned (GT):       {summary['assigned_gt_acc']:.1%} ({summary['assigned_correct']}/{summary['gt_total']})")
    print(f"  Locked tracks:       {summary['locked_total']} ({summary['locked_coverage']:.1%} coverage)")
    print(f"  Locked accuracy:     {summary['locked_accuracy']:.1%}")
    print(f"  False lock rate:     {summary['false_lock_rate']:.1%} ({summary['false_locks']})")

    if summary.get("raw_by_team"):
        print("\n  Per-team raw top-1:")
        for team, stats in sorted(summary["raw_by_team"].items()):
            print(f"    {team}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    summary_path = output_dir / "jersey_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    errors_path = output_dir / "jersey_evaluation_errors.json"
    with open(errors_path, "w") as f:
        json.dump(metrics["errors"], f, indent=2)
    print(f"Errors: {errors_path}")


if __name__ == "__main__":
    main()