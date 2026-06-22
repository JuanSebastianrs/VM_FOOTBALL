"""
End-to-end jersey evaluation using IoU+Hungarian matching between pipeline
outputs and SoccerNet GT.

Unlike dataset-native evaluation, pipeline track_ids do not match GT track_ids.
This script matches Pred↔GT bboxes per-frame using IoU+Hungarian, then
resolves each pred tracklet to its most-voted GT tracklet, and finally
compares predicted jersey numbers.

Usage:
    python scripts/evaluate_jersey_e2e.py \
        --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
        --detections_json outputs/SNMOT-148/SNMOT-148_detections.json \
        --jersey_json outputs/SNMOT-148/SNMOT-148_jersey_identity_v4.json \
        --team_mapping "0:right,1:left" \
        --output_json outputs/SNMOT-148/jersey_e2e_eval.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.identity.schema_utils import (
    extract_bbox_xyxy,
    iter_frame_entries,
)


def _safe_rate(num, den):
    if den <= 0:
        return 0.0
    value = num / den
    assert 0 <= num <= den, f"Numerator {num} must be between 0 and denominator {den}"
    assert 0.0 <= value <= 1.0, f"Rate out of bounds: {num} / {den} = {value}"
    return value


def parse_gameinfo(gameinfo_path):
    """Parse gameinfo.ini -> {track_id: {team_side, jersey_number, role}}."""
    tracklet_info = {}
    content = Path(gameinfo_path).read_text(encoding="utf-8", errors="replace")
    for line in content.split("\n"):
        line = line.strip()
        if not line.startswith("trackletID_") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        tid_str = key.strip().split("_")[1]
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        val = val.strip()
        if ";" not in val:
            continue
        team_desc, jersey_raw = val.rsplit(";", 1)
        team_desc = team_desc.strip().lower()

        if "referee" in team_desc or "ball" in team_desc:
            role = "referee" if "referee" in team_desc else "ball"
            tracklet_info[tid] = {"team_side": "other", "role": role, "jersey_number": -1}
            continue

        if "left" in team_desc:
            side = "left"
        elif "right" in team_desc:
            side = "right"
        else:
            continue

        is_gk = "goalkeeper" in team_desc
        role = "goalkeeper" if is_gk else "player"

        try:
            jersey = int(jersey_raw.strip())
        except ValueError:
            jersey = -1

        tracklet_info[tid] = {"team_side": side, "role": role, "jersey_number": jersey}
    return tracklet_info


def parse_gt_txt(gt_path, gt_tracklet_info=None):
    """Parse gt/gt.txt -> {frame: [{track_id, bbox}]}.

    If gt_tracklet_info is provided, only include tracks with numeric jersey
    numbers (>=1) and valid team sides (left/right) to avoid matching
    predicted players against referees or non-numeric GT tracks.
    """
    detections = defaultdict(list)
    df = pd.read_csv(
        gt_path, header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis", "_extra"],
    )
    df = df[df["conf"] > 0]
    for _, row in df.iterrows():
        tid = int(row["track_id"])
        if gt_tracklet_info is not None:
            info = gt_tracklet_info.get(tid)
            if info is None:
                continue
            if info.get("jersey_number", -1) < 1:
                continue
            if info.get("team_side") not in ("left", "right"):
                continue
        detections[int(row["frame"])].append({
            "track_id": tid,
            "bbox": [float(row["x"]), float(row["y"]),
                     float(row["x"] + row["w"]), float(row["y"] + row["h"])],
        })
    return detections


def load_pred_detections(detections_json):
    """Load predicted detections -> {frame: [{track_id, bbox}]}."""
    with open(detections_json) as f:
        det_data = json.load(f)
    detections = defaultdict(list)
    for entry in iter_frame_entries(det_data):
        fid = entry.get("frame_id")
        if fid is None:
            continue
        for p in entry.get("players", entry.get("player_detections", [])):
            tid = p.get("track_id")
            if tid is None:
                continue
            bbox = extract_bbox_xyxy(p)
            if bbox is None:
                continue
            detections[int(fid)].append({
                "track_id": int(tid),
                "bbox": bbox,
            })
    return detections


def load_jersey_predictions(jersey_json):
    """Load jersey predictions -> {track_id: {predicted_number, state, team_id}}."""
    with open(jersey_json) as f:
        data = json.load(f)
    lookup = {}
    for t in data.get("tracklets", []):
        tid = t.get("track_id")
        if tid is None:
            continue
        lookup[int(tid)] = {
            "predicted_number": t.get("predicted_number"),
            "state": t.get("state", "unknown"),
            "team_id": t.get("team_id", -1),
            "confidence": t.get("confidence", 0.0),
            "alternatives": t.get("alternatives", []),
        }
    return lookup


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def match_per_frame(pred_dets, gt_dets, iou_threshold=0.5):
    if not pred_dets or not gt_dets:
        return []
    n_pred = len(pred_dets)
    n_gt = len(gt_dets)
    cost = np.zeros((n_pred, n_gt))
    for i, pd in enumerate(pred_dets):
        for j, gd in enumerate(gt_dets):
            iou = compute_iou(pd["bbox"], gd["bbox"])
            cost[i, j] = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind, col_ind):
        iou = 1.0 - cost[i, j]
        if iou >= iou_threshold:
            matches.append((pred_dets[i]["track_id"], gt_dets[j]["track_id"], iou))
    return matches


def main():
    parser = argparse.ArgumentParser(description="End-to-end jersey evaluation (IoU matched)")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--detections_json", type=str, required=True)
    parser.add_argument("--jersey_json", type=str, required=True)
    parser.add_argument("--team_mapping", type=str, default=None,
                        help="Pred team_id -> side, e.g. '0:right,1:left' or path to team_audit.json")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    seq_dir = Path(args.sequence_dir)
    gi_path = seq_dir / "gameinfo.ini"
    gt_path = seq_dir / "gt" / "gt.txt"

    if not gi_path.exists():
        raise FileNotFoundError(f"gameinfo.ini not found: {gi_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    # Load GT
    gt_tracklet_info = parse_gameinfo(gi_path)
    gt_detections = parse_gt_txt(gt_path, gt_tracklet_info=gt_tracklet_info)

    # Load predictions
    pred_detections = load_pred_detections(args.detections_json)
    jersey_lookup = load_jersey_predictions(args.jersey_json)

    # Resolve team mapping
    team_mapping = {}
    if args.team_mapping:
        mapping_path = Path(args.team_mapping)
        if mapping_path.exists() and mapping_path.suffix == ".json":
            with open(mapping_path) as f:
                audit = json.load(f)
            team_mapping = {int(k): v for k, v in audit.get("team_mapping", {}).items()}
        else:
            for pair in args.team_mapping.split(","):
                k, v = pair.strip().split(":")
                team_mapping[int(k)] = v.strip()
    print(f"Team mapping: {team_mapping}")

    # Match per frame and accumulate votes
    pred_to_gt_votes = defaultdict(Counter)
    total_frames_matched = 0
    per_frame_matches = []  # frame-by-frame for visual QA correctness

    all_frames = sorted(set(pred_detections.keys()) | set(gt_detections.keys()))
    for frame in all_frames:
        p_dets = pred_detections.get(frame, [])
        g_dets = gt_detections.get(frame, [])
        matches = match_per_frame(p_dets, g_dets, iou_threshold=args.iou_threshold)
        for pred_tid, gt_tid, iou in matches:
            pred_to_gt_votes[pred_tid][gt_tid] += 1
            total_frames_matched += 1
            gt_info = gt_tracklet_info.get(gt_tid)
            if gt_info and gt_info.get("jersey_number", -1) >= 1:
                per_frame_matches.append({
                    "frame_id": int(frame),
                    "pred_track_id": int(pred_tid),
                    "gt_track_id": int(gt_tid),
                    "gt_jersey": int(gt_info["jersey_number"]),
                    "iou": float(iou),
                })

    # Resolve each pred track -> best GT track
    pred_to_gt = {}
    for pred_tid, votes in pred_to_gt_votes.items():
        if not votes:
            continue
        best_gt_tid = votes.most_common(1)[0][0]
        pred_to_gt[pred_tid] = best_gt_tid

    # Build fragment-level results
    fragment_results = []
    for pred_tid, gt_tid in pred_to_gt.items():
        gt_info = gt_tracklet_info.get(gt_tid)
        if gt_info is None:
            continue
        gt_jersey = gt_info.get("jersey_number", -1)
        if gt_jersey < 1:
            continue

        pred_jersey_data = jersey_lookup.get(pred_tid, {})
        pred_num = pred_jersey_data.get("predicted_number")
        state = pred_jersey_data.get("state", "unknown")
        conf = pred_jersey_data.get("confidence", 0.0)
        alts = pred_jersey_data.get("alternatives", [])

        raw_top1 = alts[0][0] if isinstance(alts, list) and len(alts) > 0 and isinstance(alts[0], (list, tuple)) else None
        if raw_top1 is None and isinstance(alts, list) and len(alts) > 0:
            raw_top1 = alts[0]

        fragment_results.append({
            "pred_track_id": pred_tid,
            "gt_track_id": gt_tid,
            "gt_jersey": gt_jersey,
            "gt_side": gt_info.get("team_side"),
            "gt_role": gt_info.get("role"),
            "pred_team_id": pred_jersey_data.get("team_id"),
            "predicted_number": pred_num,
            "raw_top1": raw_top1,
            "state": state,
            "confidence": conf,
            "match_frames": int(pred_to_gt_votes[pred_tid][gt_tid]),
        })

    # Aggregate per-GT tracklet (handle fragmentation)
    # For each GT, pick the representative pred fragment with the most matched frames
    gt_to_fragments = defaultdict(list)
    for fr in fragment_results:
        gt_to_fragments[fr["gt_track_id"]].append(fr)

    gt_results = []
    raw_top1_correct_gt = 0
    assigned_correct_gt = 0
    locked_total_gt = 0
    locked_correct_gt = 0
    false_locks_gt = 0
    best_jersey_correct_gt = 0
    best_jersey_assigned_correct_gt = 0
    best_frag_locked_total = 0
    best_frag_locked_correct = 0
    team_side_ok = 0
    team_side_total = 0

    for gt_tid, frags in gt_to_fragments.items():
        # Representative for primary metrics: fragment with most matched frames
        rep = max(frags, key=lambda x: x["match_frames"])
        gt_jersey = rep["gt_jersey"]
        pred_num = rep["predicted_number"]
        raw_top1 = rep["raw_top1"]
        state = rep["state"]
        pred_team_id = rep.get("pred_team_id", -1)

        is_raw_correct = (raw_top1 is not None and raw_top1 == gt_jersey)
        is_assigned_correct = (pred_num is not None and pred_num == gt_jersey)

        # Team side consistency
        gt_side = rep.get("gt_side")
        mapped_side = team_mapping.get(pred_team_id)
        if mapped_side is not None and gt_side is not None:
            team_side_total += 1
            if mapped_side == gt_side:
                team_side_ok += 1

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

        # Best-jersey fragment metric: fragment with highest raw/assigned correctness confidence
        # Pick the fragment whose assigned prediction is correct (if any)
        best_frag = rep
        for frag in frags:
            f_pred = frag["predicted_number"]
            f_raw = frag["raw_top1"]
            if f_pred is not None and f_pred == gt_jersey:
                # Prefer correct assigned
                if best_frag["predicted_number"] != gt_jersey or frag.get("confidence", 0) > best_frag.get("confidence", 0):
                    best_frag = frag
            elif f_raw is not None and f_raw == gt_jersey and best_frag["predicted_number"] != gt_jersey:
                # Fallback to raw correct
                if best_frag["raw_top1"] != gt_jersey or frag.get("confidence", 0) > best_frag.get("confidence", 0):
                    best_frag = frag

        if best_frag["raw_top1"] is not None and best_frag["raw_top1"] == gt_jersey:
            best_jersey_correct_gt += 1
        if best_frag["predicted_number"] is not None and best_frag["predicted_number"] == gt_jersey:
            best_jersey_assigned_correct_gt += 1
        if best_frag["state"] == "locked":
            best_frag_locked_total += 1
            if best_frag["predicted_number"] is not None and best_frag["predicted_number"] == gt_jersey:
                best_frag_locked_correct += 1

        gt_results.append({
            "gt_track_id": gt_tid,
            "gt_jersey": gt_jersey,
            "representative_pred_track_id": rep["pred_track_id"],
            "num_fragments": len(frags),
            "predicted_number": pred_num,
            "raw_top1": raw_top1,
            "state": state,
            "confidence": rep["confidence"],
            "raw_correct": is_raw_correct,
            "assigned_correct": is_assigned_correct,
            "team_side_consistent": (mapped_side == gt_side) if (mapped_side is not None and gt_side is not None) else None,
            "best_frag_pred_track_id": best_frag["pred_track_id"],
            "best_frag_raw_correct": (best_frag["raw_top1"] == gt_jersey) if best_frag["raw_top1"] is not None else False,
            "best_frag_assigned_correct": (best_frag["predicted_number"] == gt_jersey) if best_frag["predicted_number"] is not None else False,
        })



    # Also compute fragment-level metrics for reference
    total_fragments = len(fragment_results)
    n_frag = max(total_fragments, 1)
    raw_top1_correct_frag = sum(1 for fr in fragment_results if fr["raw_top1"] is not None and fr["raw_top1"] == fr["gt_jersey"])
    assigned_correct_frag = sum(1 for fr in fragment_results if fr["predicted_number"] is not None and fr["predicted_number"] == fr["gt_jersey"])
    locked_total_frag = sum(1 for fr in fragment_results if fr["state"] == "locked")
    locked_correct_frag = sum(1 for fr in fragment_results if fr["state"] == "locked" and fr["predicted_number"] is not None and fr["predicted_number"] == fr["gt_jersey"])
    false_locks_frag = sum(1 for fr in fragment_results if fr["state"] == "locked" and fr["predicted_number"] is not None and fr["predicted_number"] != fr["gt_jersey"])

    # GT denominator
    numeric_gt_total = len([k for k, v in gt_tracklet_info.items() if v.get("jersey_number", -1) >= 1 and v.get("team_side") in ("left", "right")])
    unique_gt_matched = len(gt_to_fragments)
    gt_coverage = _safe_rate(unique_gt_matched, max(numeric_gt_total, 1))
    n_gt = max(unique_gt_matched, 1)
    n_gt_total = max(numeric_gt_total, 1)

    frag_raw_top1_acc = _safe_rate(raw_top1_correct_frag, n_frag)
    frag_assigned_acc = _safe_rate(assigned_correct_frag, n_frag)
    frag_locked_accuracy = _safe_rate(locked_correct_frag, locked_total_frag) if locked_total_frag > 0 else 0.0
    frag_locked_coverage = _safe_rate(locked_total_frag, n_frag)
    frag_false_lock_rate = _safe_rate(false_locks_frag, locked_total_frag) if locked_total_frag > 0 else 0.0

    gt_raw_top1_matched_acc = _safe_rate(raw_top1_correct_gt, n_gt)
    gt_assigned_matched_acc = _safe_rate(assigned_correct_gt, n_gt)
    gt_locked_accuracy = _safe_rate(locked_correct_gt, locked_total_gt) if locked_total_gt > 0 else 0.0
    gt_locked_coverage = _safe_rate(locked_total_gt, n_gt)
    gt_false_lock_rate = _safe_rate(false_locks_gt, locked_total_gt) if locked_total_gt > 0 else 0.0

    gt_raw_top1_total_acc = _safe_rate(raw_top1_correct_gt, n_gt_total)
    gt_assigned_total_acc = _safe_rate(assigned_correct_gt, n_gt_total)
    gt_locked_total_coverage = _safe_rate(locked_total_gt, n_gt_total)

    gt_best_frag_raw_top1_acc = _safe_rate(best_jersey_correct_gt, n_gt)
    gt_best_frag_assigned_acc = _safe_rate(best_jersey_assigned_correct_gt, n_gt)
    gt_best_frag_locked_acc = _safe_rate(best_frag_locked_correct, best_frag_locked_total) if best_frag_locked_total > 0 else 0.0
    team_side_consistency = _safe_rate(team_side_ok, team_side_total) if team_side_total > 0 else None

    summary = {
        "sequence": seq_dir.name,
        "team_mapping": team_mapping,
        "iou_threshold": args.iou_threshold,
        "total_frames_matched": total_frames_matched,
        "total_matched_pairs": total_fragments,
        "unique_gt_matched": unique_gt_matched,
        "gt_total": numeric_gt_total,
        "gt_coverage": round(gt_coverage, 4),
        # Fragment-level metrics
        "frag_raw_top1_acc": round(frag_raw_top1_acc, 4),
        "frag_assigned_acc": round(frag_assigned_acc, 4),
        "frag_locked_total": locked_total_frag,
        "frag_locked_accuracy": round(frag_locked_accuracy, 4),
        "frag_locked_coverage": round(frag_locked_coverage, 4),
        "frag_false_lock_rate": round(frag_false_lock_rate, 4),
        # GT-level metrics over matched GTs only
        "gt_raw_top1_matched_acc": round(gt_raw_top1_matched_acc, 4),
        "gt_assigned_matched_acc": round(gt_assigned_matched_acc, 4),
        "gt_locked_total": locked_total_gt,
        "gt_locked_accuracy": round(gt_locked_accuracy, 4),
        "gt_locked_coverage": round(gt_locked_coverage, 4),
        "gt_false_lock_rate": round(gt_false_lock_rate, 4),
        # GT-level metrics over ALL numeric GTs (includes unmatched as wrong)
        "gt_raw_top1_total_acc": round(gt_raw_top1_total_acc, 4),
        "gt_assigned_total_acc": round(gt_assigned_total_acc, 4),
        "gt_locked_total_coverage": round(gt_locked_total_coverage, 4),
        "gt_best_frag_raw_top1_acc": round(gt_best_frag_raw_top1_acc, 4),
        "gt_best_frag_assigned_acc": round(gt_best_frag_assigned_acc, 4),
        "gt_best_frag_locked_total": best_frag_locked_total,
        "gt_best_frag_locked_correct": best_frag_locked_correct,
        "gt_best_frag_locked_acc": round(gt_best_frag_locked_acc, 4),
        "team_side_consistency": round(team_side_consistency, 4) if team_side_consistency is not None else None,
        "team_side_total": team_side_total,
        "team_side_ok": team_side_ok,
    }

    # Verify all metrics in summary are in range [0.0, 1.0]
    for key, value in summary.items():
        if value is not None and key.endswith(("_acc", "_accuracy", "_rate", "_coverage", "_consistency")):
            assert 0.0 <= value <= 1.0, f"Summary metric {key} is out of bounds: {value}"

    print(f"\n=== End-to-End Jersey Evaluation ({seq_dir.name}) ===")
    print(f"  GT tracklets:        {numeric_gt_total}")
    print(f"  Unique GT matched:   {unique_gt_matched} (coverage {gt_coverage:.1%})")
    print(f"  Total fragments:     {total_fragments}")
    print(f"  Frames matched:      {total_frames_matched}")
    print()
    print(f"  [GT-matched] Raw top-1:  {summary['gt_raw_top1_matched_acc']:.1%} ({raw_top1_correct_gt}/{unique_gt_matched})")
    print(f"  [GT-matched] Assigned:   {summary['gt_assigned_matched_acc']:.1%} ({assigned_correct_gt}/{unique_gt_matched})")
    print(f"  [GT-matched] Locked:     {locked_total_gt} ({summary['gt_locked_coverage']:.1%})")
    print(f"  [GT-matched] Lock acc:   {summary['gt_locked_accuracy']:.1%}")
    print(f"  [GT-matched] False lock: {summary['gt_false_lock_rate']:.1%}")
    print(f"  [GT-best-frag] Raw top-1: {summary['gt_best_frag_raw_top1_acc']:.1%} ({best_jersey_correct_gt}/{unique_gt_matched})")
    print(f"  [GT-best-frag] Assigned:  {summary['gt_best_frag_assigned_acc']:.1%} ({best_jersey_assigned_correct_gt}/{unique_gt_matched})")
    print(f"  [GT-best-frag] Lock acc:  {summary['gt_best_frag_locked_acc']:.1%}")
    if summary.get("team_side_consistency") is not None:
        print(f"  Team side consistency:   {summary['team_side_consistency']:.1%} ({team_side_ok}/{team_side_total})")
    print()
    print(f"  [GT-total]   Raw top-1:  {summary['gt_raw_top1_total_acc']:.1%} ({raw_top1_correct_gt}/{numeric_gt_total})")
    print(f"  [GT-total]   Assigned:   {summary['gt_assigned_total_acc']:.1%} ({assigned_correct_gt}/{numeric_gt_total})")
    print(f"  [GT-total]   Locked cov: {summary['gt_locked_total_coverage']:.1%} ({locked_total_gt}/{numeric_gt_total})")
    print()
    print(f"  [Fragment]   Raw top-1:  {summary['frag_raw_top1_acc']:.1%} ({raw_top1_correct_frag}/{total_fragments})")
    print(f"  [Fragment]   Assigned:   {summary['frag_assigned_acc']:.1%} ({assigned_correct_frag}/{total_fragments})")

    out = {"summary": summary, "fragment_results": fragment_results, "gt_results": gt_results, "per_frame_matches": per_frame_matches}
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
