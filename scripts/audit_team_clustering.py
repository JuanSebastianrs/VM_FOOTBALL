"""
Audit team clustering quality by comparing predicted team_id assignments
against ground truth from gameinfo.ini using IoU + Hungarian matching.

Reports team_purity (GATE: must be >= 90% before applying roster per team),
GK detection rate, referee detection rate, and coverage.

Usage:
    python scripts/audit_team_clustering.py \
        --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
        --team_json outputs/SNMOT-148/SNMOT-148_team_assignments.json \
        --detections_json outputs/SNMOT-148/SNMOT-148_detections.json \
        --output_json outputs/SNMOT-148/team_audit.json
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
    load_team_map,
)


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


def parse_gt_txt(gt_path):
    """Parse gt/gt.txt -> {frame: [{track_id, bbox}]}."""
    detections = defaultdict(list)
    df = pd.read_csv(
        gt_path, header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis", "_extra"],
    )
    df = df[df["conf"] > 0]
    for _, row in df.iterrows():
        detections[int(row["frame"])].append({
            "track_id": int(row["track_id"]),
            "bbox": [float(row["x"]), float(row["y"]),
                     float(row["x"] + row["w"]), float(row["y"] + row["h"])],
        })
    return detections


def load_pred_team_assignments(team_json):
    """Load predicted team assignments -> {track_id: team_id}."""
    with open(team_json) as f:
        data = json.load(f)
    return load_team_map(data)


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


def compute_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
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
    """Match pred and GT detections per frame using IoU + Hungarian."""
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
    parser = argparse.ArgumentParser(description="Audit team clustering quality")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--team_json", type=str, required=True)
    parser.add_argument("--detections_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    seq_dir = Path(args.sequence_dir)
    gi_path = seq_dir / "gameinfo.ini"
    gt_path = seq_dir / "gt" / "gt.txt"

    if not gi_path.exists():
        print(f"ERROR: gameinfo.ini not found at {gi_path}")
        return
    if not gt_path.exists():
        print(f"ERROR: gt.txt not found at {gt_path}")
        return

    # Load GT info
    gt_tracklet_info = parse_gameinfo(gi_path)
    gt_detections = parse_gt_txt(gt_path)

    # Load predictions
    pred_team_map = load_pred_team_assignments(args.team_json)
    pred_detections = load_pred_detections(args.detections_json)

    # Match per frame and accumulate votes
    # For each pred track, count how many times it matches each GT track
    pred_to_gt_votes = defaultdict(Counter)  # pred_tid -> Counter{gt_tid: count}
    total_frames_matched = 0

    all_frames = sorted(set(pred_detections.keys()) | set(gt_detections.keys()))
    for frame in all_frames:
        p_dets = pred_detections.get(frame, [])
        g_dets = gt_detections.get(frame, [])
        matches = match_per_frame(p_dets, g_dets, iou_threshold=args.iou_threshold)
        for pred_tid, gt_tid, iou in matches:
            pred_to_gt_votes[pred_tid][gt_tid] += 1
            total_frames_matched += 1

    # Resolve: each pred track -> its most-voted GT track
    pred_to_gt = {}
    for pred_tid, votes in pred_to_gt_votes.items():
        best_gt_tid = votes.most_common(1)[0][0]
        pred_to_gt[pred_tid] = best_gt_tid

    # Build confusion matrix: pred_team_id vs gt_team_side
    confusion = defaultdict(Counter)  # pred_team_id -> Counter{gt_team_side: count}
    matched_gt_tids = set()

    for pred_tid, gt_tid in pred_to_gt.items():
        pred_team = pred_team_map.get(pred_tid, -1)
        gt_info = gt_tracklet_info.get(gt_tid)
        if gt_info is None:
            continue
        gt_side = gt_info["team_side"]
        confusion[pred_team][gt_side] += 1
        matched_gt_tids.add(gt_tid)

    # Resolve mapping pred_team_id -> gt_team_side using Hungarian on 2x2 vote matrix
    pred_teams = [0, 1]
    gt_sides = ["left", "right"]
    vote_matrix = np.zeros((2, 2))
    for i, pt in enumerate(pred_teams):
        for j, gs in enumerate(gt_sides):
            vote_matrix[i, j] = confusion[pt][gs]

    # Hungarian on -votes (maximize match)
    row_ind, col_ind = linear_sum_assignment(-vote_matrix)
    team_mapping = {}
    for r, c in zip(row_ind, col_ind):
        team_mapping[pred_teams[r]] = gt_sides[c]

    print(f"\nTeam mapping: {team_mapping}")
    print(f"Vote matrix:\n  {gt_sides}")
    for i, pt in enumerate(pred_teams):
        print(f"  pred_{pt}: {[int(vote_matrix[i,j]) for j in range(2)]}")

    # Compute team_purity (all matched) and assigned_purity (only team 0/1)
    total_matched = 0
    correct_team = 0
    assigned_total = 0
    assigned_correct = 0
    unassigned_count = 0
    gk_total_gt = 0
    gk_detected = 0
    referee_total_gt = 0
    referee_detected = 0

    for pred_tid, gt_tid in pred_to_gt.items():
        gt_info = gt_tracklet_info.get(gt_tid)
        if gt_info is None:
            continue
        pred_team = pred_team_map.get(pred_tid, -1)
        gt_side = gt_info["team_side"]

        if gt_side in ("left", "right"):
            total_matched += 1
            mapped_side = team_mapping.get(pred_team)
            if mapped_side == gt_side:
                correct_team += 1
            # Assigned purity: only tracks with team 0 or 1
            if pred_team in (0, 1):
                assigned_total += 1
                if mapped_side == gt_side:
                    assigned_correct += 1
            else:
                unassigned_count += 1

    # GK and referee matching stats from GT
    # NOTE: This only checks if a GT GK/referee track was spatially matched (IoU)
    # to ANY pred track. It does NOT verify the predicted role/class is correct.
    for gt_tid, gt_info in gt_tracklet_info.items():
        if gt_info["role"] == "goalkeeper":
            gk_total_gt += 1
            if gt_tid in matched_gt_tids:
                gk_detected += 1
        elif gt_info["role"] == "referee":
            referee_total_gt += 1
            if gt_tid in matched_gt_tids:
                referee_detected += 1

    # Coverage: % of GT player/GK tracklets with a pred match
    gt_player_tids = {tid for tid, info in gt_tracklet_info.items()
                      if info["team_side"] in ("left", "right")}
    coverage = len(matched_gt_tids & gt_player_tids) / max(len(gt_player_tids), 1)

    team_purity = correct_team / max(total_matched, 1)
    assigned_purity = assigned_correct / max(assigned_total, 1)

    results = {
        "team_mapping": team_mapping,
        "team_purity": round(team_purity, 4),
        "assigned_purity": round(assigned_purity, 4),
        "total_matched_tracks": total_matched,
        "correct_team_tracks": correct_team,
        "assigned_total": assigned_total,
        "assigned_correct": assigned_correct,
        "unassigned_count": unassigned_count,
        "coverage": round(coverage, 4),
        "gk_total_gt": gk_total_gt,
        "gk_matched": gk_detected,
        "gk_matched_rate": round(gk_detected / max(gk_total_gt, 1), 4),
        "referee_total_gt": referee_total_gt,
        "referee_matched": referee_detected,
        "referee_matched_rate": round(referee_detected / max(referee_total_gt, 1), 4),
        "vote_matrix": {
            "pred_teams": pred_teams,
            "gt_sides": gt_sides,
            "votes": vote_matrix.tolist(),
        },
        "total_frames_with_matches": total_frames_matched,
        "confusion_raw": {str(k): dict(v) for k, v in confusion.items()},
    }

    print(f"\n{'='*50}")
    print(f"Team Clustering Audit: {seq_dir.name}")
    print(f"{'='*50}")
    print(f"  team_purity (all):    {team_purity:.1%}  (includes unassigned as wrong)")
    print(f"  assigned_purity:      {assigned_purity:.1%}  ({'PASS' if assigned_purity >= 0.90 else 'FAIL (< 90%)'})")
    print(f"  coverage:             {coverage:.1%}")
    print(f"  matched tracks:       {total_matched} (assigned: {assigned_total}, unassigned: {unassigned_count})")
    print(f"  correct team:         {correct_team}")
    print(f"  GK matched (IoU):     {gk_detected}/{gk_total_gt} (spatial match, not role classification)")
    print(f"  Referee matched:      {referee_detected}/{referee_total_gt} (spatial match only)")
    print(f"  Team mapping:         {team_mapping}")

    if assigned_purity < 0.90:
        print(f"\n  [CRITICAL] assigned_purity < 90%. Do NOT apply roster per team.")
        print(f"  Fix team clustering upstream before proceeding with jersey assignment.")
    elif team_purity < 0.90:
        print(f"\n  [WARNING] {unassigned_count} tracks have team_id=-1 (unassigned).")
        print(f"  Assigned tracks are {assigned_purity:.1%} pure. Roster can be applied")
        print(f"  to assigned tracks; unassigned tracks will be marked unknown.")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    main()
