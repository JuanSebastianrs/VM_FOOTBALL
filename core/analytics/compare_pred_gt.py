"""
Compare predicted tracking_2d.csv against GT tracking_2d.csv.

Outputs:
  - frame_matches.csv: per-frame match details
  - matches_detailed.csv: per-match pair info (for team mapping)
  - comparison_summary.json: aggregated metrics

Matching: per-frame Hungarian algorithm on projected 2D distance (metres).
"""

import os
import sys
import csv
import json
import argparse
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def parse_visible_players(rows, source_label):
    frames = defaultdict(list)
    for r in rows:
        if r.get("entity_type") != "player":
            continue
        visible = int(r.get("visible", 0))
        if not visible:
            continue
        team_id = int(r.get("team_id", -1))
        if team_id < 0:
            continue
        x = float(r["x_m"]) if r.get("x_m") != '' else None
        y = float(r["y_m"]) if r.get("y_m") != '' else None
        if x is None or y is None:
            continue
        fid = int(r["frame_id"])
        frames[fid].append({
            "track_id": int(r["track_id"]),
            "team_id": team_id,
            "x": x,
            "y": y,
            "source": source_label,
        })
    return frames


def compute_matches(pred_frames, gt_frames, max_dist_m=3.0):
    all_frames = sorted(set(pred_frames.keys()) | set(gt_frames.keys()))
    frame_stats = []
    total_gt = 0
    total_pred = 0
    total_matched = 0
    all_distances = []
    detailed = []

    for fid in all_frames:
        preds = pred_frames.get(fid, [])
        gts = gt_frames.get(fid, [])

        n_p = len(preds)
        n_g = len(gts)
        total_gt += n_g
        total_pred += n_p

        if n_p == 0 or n_g == 0:
            frame_stats.append({
                "frame_id": fid,
                "n_pred": n_p,
                "n_gt": n_g,
                "n_matched": 0,
                "mean_error_m": None,
                "max_error_m": None,
            })
            continue

        cost = np.zeros((n_p, n_g))
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                cost[i, j] = math.hypot(p["x"] - g["x"], p["y"] - g["y"])

        row_ind, col_ind = linear_sum_assignment(cost)

        frame_dists = []
        matched = 0
        for i, j in zip(row_ind, col_ind):
            d = cost[i, j]
            if d <= max_dist_m:
                matched += 1
                frame_dists.append(d)
                all_distances.append(d)
                detailed.append({
                    "frame_id": fid,
                    "pred_track_id": preds[i]["track_id"],
                    "gt_track_id": gts[j]["track_id"],
                    "pred_team_id": preds[i]["team_id"],
                    "gt_team_id": gts[j]["team_id"],
                    "error_m": round(d, 3),
                })

        total_matched += matched
        frame_stats.append({
            "frame_id": fid,
            "n_pred": n_p,
            "n_gt": n_g,
            "n_matched": matched,
            "mean_error_m": round(float(np.mean(frame_dists)), 3) if frame_dists else None,
            "max_error_m": round(float(np.max(frame_dists)), 3) if frame_dists else None,
        })

    summary = {
        "total_frames": len(all_frames),
        "total_gt_players": total_gt,
        "total_pred_players": total_pred,
        "total_matched": total_matched,
        "coverage": round(total_matched / total_gt, 3) if total_gt > 0 else 0.0,
        "precision": round(total_matched / total_pred, 3) if total_pred > 0 else 0.0,
        "match_threshold_m": max_dist_m,
        "mean_error_m": round(float(np.mean(all_distances)), 3) if all_distances else None,
        "overall_rmse_m": round(float(np.sqrt(np.mean([d**2 for d in all_distances]))), 3) if all_distances else None,
        "max_error_m": round(float(np.max(all_distances)), 3) if all_distances else None,
        "unmatched_gt": total_gt - total_matched,
        "unmatched_pred": total_pred - total_matched,
    }

    # Infer team mapping from matched pairs (majority vote per pred_team)
    team_votes = {}
    for m in detailed:
        key = (m["pred_team_id"], m["gt_team_id"])
        team_votes[key] = team_votes.get(key, 0) + 1
    team_mapping = {}
    for (pred_team, gt_team), count in team_votes.items():
        if pred_team not in team_mapping or team_mapping[pred_team][1] < count:
            team_mapping[pred_team] = (gt_team, count)
    summary["team_mapping_pred_to_gt"] = {k: v[0] for k, v in team_mapping.items()}

    return frame_stats, detailed, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--gt_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_dist_m", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading prediction CSV...")
    pred_rows = load_csv(args.pred_csv)
    pred_frames = parse_visible_players(pred_rows, "pred")

    print("Loading GT CSV...")
    gt_rows = load_csv(args.gt_csv)
    gt_frames = parse_visible_players(gt_rows, "gt")

    print("Running per-frame Hungarian matching...")
    frame_stats, detailed, summary = compute_matches(pred_frames, gt_frames, args.max_dist_m)

    stats_path = os.path.join(args.output_dir, "frame_matches.csv")
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "n_pred", "n_gt", "n_matched", "mean_error_m", "max_error_m"
        ])
        writer.writeheader()
        for m in frame_stats:
            writer.writerow(m)

    detailed_path = os.path.join(args.output_dir, "matches_detailed.csv")
    with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "pred_track_id", "gt_track_id", "pred_team_id", "gt_team_id", "error_m"
        ])
        writer.writeheader()
        for m in detailed:
            writer.writerow(m)

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nFrame stats      -> {stats_path}")
    print(f"Detailed matches -> {detailed_path}")
    print(f"Summary          -> {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
