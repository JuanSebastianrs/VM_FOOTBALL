"""
TacticalVision AI — Compute Tactical Metrics from 2D Tracking CSV.

Reads a tracking_2d.csv and exports:
  - player_physical_metrics.csv
  - player_physical_timeseries.csv
  - team_shape_metrics.csv

Usage:
    python core/analytics/compute_metrics.py \
        --input_csv outputs/SNMOT-148/SNMOT-148_tracking_2d.csv \
        --output_dir outputs/SNMOT-148 \
        --fps 25.0
"""

import os
import argparse
import csv
import math
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull


# ── Configurable thresholds ──
WALK_KMH = 7.0
JOG_KMH = 15.0
RUN_KMH = 20.0
SPRINT_KMH = 25.2
MAX_SPEED_KMH = 40.0          # cap unrealistic projection spikes
OUTLIER_DIST_M = 3.0          # >3 m in one frame @25fps ≈ 270 km/h → outlier
MIN_SPRINT_FRAMES = 5         # minimum consecutive frames to count a sprint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


def load_tracking_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def parse_numeric(value):
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compute_physical_metrics(rows, fps):
    dt = 1.0 / fps

    player_frames = defaultdict(list)
    for r in rows:
        if r.get("entity_type") != "player":
            continue
        tid = int(r["track_id"])
        fid = int(r["frame_id"])
        x = parse_numeric(r.get("x_m"))
        y = parse_numeric(r.get("y_m"))
        visible = int(r.get("visible", 0))
        team_id = int(r.get("team_id", -1))
        if x is None or y is None or not visible or team_id < 0:
            continue
        player_frames[tid].append({
            "frame_id": fid,
            "x": x,
            "y": y,
            "team_id": team_id,
            "role": r.get("role", "unknown"),
        })

    for tid in player_frames:
        player_frames[tid].sort(key=lambda d: d["frame_id"])

    timeseries = []
    summary = {}

    for tid, frames in player_frames.items():
        total_dist = 0.0
        speeds = []
        accels = []
        sprint_count = 0
        sprint_dist = 0.0
        high_speed_dist = 0.0
        sprint_frames = 0
        sprint_dist_candidate = 0.0

        for i in range(len(frames)):
            f = frames[i]
            x, y = f["x"], f["y"]
            speed = 0.0
            accel = 0.0
            dist = 0.0
            outlier = False

            if i > 0:
                prev = frames[i - 1]
                gap = f["frame_id"] - prev["frame_id"]
                if gap == 1:
                    dx = x - prev["x"]
                    dy = y - prev["y"]
                    dist = math.hypot(dx, dy)
                    if dist > OUTLIER_DIST_M:
                        dist = 0.0
                        outlier = True
                    else:
                        total_dist += dist
                        speed = (dist / dt) * 3.6
                        speed = min(speed, MAX_SPEED_KMH)
                        if len(speeds) > 0:
                            prev_speed = speeds[-1]
                            accel = ((speed / 3.6) - (prev_speed / 3.6)) / dt

            speeds.append(speed)
            accels.append(accel)

            if speed >= SPRINT_KMH and not outlier:
                sprint_frames += 1
                sprint_dist_candidate += dist
                high_speed_dist += dist
            elif speed >= RUN_KMH and not outlier:
                high_speed_dist += dist
                if sprint_frames >= MIN_SPRINT_FRAMES:
                    sprint_count += 1
                    sprint_dist += sprint_dist_candidate
                sprint_frames = 0
                sprint_dist_candidate = 0.0
            else:
                if sprint_frames >= MIN_SPRINT_FRAMES:
                    sprint_count += 1
                    sprint_dist += sprint_dist_candidate
                sprint_frames = 0
                sprint_dist_candidate = 0.0

            timeseries.append({
                "frame_id": f["frame_id"],
                "time_s": round((f["frame_id"] - 1) * dt, 3),
                "track_id": tid,
                "team_id": f["team_id"],
                "role": f["role"],
                "x_m": x,
                "y_m": y,
                "distance_m": round(dist, 3),
                "speed_kmh": round(speed, 3),
                "accel_mps2": round(accel, 3),
            })

        if sprint_frames >= MIN_SPRINT_FRAMES:
            sprint_count += 1
            sprint_dist += sprint_dist_candidate

        visible_frames = len(frames)
        visible_minutes = visible_frames / fps / 60.0
        summary[tid] = {
            "track_id": tid,
            "team_id": frames[0]["team_id"] if frames else -1,
            "role": frames[0]["role"] if frames else "unknown",
            "minutes_visible": round(visible_minutes, 2),
            "distance_m": round(total_dist, 2),
            "max_speed_kmh": round(max(speeds) if speeds else 0.0, 2),
            "mean_speed_kmh": round(np.mean(speeds) if speeds else 0.0, 2),
            "sprint_count": sprint_count,
            "sprint_distance_m": round(sprint_dist, 2),
            "high_speed_distance_m": round(high_speed_dist, 2),
        }

    return timeseries, summary


def compute_team_shape_metrics(rows, fps):
    dt = 1.0 / fps

    frame_teams = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("entity_type") != "player":
            continue
        role = r.get("role", "")
        if role in ("goalkeeper", "referee"):
            continue
        visible = int(r.get("visible", 0))
        if not visible:
            continue
        x = parse_numeric(r.get("x_m"))
        y = parse_numeric(r.get("y_m"))
        if x is None or y is None:
            continue
        tid = int(r.get("team_id", -1))
        if tid < 0:
            continue
        fid = int(r["frame_id"])
        frame_teams[fid][tid].append((x, y))

    results = []
    for fid in sorted(frame_teams.keys()):
        time_s = round((fid - 1) * dt, 3)
        for tid, pts in frame_teams[fid].items():
            n = len(pts)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            centroid_x = np.mean(xs)
            centroid_y = np.mean(ys)
            height_m = max(xs) - min(xs)
            width_m = max(ys) - min(ys)

            area = None
            compactness = None
            if n >= 3:
                try:
                    hull = ConvexHull(pts)
                    area = hull.volume
                    compactness = area / (height_m * width_m) if height_m * width_m > 0 else None
                except Exception:
                    area = None
                    compactness = None

            results.append({
                "frame_id": fid,
                "time_s": time_s,
                "team_id": tid,
                "n_players_visible": n,
                "centroid_x_m": round(centroid_x, 3),
                "centroid_y_m": round(centroid_y, 3),
                "height_m": round(height_m, 3),
                "width_m": round(width_m, 3),
                "convex_hull_area_m2": round(area, 3) if area is not None else '',
                "compactness": round(compactness, 3) if compactness is not None else '',
            })
    return results


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_tracking_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows from {args.input_csv}")

    print("Computing physical metrics...")
    timeseries, summary = compute_physical_metrics(rows, args.fps)

    base = os.path.splitext(os.path.basename(args.input_csv))[0]
    # Strip common suffixes for cleaner output names
    if base.endswith("_tracking_2d"):
        base = base[:-len("_tracking_2d")].rstrip("_")
    if not base:
        base = "metrics"
    ts_path = os.path.join(args.output_dir, f"{base}_player_physical_timeseries.csv")
    with open(ts_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "time_s", "track_id", "team_id", "role",
            "x_m", "y_m", "distance_m", "speed_kmh", "accel_mps2"
        ])
        writer.writeheader()
        writer.writerows(timeseries)
    print(f"  Timeseries -> {ts_path}")

    summary_path = os.path.join(args.output_dir, f"{base}_player_physical_metrics.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "team_id", "role", "minutes_visible", "distance_m",
            "max_speed_kmh", "mean_speed_kmh", "sprint_count", "sprint_distance_m",
            "high_speed_distance_m"
        ])
        writer.writeheader()
        writer.writerows(summary.values())
    print(f"  Summary -> {summary_path}")

    print("Computing team shape metrics...")
    shape = compute_team_shape_metrics(rows, args.fps)
    shape_path = os.path.join(args.output_dir, f"{base}_team_shape_metrics.csv")
    with open(shape_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "time_s", "team_id", "n_players_visible",
            "centroid_x_m", "centroid_y_m", "height_m", "width_m",
            "convex_hull_area_m2", "compactness"
        ])
        writer.writeheader()
        writer.writerows(shape)
    print(f"  Shape -> {shape_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
