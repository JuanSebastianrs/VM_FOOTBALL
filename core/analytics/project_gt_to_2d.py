"""
Project SoccerNet GT (gt.txt) onto the 2D pitch using pre-computed H_inv.

Usage:
    python core/analytics/project_gt_to_2d.py \
        --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
        --calibration_json outputs/SNMOT-148/calibration_hinv.json \
        --output_csv outputs/SNMOT-148/SNMOT-148_gt_tracking_2d.csv \
        --fps 25.0
"""

import os
import sys
import csv
import json
import argparse
import configparser
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.mapping.tactical_vision_2d_mapper import project_point_to_world


def parse_gameinfo(sequence_dir):
    path = os.path.join(sequence_dir, "gameinfo.ini")
    config = configparser.ConfigParser()
    config.read(path, encoding='utf-8')

    tracklet_map = {}
    if not config.has_section("Sequence"):
        return tracklet_map

    seq = config["Sequence"]
    num_tracklets = int(seq.get("num_tracklets", 0))

    for i in range(1, num_tracklets + 1):
        key = f"trackletID_{i}"
        val = seq.get(key, "")
        if ";" in val:
            desc, _ = val.rsplit(";", 1)
        else:
            desc = val

        desc_lower = desc.lower().strip()
        if desc_lower == "ball":
            tracklet_map[i] = {"team_id": -1, "role": "ball"}
        elif "referee" in desc_lower:
            tracklet_map[i] = {"team_id": -2, "role": "referee"}
        elif "goalkeepers" in desc_lower:
            team_id = 0 if "left" in desc_lower else 1
            tracklet_map[i] = {"team_id": team_id, "role": "goalkeeper"}
        elif "player" in desc_lower:
            team_id = 0 if "left" in desc_lower else 1
            tracklet_map[i] = {"team_id": team_id, "role": "player"}
        else:
            tracklet_map[i] = {"team_id": -1, "role": "unknown"}

    return tracklet_map


def load_gt(gt_path):
    rows = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            rows.append({
                "frame": int(parts[0]),
                "id": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "w": float(parts[4]),
                "h": float(parts[5]),
                "conf": float(parts[6]),
                "class": int(parts[7]) if len(parts) > 7 else -1,
                "visibility": float(parts[8]) if len(parts) > 8 else 1.0,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--calibration_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    gt_path = os.path.join(args.sequence_dir, "gt", "gt.txt")
    print(f"Loading GT from {gt_path}...")
    gt_rows = load_gt(gt_path)
    print(f"  {len(gt_rows)} GT rows loaded.")

    tracklet_map = parse_gameinfo(args.sequence_dir)
    print(f"  {len(tracklet_map)} tracklets parsed.")

    print(f"Loading calibration from {args.calibration_json}...")
    with open(args.calibration_json, 'r', encoding='utf-8') as f:
        calib = json.load(f)
    print(f"  {len(calib)} frames calibrated.")

    dt = 1.0 / args.fps
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_id', 'time_s', 'entity_type', 'track_id', 'team_id', 'role',
            'x_m', 'y_m', 'visible', 'source', 'confidence'
        ])

        for row in gt_rows:
            fid = row["frame"]
            tid = row["id"]
            info = tracklet_map.get(tid, {"team_id": -1, "role": "unknown"})

            if info['role'] == 'ball':
                x_pt = row["x"] + row["w"] / 2.0
                y_pt = row["y"] + row["h"] / 2.0
            else:
                x_pt = row["x"] + row["w"] / 2.0
                y_pt = row["y"] + row["h"]

            frame_calib = calib.get(str(fid))
            if frame_calib is not None and frame_calib.get("H_inv") is not None:
                H_inv = np.array(frame_calib["H_inv"])
                x_world, y_world = project_point_to_world(x_pt, y_pt, H_inv)
                visible = x_world is not None
            else:
                visible = False
                x_world = None
                y_world = None

            writer.writerow([
                fid,
                round((fid - 1) * dt, 3),
                'ball' if info['role'] == 'ball' else 'player',
                tid,
                info['team_id'],
                info['role'],
                round(x_world, 3) if visible else '',
                round(y_world, 3) if visible else '',
                1 if visible else 0,
                'gt',
                row['conf']
            ])

    print(f"\nGT tracking 2D CSV saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
