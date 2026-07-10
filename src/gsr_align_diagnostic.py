"""
Diagnose the pitch coordinate registration between our map2d output and the
SoccerNet-GSR ground truth for one sequence.

A camera-projected pitch admits only two relative orientations: identity, or a
180-degree rotation (x -> -x, y -> -y, which also swaps left/right). This tool
projects our tracking_2d positions into GSR coordinates under each candidate
transform and reports the mean nearest-neighbour distance to the GT athletes of
the same frame. The transform with the smallest distance is the correct global
registration of the reference frame (it aligns the coordinate system, not
individual identities).

Usage:
    python src/gsr_align_diagnostic.py --seq SNGS-021
"""

import os
import csv
import json
import argparse
import collections

import numpy as np

OUR_PITCH_L = 105.0
OUR_PITCH_W = 68.0

# Candidate transforms from our (x_m in [0,105], y_m in [0,68], corner origin)
# to GSR centered meters. Each is (name, x0, y0, sx, sy).
CANDIDATES = [
    ("identity",  OUR_PITCH_L / 2, OUR_PITCH_W / 2,  1.0,  1.0),
    ("rot180",    OUR_PITCH_L / 2, OUR_PITCH_W / 2, -1.0, -1.0),
    ("flip_x",    OUR_PITCH_L / 2, OUR_PITCH_W / 2, -1.0,  1.0),
    ("flip_y",    OUR_PITCH_L / 2, OUR_PITCH_W / 2,  1.0, -1.0),
]


def load_our_positions(seq):
    path = os.path.join("outputs", seq, f"{seq}_tracking_2d.csv")
    by_frame = collections.defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["entity_type"] == "ball" or not r["x_m"] or not r["y_m"]:
                continue
            by_frame[int(r["frame_id"])].append((float(r["x_m"]), float(r["y_m"])))
    return by_frame


def load_gt_positions(seq):
    gt = json.load(open(os.path.join(
        "data/SoccerNetGS/gamestate-2024/valid", seq, "Labels-GameState.json")))
    frame_of = {}
    for img in gt["images"]:
        frame_of[img["image_id"]] = int(os.path.splitext(img["file_name"])[0])
    by_frame = collections.defaultdict(list)
    for a in gt["annotations"]:
        if a.get("supercategory") != "object":
            continue
        if a.get("attributes", {}).get("role") == "ball":
            continue
        bp = a.get("bbox_pitch")
        if not bp:
            continue
        fr = frame_of.get(a["image_id"])
        if fr is not None:
            by_frame[fr].append((bp["x_bottom_middle"], bp["y_bottom_middle"]))
    return by_frame


def mean_nn_distance(ours, gt, x0, y0, sx, sy):
    dists = []
    for frame, pts in ours.items():
        gpts = gt.get(frame)
        if not gpts:
            continue
        gpts = np.array(gpts)
        for x_m, y_m in pts:
            gx = (x_m - x0) * sx
            gy = (y_m - y0) * sy
            d = np.sqrt(((gpts - [gx, gy]) ** 2).sum(axis=1))
            dists.append(d.min())
    return float(np.mean(dists)) if dists else float("inf"), len(dists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    args = ap.parse_args()

    ours = load_our_positions(args.seq)
    gt = load_gt_positions(args.seq)
    print(f"{args.seq}: our frames={len(ours)} gt frames={len(gt)}")
    print(f"{'transform':<12}{'mean_nn_dist_m':>16}{'n_matched':>12}")
    results = []
    for name, x0, y0, sx, sy in CANDIDATES:
        d, n = mean_nn_distance(ours, gt, x0, y0, sx, sy)
        results.append((d, name))
        print(f"{name:<12}{d:>16.2f}{n:>12}")
    best = min(results)
    print(f"\nBest registration: {best[1]} (mean NN distance {best[0]:.2f} m)")


if __name__ == "__main__":
    main()
