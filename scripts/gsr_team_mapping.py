"""
Derive the team_id -> {left,right} mapping that core/identity/jersey_identity_phase.py
needs in order to apply a roster mask.

On SNMOT this mapping comes from the `audit` phase, which reads gameinfo.ini. GSR
has no gameinfo.ini, so we recover the side from geometry instead: after map2d we
have every track's pitch position in meters, and GSR's pitch frame is centered
(x < 0 is the left half). The team whose players average a smaller x plays left.

The mapping is only meaningful once map2d has produced <seq>_tracking_2d.csv.

Prints "0:left,1:right", ready to hand to --team_mapping. With --check it also
reads the GT and reports whether the recovered sides agree with the annotation.

Usage:
    python scripts/gsr_team_mapping.py --sequence SNGS-021
    python scripts/gsr_team_mapping.py --all --check
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

OUR_PITCH_L = 105.0  # our map2d frame: corner origin, x in [0, 105]
OUR_PITCH_W = 68.0   #                                 y in [0, 68]


def read_our_tracks(csv_path):
    """frame -> [(team_id, x, y)] in GSR pitch meters, players/goalkeepers only."""
    per_frame = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["entity_type"] == "ball" or r["role"] not in ("player", "goalkeeper"):
                continue
            if r["team_id"] in ("", "-1", "-2") or not r["x_m"] or not r["y_m"]:
                continue
            per_frame[int(r["frame_id"])].append(
                (int(r["team_id"]),
                 float(r["x_m"]) - OUR_PITCH_L / 2.0,
                 float(r["y_m"]) - OUR_PITCH_W / 2.0))
    return per_frame


def our_sides(csv_path):
    """team_id -> 'left'/'right', from mean pitch x of players/goalkeepers.

    The clip may show only part of the pitch, so both means can share a sign;
    only their ORDER carries the side. The team averaging a smaller x plays left.
    """
    xs = defaultdict(list)
    for dets in read_our_tracks(csv_path).values():
        for team_id, x, _ in dets:
            xs[team_id].append(x)
    means = {t: float(np.mean(v)) for t, v in xs.items() if v}
    if len(means) < 2:
        return {t: "left" for t in means}, means
    ordered = sorted(means, key=means.get)
    sides = {ordered[0]: "left", ordered[-1]: "right"}
    for t in means:
        sides.setdefault(t, "left" if means[t] < means[ordered[0]] else "right")
    return sides, means


def read_gt_tracks(labels_path):
    """frame -> [(side, x, y)] in GSR pitch meters, players/goalkeepers only."""
    with open(labels_path) as f:
        data = json.load(f)
    frame_of = {}
    for img in data["images"]:
        name = os.path.splitext(os.path.basename(img.get("file_name", "")))[0]
        try:
            frame_of[img["image_id"]] = int(name)
        except ValueError:
            frame_of[img["image_id"]] = int(img["image_id"][-6:])

    per_frame = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes") or {}
        if attrs.get("role") not in ("player", "goalkeeper"):
            continue
        side, bp = attrs.get("team"), ann.get("bbox_pitch")
        if side not in ("left", "right") or not bp:
            continue
        frame = frame_of.get(ann["image_id"])
        if frame is not None:
            per_frame[frame].append((side, bp["x_bottom_middle"], bp["y_bottom_middle"]))
    return per_frame


def side_agreement(csv_path, labels_path, sides, tol=2.0):
    """Match each of our detections to the nearest GT one (within `tol` meters) and
    count how often our recovered side equals the annotated side."""
    ours, gt = read_our_tracks(csv_path), read_gt_tracks(labels_path)
    hit = miss = unmatched = 0
    for frame, dets in ours.items():
        gt_dets = gt.get(frame)
        if not gt_dets:
            continue
        gt_xy = np.array([[x, y] for _, x, y in gt_dets])
        gt_side = [s for s, _, _ in gt_dets]
        for team_id, x, y in dets:
            d = np.hypot(gt_xy[:, 0] - x, gt_xy[:, 1] - y)
            j = int(np.argmin(d))
            if d[j] > tol:
                unmatched += 1
            elif gt_side[j] == sides.get(team_id):
                hit += 1
            else:
                miss += 1
    total = hit + miss
    return (hit / total if total else float("nan")), total, unmatched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true", help="compara con el GT")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--gt_root", default="data/SoccerNetGS/gamestate-2024/valid")
    args = ap.parse_args()

    if args.all:
        seqs = sorted(d.name for d in Path(args.outputs).iterdir()
                      if d.is_dir() and d.name.startswith("SNGS-"))
    else:
        seqs = [args.sequence]

    accs = []
    for seq in seqs:
        csv_path = os.path.join(args.outputs, seq, f"{seq}_tracking_2d.csv")
        if not os.path.exists(csv_path):
            if not args.all:
                raise SystemExit(f"falta {csv_path} (corre la fase map2d primero)")
            continue
        sides, means = our_sides(csv_path)
        mapping = ",".join(f"{t}:{s}" for t, s in sorted(sides.items()))

        if args.check:
            acc, matched, unmatched = side_agreement(
                csv_path, os.path.join(args.gt_root, seq, "Labels-GameState.json"), sides)
            gap = abs(means[max(means, key=means.get)] - means[min(means, key=means.get)]) \
                if len(means) > 1 else 0.0
            accs.append(acc)
            print(f"{seq}: {mapping:<20} | separacion {gap:5.1f} m "
                  f"| lado correcto {100*acc:5.1f}% ({matched} emparejados, {unmatched} sin GT)")
        else:
            print(mapping)

    if args.check and accs:
        accs = np.array(accs)
        print(f"\nlado correcto: media {100*accs.mean():.1f}%  "
              f"min {100*accs.min():.1f}%  sobre {len(accs)} secuencias")


if __name__ == "__main__":
    main()
