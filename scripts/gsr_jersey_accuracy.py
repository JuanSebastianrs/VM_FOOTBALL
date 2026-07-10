"""
Score a jersey_identity JSON against the SoccerNet-GSR ground truth.

GS-HOTA zeroes the similarity of a detection whose jersey disagrees with the GT,
and GT jerseys are constant per track (a number on every frame, or None on every
frame). So a jersey prediction has three possible effects:

  correct    predicted N, GT track carries N          -> recovers the track
  wrong      predicted N, GT track carries M != N     -> loses a track we had
  spurious   predicted N, GT track carries None       -> loses a track we had
  abstain    predicted None                           -> matches iff GT is None

That makes "how many did we lock" the wrong question: locking a number on a track
the annotators never labelled is a net loss. This script reports the breakdown.

Our track ids are matched to GT track ids by nearest-neighbour in pitch meters,
majority-voted over the frames where both exist.

Usage:
    python scripts/gsr_jersey_accuracy.py --sequence SNGS-021 \
        --jersey_json outputs/SNGS-021/SNGS-021_jersey_rosterGame.json
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

OUR_PITCH_L, OUR_PITCH_W = 105.0, 68.0


def read_our(csv_path):
    per_frame = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["entity_type"] == "ball" or r["role"] != "player":
                continue
            if not r["x_m"] or not r["y_m"]:
                continue
            per_frame[int(r["frame_id"])].append(
                (int(r["track_id"]),
                 float(r["x_m"]) - OUR_PITCH_L / 2.0,
                 float(r["y_m"]) - OUR_PITCH_W / 2.0))
    return per_frame


def read_gt(labels_path):
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
    jersey_of = {}
    for ann in data["annotations"]:
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes") or {}
        if attrs.get("role") != "player":
            continue
        bp = ann.get("bbox_pitch")
        frame = frame_of.get(ann["image_id"])
        if not bp or frame is None:
            continue
        tid = ann["track_id"]
        jersey_of[tid] = attrs.get("jersey")
        per_frame[frame].append((tid, bp["x_bottom_middle"], bp["y_bottom_middle"]))
    return per_frame, jersey_of


def match_tracks(ours, gt, tol=2.0):
    """our track_id -> GT track_id, by majority nearest-neighbour vote."""
    votes = defaultdict(Counter)
    for frame, dets in ours.items():
        gt_dets = gt.get(frame)
        if not gt_dets:
            continue
        xy = np.array([[x, y] for _, x, y in gt_dets])
        ids = [t for t, _, _ in gt_dets]
        for tid, x, y in dets:
            d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
            j = int(np.argmin(d))
            if d[j] <= tol:
                votes[tid][ids[j]] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items() if c}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--jersey_json", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--gt_root", default="data/SoccerNetGS/gamestate-2024/valid")
    ap.add_argument("--states", nargs="*", default=["locked"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    seq = args.sequence
    ours = read_our(os.path.join(args.outputs, seq, f"{seq}_tracking_2d.csv"))
    gt, gt_jersey = read_gt(os.path.join(args.gt_root, seq, "Labels-GameState.json"))
    mapping = match_tracks(ours, gt)

    with open(args.jersey_json) as f:
        tracklets = json.load(f).get("tracklets", [])

    correct = wrong = spurious = unmatched = 0
    rows = []
    for t in tracklets:
        if t.get("state") not in args.states or t.get("predicted_number") is None:
            continue
        pred = str(int(t["predicted_number"]))
        gt_tid = mapping.get(t["track_id"])
        if gt_tid is None:
            unmatched += 1
            verdict = "sin GT"
            truth = "-"
        else:
            truth = gt_jersey.get(gt_tid)
            if truth is None:
                spurious += 1
                verdict = "ESPURIO (GT None)"
                truth = "None"
            elif str(truth) == pred:
                correct += 1
                verdict = "correcto"
            else:
                wrong += 1
                verdict = "ERRONEO"
        rows.append((t["track_id"], pred, truth, t.get("confidence", 0.0), verdict))

    predicted = correct + wrong + spurious + unmatched
    # Upper bound: GT player tracks that carry a number and that we actually track.
    gt_tracks_with_number = {g for g in set(mapping.values()) if gt_jersey.get(g) is not None}

    if not args.quiet:
        print(f"{'our_tid':>7} {'pred':>5} {'gt':>6} {'conf':>6}  veredicto")
        for tid, pred, truth, conf, verdict in sorted(rows):
            print(f"{tid:>7} {pred:>5} {str(truth):>6} {conf:>6.3f}  {verdict}")

    print(f"\n{seq} | estados={args.states} | {os.path.basename(args.jersey_json)}")
    print(f"  predicciones emitidas : {predicted}")
    print(f"    correctas           : {correct}")
    print(f"    erroneas            : {wrong}")
    print(f"    espurias (GT None)  : {spurious}")
    print(f"    sin GT emparejado   : {unmatched}")
    if predicted:
        print(f"  precision util        : {100*correct/predicted:.1f}% "
              f"({correct}/{predicted})")
        print(f"  danino (err+esp)      : {100*(wrong+spurious)/predicted:.1f}%")
    print(f"  tracks GT con dorsal que seguimos: {len(gt_tracks_with_number)}")
    print(f"  cobertura             : {100*correct/max(len(gt_tracks_with_number),1):.1f}%")


if __name__ == "__main__":
    main()
