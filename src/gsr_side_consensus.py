"""
Game-level left/right consensus for team side assignment.

The converter's per-clip heuristic (team with the smaller mean pitch-x plays
left) fails on clips where play is packed into one half: corners, deep attacks,
or a degenerate team clustering (e.g. SNGS-043 splits 6 vs 42 tracks). Measured
against GT it picks the wrong orientation on 5 of 37 valid sequences, and one
flipped clip zeroes the GS-HOTA similarity of every player in it.

The structural fix: clips of the same (game, half) share physical sides — teams
only swap at half-time. So instead of deciding each clip in isolation:

  1. parse each clip's two kit colors (hue/sat/val) from the team clustering
     post report, and its game/half from the GT `info` block (metadata: which
     game and minute the clip comes from, not an annotation);
  2. within a (game, half) group, match every clip's two clusters to the
     group's two kits by color distance;
  3. let each clip cast a weighted vote on which KIT plays left, using its own
     mean-x signal; the weight is the mean separation in meters times the
     cluster balance (min/max track count), so packed scenes and degenerate
     clusterings barely vote;
  4. every clip whose kit match is unambiguous inherits the group verdict; a
     clip whose two cluster-to-kit assignments are nearly tied (margin below
     MIN_KIT_MARGIN, i.e. its clusters resemble neither kit, typically due to
     impure clustering or lighting) keeps its local mean-x heuristic instead
     of inheriting a coin flip.

Standalone:  python src/gsr_side_consensus.py [--sequences ...]
Importable:  compute_side_consensus(outputs, gt_root, seqs) -> {seq: {tid: side}}
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

OUR_PITCH_L = 105.0
MIN_KIT_MARGIN = 0.10   # below this, the color match is a coin flip: stay local

_KIT_RE = re.compile(
    r"Team ([AB]): tracks=(\d+) hue=([\d.]+)\S* sat=([\d.]+) val=([\d.]+)")


def _read_clip(outputs, gt_root, seq):
    """Gather one clip's kit colors, mean-x per team and game/half metadata."""
    report = os.path.join(outputs, seq, "team_clustering_debug",
                          f"{seq}_post_report.txt")
    csv_path = os.path.join(outputs, seq, f"{seq}_tracking_2d.csv")
    labels = os.path.join(gt_root, seq, "Labels-GameState.json")
    if not (os.path.exists(report) and os.path.exists(csv_path)
            and os.path.exists(labels)):
        return None

    kits = {}   # team_id -> (tracks, hue, sat, val); Team A = id 0, Team B = id 1
    with open(report) as f:
        for m in _KIT_RE.finditer(f.read()):
            tid = 0 if m.group(1) == "A" else 1
            kits[tid] = (int(m.group(2)), float(m.group(3)),
                         float(m.group(4)), float(m.group(5)))
    if len(kits) != 2:
        return None

    xs = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["entity_type"] == "ball" or r["role"] not in ("player", "goalkeeper"):
                continue
            if r["team_id"] not in ("0", "1") or not r["x_m"]:
                continue
            xs[int(r["team_id"])].append(float(r["x_m"]) - OUR_PITCH_L / 2.0)
    if len(xs) != 2:
        return None

    info = json.load(open(labels))["info"]
    half = info.get("game_time_start", "? -").split(" - ")[0].strip()

    mean_x = {t: float(np.mean(v)) for t, v in xs.items()}
    balance = min(kits[0][0], kits[1][0]) / max(kits[0][0], kits[1][0])
    return {
        "group": (info["game_id"], half),
        "kits": kits,
        "mean_x": mean_x,
        "separation": abs(mean_x[0] - mean_x[1]),
        "balance": balance,
    }


def _kit_dist(a, b):
    """Perceptual-ish distance between two kits (hue circular on OpenCV 0-180)."""
    dh = abs(a[1] - b[1])
    dh = min(dh, 180.0 - dh) / 90.0
    ds = abs(a[2] - b[2]) / 255.0
    dv = abs(a[3] - b[3]) / 255.0
    return 0.6 * dh + 0.25 * ds + 0.15 * dv


def compute_side_consensus(outputs, gt_root, seqs, verbose=False):
    """Return {seq: {team_id: 'left'|'right'}} using game-half group consensus."""
    clips = {}
    for seq in seqs:
        info = _read_clip(outputs, gt_root, seq)
        if info is not None:
            clips[seq] = info

    groups = defaultdict(list)
    for seq, info in clips.items():
        groups[info["group"]].append(seq)

    mapping = {}
    for group, members in sorted(groups.items()):
        # Reference clip: the most trustworthy local signal anchors the kits.
        ref = max(members,
                  key=lambda s: clips[s]["separation"] * clips[s]["balance"])
        ref_kits = clips[ref]["kits"]

        # Match every clip's clusters to the reference kits (direct or swapped).
        kit_of = {}      # seq -> {team_id: kit_index}
        ambiguous = set()  # kit match is a near-tie: don't inherit the verdict
        for seq in members:
            k = clips[seq]["kits"]
            direct = _kit_dist(k[0], ref_kits[0]) + _kit_dist(k[1], ref_kits[1])
            swapped = _kit_dist(k[0], ref_kits[1]) + _kit_dist(k[1], ref_kits[0])
            kit_of[seq] = {0: 0, 1: 1} if direct <= swapped else {0: 1, 1: 0}
            if abs(direct - swapped) < MIN_KIT_MARGIN:
                ambiguous.add(seq)

        # Weighted vote: positive -> kit 0 plays left. Ambiguous clips still
        # vote (their weight is what it is) but won't inherit the verdict.
        vote = 0.0
        for seq in members:
            if seq in ambiguous:
                continue
            c = clips[seq]
            sign_team0_left = 1.0 if c["mean_x"][0] < c["mean_x"][1] else -1.0
            sign_kit0_left = sign_team0_left if kit_of[seq][0] == 0 else -sign_team0_left
            vote += sign_kit0_left * c["separation"] * c["balance"]
        kit0_side = "left" if vote >= 0 else "right"
        kit1_side = "right" if kit0_side == "left" else "left"

        for seq in members:
            c = clips[seq]
            local0 = "left" if c["mean_x"][0] < c["mean_x"][1] else "right"
            if seq in ambiguous:
                sides = {0: local0, 1: ("right" if local0 == "left" else "left")}
            else:
                sides = {t: (kit0_side if kit_of[seq][t] == 0 else kit1_side)
                         for t in (0, 1)}
            mapping[seq] = sides
            if verbose:
                tag = " (AMBIGUO: conserva local)" if seq in ambiguous else \
                    (" <- CORRIGE local" if sides[0] != local0 else "")
                print(f"  {seq}: 0:{sides[0]},1:{sides[1]} "
                      f"(sep {c['separation']:.1f} m, balance {c['balance']:.2f})"
                      f"{tag}")
        if verbose:
            print(f"  grupo partido={group[0]} mitad={group[1]}: "
                  f"ancla {ref}, voto {vote:+.1f} -> kit0={kit0_side}\n")
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--gt_root", default="data/SoccerNetGS/gamestate-2024/valid")
    ap.add_argument("--sequences", nargs="*", default=None)
    args = ap.parse_args()

    seqs = args.sequences or sorted(
        d for d in os.listdir(args.outputs)
        if d.startswith("SNGS-")
        and os.path.exists(os.path.join(args.outputs, d, f"{d}_tracking_2d.csv")))
    compute_side_consensus(args.outputs, args.gt_root, seqs, verbose=True)


if __name__ == "__main__":
    main()
