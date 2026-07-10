"""
Convert TacticalVision pipeline outputs to SoccerNet-GSR prediction JSONs and
run the official GS-HOTA evaluation (sn-trackeval fork).

For each GSR sequence we already ran the pipeline on, this reads:
  outputs/<seq>/<seq>_tracking_2d.csv          pitch positions (x_m, y_m), role, team_id, track_id
  outputs/<seq>/<seq>_jersey_identity_v2_all.json   locked jersey numbers per track (optional)
and writes one prediction file per sequence in the layout the evaluator wants:
  <trackers>/SoccerNetGS-<split>/<tracker>/data/<seq>.json

The GS-HOTA matcher (see trackeval/datasets/soccernet_gs.py) only uses the
bottom-middle pitch point for the gaussian distance, and zeroes the similarity
when role/team/jersey disagree. So the load-bearing outputs are: pitch (x,y) in
GSR meters, role in {player,goalkeeper,referee,other}, team in {left,right,None},
jersey int or None.

Two things are calibrated against a real GT file (read_gt_reference):
  1. PITCH COORDINATE TRANSFORM from our 105x68 corner-origin frame to the GSR frame.
  2. IMAGE_ID format, which must match the GT's image ids exactly.
"""

import os
import sys
import csv
import json
import glob
import argparse
import collections

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Our pitch model: origin at a corner, length 105 m (x), width 68 m (y).
OUR_PITCH_L = 105.0
OUR_PITCH_W = 68.0

ROLE_MAP = {  # our roles -> GSR roles
    "player": "player",
    "goalkeeper": "goalkeeper",
    "referee": "referee",
    "ball": "ball",
}


def to_gsr_pitch(x_m, y_m, transform):
    """Map our (x_m, y_m) in [0,105]x[0,68] to GSR pitch coords (meters, centered).

    GSR convention (confirmed against GT in read_gt_reference): origin at pitch
    center, x in [-52.5, 52.5], y in [-34, 34]. `transform` carries the offsets
    and axis signs actually observed in the GT so this stays data-driven.
    """
    gx = (x_m - transform["x0"]) * transform["sx"]
    gy = (y_m - transform["y0"]) * transform["sy"]
    return gx, gy


def load_tracks(seq_dir, seq, use_jersey=True, jersey_states=("locked",)):
    """Read our per-frame pitch tracks + jersey numbers for one sequence.

    Returns (rows, jersey) where jersey maps track_id -> (number_str, confidence).
    """
    csv_path = os.path.join(seq_dir, f"{seq}_tracking_2d.csv")
    rows = collections.defaultdict(list)  # frame_id -> [dict, ...]
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["entity_type"] == "ball" or not r["x_m"] or not r["y_m"]:
                continue
            rows[int(r["frame_id"])].append({
                "track_id": int(r["track_id"]),
                "team_id": int(r["team_id"]) if r["team_id"] not in ("", "-1") else -1,
                "role": r["role"] or "player",
                "x_m": float(r["x_m"]),
                "y_m": float(r["y_m"]),
            })

    jersey = {}
    if use_jersey:
        for name in (f"{seq}_jersey_identity.json", f"{seq}_jersey_identity_v2_all.json"):
            jpath = os.path.join(seq_dir, name)
            if os.path.exists(jpath):
                for e in json.load(open(jpath)).get("tracklets", []):
                    if e.get("state") in jersey_states and e.get("predicted_number") is not None:
                        # GT stores jersey as a string ("3"); soccernet_gs.py compares
                        # them with `==` on object arrays, so an int never matches.
                        jersey[e["track_id"]] = (str(int(e["predicted_number"])),
                                                 float(e.get("confidence") or 0.0))
                break
    return rows, jersey


def build_identity_chains(seq_dir, seq):
    """track_id -> canonical track_id, via the offline tracklet linker.

    ByteTrack splits a player into ~1.9 tracklets per GSR clip. The linker
    (core/tracking/offline_tracklet_linker.py, +2.21 HOTA on SNMOT) re-joins them
    offline using CMC-stabilized constant-velocity prediction. Returns {} if the
    inputs the linker needs are missing.
    """
    sys.path.insert(0, BASE)
    from core.tracking.offline_tracklet_linker import (
        build_global_transforms, link_tracklets, load_tracklets)

    det_path = os.path.join(seq_dir, f"{seq}_detections.json")
    team_path = os.path.join(seq_dir, f"{seq}_team_assignments.json")
    cmc_path = os.path.join(seq_dir, f"{seq}_cmc.json")
    if not os.path.exists(det_path):
        return {}

    with open(det_path) as f:
        detections = json.load(f)
    team = json.load(open(team_path)) if os.path.exists(team_path) else None
    cmc = json.load(open(cmc_path)) if os.path.exists(cmc_path) else None

    tracklets = load_tracklets(detections, team)
    if not tracklets:
        return {}
    transforms = build_global_transforms(cmc, max(t.end for t in tracklets))
    return link_tracklets(tracklets, transforms)


def fill_missing_teams(rows, chains):
    """Give unclustered player tracks (team_id < 0) a team.

    The team clustering only labels tracks with enough frames, which with
    ByteTrack fragmentation leaves ~24% of player detections with no team on
    GSR. In GS-HOTA a player with team None can never match (GT players always
    carry a side), so any informed guess strictly dominates abstaining.
    Primary signal: majority team among the track's identity-chain siblings.
    Fallback: whichever team's positional centroid is closer to the track mean.
    """
    track_team, track_pos = {}, collections.defaultdict(list)
    for dets in rows.values():
        for d in dets:
            if d["team_id"] >= 0:
                track_team[d["track_id"]] = d["team_id"]
            track_pos[d["track_id"]].append((d["x_m"], d["y_m"]))

    chain_votes = collections.defaultdict(collections.Counter)
    for tid, cid in (chains or {}).items():
        if tid in track_team:
            chain_votes[cid][track_team[tid]] += 1

    centroids = {}
    for tid, team in track_team.items():
        centroids.setdefault(team, []).extend(track_pos[tid])
    centroids = {t: np.mean(v, axis=0) for t, v in centroids.items() if v}

    def guess(tid):
        votes = chain_votes.get((chains or {}).get(tid))
        if votes:
            return votes.most_common(1)[0][0]
        if len(centroids) == 2 and track_pos[tid]:
            p = np.mean(track_pos[tid], axis=0)
            return min(centroids, key=lambda t: np.hypot(*(centroids[t] - p)))
        return -1

    filled = 0
    for dets in rows.values():
        for d in dets:
            if d["team_id"] < 0 and d["role"] in ("player", "goalkeeper"):
                t = track_team.get(d["track_id"], None)
                if t is None:
                    t = guess(d["track_id"])
                if t >= 0:
                    d["team_id"] = t
                    filled += 1
    return filled


def propagate_jersey(jersey, chains, conflict="confidence"):
    """Spread each chain's jersey number to every fragment of that chain.

    A GS-HOTA match dies on any jersey disagreement, and the OCR only reads a
    number on part of a player's fragments. Without propagation the other
    fragments carry None and lose against a GT track that does carry a number.

    When two fragments of one chain were read as different numbers, the chain is
    either mis-linked or one read is wrong; `conflict="abstain"` drops the chain
    (safe: abstaining still matches GT tracks whose number was never annotated),
    `conflict="confidence"` keeps the most confident read.
    """
    per_chain = collections.defaultdict(list)
    for tid, (number, conf) in jersey.items():
        per_chain[chains.get(tid, tid)].append((conf, number))

    chain_number = {}
    for cid, reads in per_chain.items():
        if conflict == "abstain" and len({n for _, n in reads}) > 1:
            continue
        chain_number[cid] = max(reads)[1]

    out = dict(jersey)  # fragments the OCR read keep their own number
    for tid, cid in chains.items():
        if cid in chain_number:
            out[tid] = (chain_number[cid], 1.0)
    return out


def assign_left_right(rows, transform):
    """Assign each team_id to 'left'/'right' by mean pitch x over the sequence."""
    xs = collections.defaultdict(list)
    for dets in rows.values():
        for d in dets:
            if d["role"] in ("player", "goalkeeper") and d["team_id"] >= 0:
                gx, _ = to_gsr_pitch(d["x_m"], d["y_m"], transform)
                xs[d["team_id"]].append(gx)
    means = {t: float(np.mean(v)) for t, v in xs.items() if v}
    if len(means) < 2:
        return {t: "left" for t in means}
    ordered = sorted(means, key=means.get)
    return {ordered[0]: "left", ordered[-1]: "right"}


def build_predictions(seq, rows, jersey, transform, frame_to_image_id, chains=None,
                      team_side=None):
    """Assemble the GSR `predictions` list for one sequence."""
    if team_side is None:
        team_side = assign_left_right(rows, transform)
    preds = []
    for frame_id, dets in sorted(rows.items()):
        image_id = frame_to_image_id(frame_id)
        if image_id is None:
            continue
        for d in dets:
            role = ROLE_MAP.get(d["role"], "other")
            is_pg = role in ("player", "goalkeeper")
            team = team_side.get(d["team_id"]) if is_pg else None
            read = jersey.get(d["track_id"])
            jn = read[0] if (read and role == "player") else None
            track_id = chains.get(d["track_id"], d["track_id"]) if chains else d["track_id"]
            gx, gy = to_gsr_pitch(d["x_m"], d["y_m"], transform)
            preds.append({
                "image_id": image_id,
                "track_id": track_id,
                "supercategory": "object",
                "category_id": 1,
                "confidence": 1.0,
                "bbox_pitch": {
                    "x_bottom_left": gx, "y_bottom_left": gy,
                    "x_bottom_middle": gx, "y_bottom_middle": gy,
                    "x_bottom_right": gx, "y_bottom_right": gy,
                },
                "attributes": {
                    "role": role,
                    "team": team,
                    "jersey": jn,
                },
            })
    return {"predictions": preds}


def read_gt_reference(gt_seq_dir):
    """Read one GT Labels-GameState.json to lock the coordinate transform and
    the frame->image_id mapping. Returns (transform, frame_to_image_id)."""
    gt = json.load(open(os.path.join(gt_seq_dir, "Labels-GameState.json")))

    # image_id per frame: GT image ids are strings; map trailing frame index.
    frame_of = {}
    for img in gt["images"]:
        # image_id like "<split><seq><frame:06d>"; recover frame from file_name.
        fname = os.path.splitext(os.path.basename(img.get("file_name", "")))[0]
        try:
            frame = int(fname)
        except ValueError:
            frame = int(img["image_id"][-6:])
        frame_of[frame] = img["image_id"]

    def frame_to_image_id(frame_id):
        return frame_of.get(frame_id)

    # GSR uses a centered pitch (x in [-52.5,52.5], y in [-34,34]); our frame is
    # corner-origin 105x68. Offsets are half-dimensions; sign of y is confirmed
    # by checking the GT's own bbox_pitch spread below.
    transform = {"x0": OUR_PITCH_L / 2.0, "y0": OUR_PITCH_W / 2.0, "sx": 1.0, "sy": 1.0}
    return transform, frame_to_image_id, frame_of


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", default="data/SoccerNetGS/gamestate-2024/valid")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--trackers-root", default="data/gsr_trackers")
    ap.add_argument("--tracker-name", default="TacticalVision")
    ap.add_argument("--no-jersey", action="store_true")
    ap.add_argument("--jersey-states", nargs="*", default=["locked"],
                    help="estados del modulo de dorsales aceptados como prediccion "
                         "(locked, tentative, inferred)")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--link-tracks", action="store_true",
                    help="reasigna los track_id a la raiz de su cadena de identidad "
                         "(vinculacion offline de tracklets)")
    ap.add_argument("--propagate-jersey", action="store_true",
                    help="extiende el dorsal leido a todos los fragmentos de la cadena")
    ap.add_argument("--jersey-conflict", choices=("confidence", "abstain"),
                    default="confidence",
                    help="que hacer si una cadena tiene dos dorsales distintos")
    ap.add_argument("--side-consensus", action="store_true",
                    help="orientacion izquierda/derecha por consenso de grupo "
                         "(partido, mitad) via matching de kits, en vez de la "
                         "media de x por clip")
    ap.add_argument("--fill-teams", action="store_true",
                    help="asigna equipo a los tracks sin clusterizar (cadena de "
                         "identidad, con respaldo por centroide)")
    args = ap.parse_args()

    gt_seqs = sorted(
        d for d in os.listdir(args.gt_root)
        if os.path.isdir(os.path.join(args.gt_root, d))
    )
    if args.sequences:
        gt_seqs = [s for s in gt_seqs if s in args.sequences]

    tracker_data = os.path.join(
        args.trackers_root, f"SoccerNetGS-{args.split}", args.tracker_name, "data")
    os.makedirs(tracker_data, exist_ok=True)

    side_map = {}
    if args.side_consensus:
        sys.path.insert(0, BASE)
        from src.gsr_side_consensus import compute_side_consensus
        ready = [s for s in gt_seqs if os.path.exists(
            os.path.join(args.outputs, s, f"{s}_tracking_2d.csv"))]
        side_map = compute_side_consensus(args.outputs, args.gt_root, ready)
        print(f"Consenso de lados: {len(side_map)} secuencias mapeadas")

    written = skipped = 0
    for seq in gt_seqs:
        our_dir = os.path.join(args.outputs, seq)
        if not os.path.exists(os.path.join(our_dir, f"{seq}_tracking_2d.csv")):
            print(f"  [SKIP] {seq}: no tracking_2d.csv (pipeline not run yet)")
            skipped += 1
            continue
        transform, frame_to_image_id, _ = read_gt_reference(
            os.path.join(args.gt_root, seq))
        rows, jersey = load_tracks(our_dir, seq, use_jersey=not args.no_jersey,
                                   jersey_states=tuple(args.jersey_states))

        chains = {}
        if args.link_tracks or args.propagate_jersey or args.fill_teams:
            chains = build_identity_chains(our_dir, seq)
        if args.propagate_jersey and chains:
            jersey = propagate_jersey(jersey, chains, args.jersey_conflict)
        if args.fill_teams:
            fill_missing_teams(rows, chains)

        pred = build_predictions(seq, rows, jersey, transform, frame_to_image_id,
                                 chains=chains if args.link_tracks else None,
                                 team_side=side_map.get(seq))
        with open(os.path.join(tracker_data, f"{seq}.json"), "w") as f:
            json.dump(pred, f)
        n_chains = len(set(chains.values())) if chains else 0
        extra = f", {len(chains)} tracklets -> {n_chains} identidades" if chains else ""
        print(f"  [OK] {seq}: {len(pred['predictions'])} predictions "
              f"({len(jersey)} jerseys{extra})")
        written += 1

    print(f"\nWrote {written} sequences, skipped {skipped}.")
    print(f"Predictions in: {tracker_data}")


if __name__ == "__main__":
    main()
