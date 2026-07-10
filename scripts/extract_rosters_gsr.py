"""
Extract team rosters for SoccerNet-GSR sequences from Labels-GameState.json.

SNMOT ships a `gameinfo.ini` per clip that lists each tracklet's team side and
jersey number; `scripts/extract_rosters.py` turns that into rosters.json. GSR has
no such file -- the only place the numbers live is the ground-truth annotation.
So the roster produced here is an ORACLE roster: it is derived from the labels of
the split being evaluated. Any result obtained with it must be reported as
"roster-constrained", never as an unconstrained end-to-end number.

Two scopes:
  --scope clip  one roster per sequence, from the numbers visible in that clip.
                This is the exact analogue of SNMOT's gameinfo.ini.
  --scope game  one roster per game_id (union over every clip of that game),
                replicated to each of its sequences. A weaker, more realistic
                constraint: it is the team sheet, not the list of players on
                screen right now. This is the default.

Output layout matches rosters.json so core/identity/jersey_identity_phase.py
consumes it unchanged:
    {"SNGS-021": {"left": {"field_players": [...], "goalkeepers": [...],
                           "non_numeric": []}, "right": {...}}, ...}

Usage:
    python scripts/extract_rosters_gsr.py \
        --gt_root data/SoccerNetGS/gamestate-2024/valid \
        --output_json datasets/jersey_tracking_v1/rosters_gsr.json \
        --scope game
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SIDES = ("left", "right")


def parse_sequence(labels_path):
    """Return (game_id, {side: {"field_players": set, "goalkeepers": set}})."""
    with open(labels_path) as f:
        data = json.load(f)

    game_id = data["info"]["game_id"]
    roster = {s: {"field_players": set(), "goalkeepers": set()} for s in SIDES}

    for ann in data["annotations"]:
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes") or {}
        role, team, jersey = attrs.get("role"), attrs.get("team"), attrs.get("jersey")
        if role not in ("player", "goalkeeper") or team not in SIDES or jersey is None:
            continue
        try:
            number = int(jersey)
        except (TypeError, ValueError):
            continue
        if not 0 < number <= 99:
            continue
        key = "goalkeepers" if role == "goalkeeper" else "field_players"
        roster[team][key].add(number)

    return game_id, roster


def merge(dst, src):
    for side in SIDES:
        for key in ("field_players", "goalkeepers"):
            dst[side][key] |= src[side][key]


def to_json(roster):
    return {
        side: {
            "field_players": sorted(roster[side]["field_players"]),
            "goalkeepers": sorted(roster[side]["goalkeepers"]),
            "non_numeric": [],
        }
        for side in SIDES
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_root", default="data/SoccerNetGS/gamestate-2024/valid")
    ap.add_argument("--output_json", default="datasets/jersey_tracking_v1/rosters_gsr.json")
    ap.add_argument("--scope", choices=("game", "clip"), default="game")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    seq_dirs = sorted(d for d in gt_root.iterdir()
                      if d.is_dir() and d.name.startswith("SNGS-"))

    per_seq, game_of = {}, {}
    for seq_dir in seq_dirs:
        labels = seq_dir / "Labels-GameState.json"
        if not labels.exists():
            print(f"  [WARN] {seq_dir.name}: sin Labels-GameState.json")
            continue
        game_id, roster = parse_sequence(labels)
        per_seq[seq_dir.name] = roster
        game_of[seq_dir.name] = game_id

    if args.scope == "game":
        per_game = defaultdict(
            lambda: {s: {"field_players": set(), "goalkeepers": set()} for s in SIDES})
        for seq, roster in per_seq.items():
            merge(per_game[game_of[seq]], roster)
        source = {seq: per_game[game_of[seq]] for seq in per_seq}
    else:
        source = per_seq

    out = {seq: to_json(roster) for seq, roster in source.items()}

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Alcance: {args.scope} | {len(out)} secuencias | {len(set(game_of.values()))} partidos")
    seen = set()
    for seq in sorted(out):
        gid = game_of[seq]
        if args.scope == "game":
            if gid in seen:
                continue
            seen.add(gid)
            label = f"partido {gid}"
        else:
            label = seq
        left = out[seq]["left"]
        right = out[seq]["right"]
        n_l = len(left["field_players"]) + len(left["goalkeepers"])
        n_r = len(right["field_players"]) + len(right["goalkeepers"])
        print(f"  {label:>12}: left={n_l:>2} nums {left['field_players']}"
              f" gk={left['goalkeepers']} | right={n_r:>2} nums {right['field_players']}"
              f" gk={right['goalkeepers']}")
    print(f"\nEscrito en {out_path}")


if __name__ == "__main__":
    main()
