"""
Extract team rosters from gameinfo.ini across all SoccerNet sequences.
Outputs rosters.json with field_players and goalkeepers separated per team side.

Usage:
    python scripts/extract_rosters.py \
        --root_dir data/tracking/SoccerNet/tracking \
        --output_json datasets/jersey_tracking_v1/rosters.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def parse_roster(gameinfo_path):
    """
    Parse gameinfo.ini and extract per-side rosters with GK separated.

    Returns:
        dict with keys "left" and "right", each containing:
            - "field_players": list of int jersey numbers
            - "goalkeepers": list of int jersey numbers
            - "non_numeric": list of non-numeric jersey labels (for audit)
    """
    roster = {
        "left": {"field_players": [], "goalkeepers": [], "non_numeric": []},
        "right": {"field_players": [], "goalkeepers": [], "non_numeric": []},
    }

    content = gameinfo_path.read_text(encoding="utf-8", errors="replace")
    for line in content.split("\n"):
        line = line.strip()
        if not line.startswith("trackletID_") or "=" not in line:
            continue
        _, val = line.split("=", 1)
        val = val.strip()
        if ";" not in val:
            continue
        team_desc, jersey_raw = val.rsplit(";", 1)
        team_desc = team_desc.strip().lower()
        jersey_raw = jersey_raw.strip()

        # Only players and goalkeepers (not referees, ball, cameras)
        if not ("player" in team_desc or "goalkeeper" in team_desc):
            continue

        # Determine side
        if "left" in team_desc:
            side = "left"
        elif "right" in team_desc:
            side = "right"
        else:
            continue

        is_gk = "goalkeeper" in team_desc

        try:
            jersey = int(jersey_raw)
            if jersey <= 0 or jersey > 99:
                roster[side]["non_numeric"].append(jersey_raw)
                continue
            if is_gk:
                if jersey not in roster[side]["goalkeepers"]:
                    roster[side]["goalkeepers"].append(jersey)
            else:
                if jersey not in roster[side]["field_players"]:
                    roster[side]["field_players"].append(jersey)
        except ValueError:
            roster[side]["non_numeric"].append(jersey_raw)

    return roster


def main():
    parser = argparse.ArgumentParser(description="Extract team rosters from SoccerNet gameinfo.ini")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root dir of SoccerNet tracking (contains train/test/challenge)")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output path for rosters.json")
    args = parser.parse_args()

    root = Path(args.root_dir)
    all_rosters = {}
    total_seqs = 0
    seqs_with_roster = 0

    for seq_dir in sorted(root.rglob("SNMOT-*")):
        if not seq_dir.is_dir():
            continue
        total_seqs += 1
        gi = seq_dir / "gameinfo.ini"
        if not gi.exists():
            continue

        roster = parse_roster(gi)
        seq_name = seq_dir.name

        left_total = len(roster["left"]["field_players"]) + len(roster["left"]["goalkeepers"])
        right_total = len(roster["right"]["field_players"]) + len(roster["right"]["goalkeepers"])

        if left_total == 0 and right_total == 0:
            continue

        seqs_with_roster += 1
        all_rosters[seq_name] = roster

        # Audit warnings
        for side in ("left", "right"):
            total = len(roster[side]["field_players"]) + len(roster[side]["goalkeepers"])
            if total < 5:
                print(f"  [WARN] {seq_name} {side}: only {total} numeric players")
            if total > 18:
                print(f"  [WARN] {seq_name} {side}: {total} players (suspiciously high)")

    # Save
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_rosters, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Roster extraction complete")
    print(f"  Total sequences: {total_seqs}")
    print(f"  Sequences with roster: {seqs_with_roster}")
    print(f"  Sequences without gameinfo.ini: {total_seqs - seqs_with_roster}")
    print(f"  Output: {out_path}")

    # Summary stats
    gk_count = sum(
        len(r[side]["goalkeepers"])
        for r in all_rosters.values()
        for side in ("left", "right")
    )
    fp_count = sum(
        len(r[side]["field_players"])
        for r in all_rosters.values()
        for side in ("left", "right")
    )
    nn_count = sum(
        len(r[side]["non_numeric"])
        for r in all_rosters.values()
        for side in ("left", "right")
    )
    print(f"  Total field players (numeric): {fp_count}")
    print(f"  Total goalkeepers (numeric): {gk_count}")
    print(f"  Total non-numeric labels: {nn_count}")


if __name__ == "__main__":
    main()
