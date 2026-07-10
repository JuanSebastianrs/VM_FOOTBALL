"""
Run the official GS-HOTA evaluation (sn-trackeval fork) on our converted
predictions, restricted to the sequences we actually processed, with optional
ablations of the identity attributes.

Because GS-HOTA zeroes the similarity when role/team/jersey disagree, we report
the official setting (all on) plus ablations that switch off jersey and team, to
show how much each identity link costs us.

Usage:
    $env:PYTHONPATH="."
    python src/gsr_eval.py                      # official (all attributes on)
    python src/gsr_eval.py --no-jersey          # ignore jersey numbers
    python src/gsr_eval.py --no-jersey --no-team --no-role   # localization only
"""

import os
import sys
import glob
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "external", "sn-trackeval"))

import trackeval  # noqa: E402

GT_FOLDER = os.path.join(BASE, "data", "SoccerNetGS", "gamestate-2024")
TRACKERS_FOLDER = os.path.join(BASE, "data", "gsr_trackers")
SPLIT = "valid"
TRACKER = "TacticalVision"


def processed_sequences():
    data_dir = os.path.join(TRACKERS_FOLDER, f"SoccerNetGS-{SPLIT}", TRACKER, "data")
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(data_dir, "*.json")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-jersey", action="store_true")
    ap.add_argument("--no-team", action="store_true")
    ap.add_argument("--no-role", action="store_true")
    ap.add_argument("--sequences", nargs="*", default=None)
    args = ap.parse_args()

    seqs = args.sequences or processed_sequences()
    if not seqs:
        sys.exit("No converted predictions found. Run gsr_convert_and_eval.py first.")
    print(f"Evaluating {len(seqs)} sequences: {seqs}")

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["USE_PARALLEL"] = False

    dcfg = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    dcfg.update({
        "GT_FOLDER": GT_FOLDER,
        "TRACKERS_FOLDER": TRACKERS_FOLDER,
        "SPLIT_TO_EVAL": SPLIT,
        "TRACKERS_TO_EVAL": [TRACKER],
        "SEQ_INFO": {s: None for s in seqs},
        "USE_JERSEY_NUMBERS": not args.no_jersey,
        "USE_TEAMS": not args.no_team,
        "USE_ROLES": not args.no_role,
        "PRINT_CONFIG": False,
    })

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.SoccerNetGS(dcfg)
    metrics = [trackeval.metrics.HOTA({"THRESHOLD": 0.5}),
               trackeval.metrics.Identity({"THRESHOLD": 0.5})]
    res, _ = evaluator.evaluate([dataset], metrics)

    ped = res["SoccerNetGS"][TRACKER]["COMBINED_SEQ"]["person"]
    import numpy as np
    hota = ped["HOTA"]
    print("\n" + "=" * 50)
    tag = f"jersey={not args.no_jersey} team={not args.no_team} role={not args.no_role}"
    print(f"  GS-HOTA RESULTS ({tag})")
    print("=" * 50)
    print(f"  GS-HOTA (mean) : {np.mean(hota['HOTA']) * 100:.2f}%")
    print(f"  DetA           : {np.mean(hota['DetA']) * 100:.2f}%")
    print(f"  AssA           : {np.mean(hota['AssA']) * 100:.2f}%")
    print(f"  IDF1           : {ped['Identity']['IDF1'] * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
