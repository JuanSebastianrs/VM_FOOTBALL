"""
Multi-sequence end-to-end jersey evaluation.

Runs jersey_identity_phase + evaluate_jersey_e2e on every sequence that has
pipeline outputs (detections + team assignments + team audit), then aggregates
GT-level metrics across sequences. This replaces single-holdout E2E validation
(SNMOT-148 only) with a multi-match E2E gate.

Prerequisites per sequence (see scripts/run_e2e_prereqs.ps1):
  outputs/<seq>/<seq>_detections.json
  outputs/<seq>/<seq>_team_assignments.json
  outputs/<seq>/team_audit.json

Usage:
    python scripts/evaluate_jersey_e2e_multi.py \
        --sequences SNMOT-148 SNMOT-116 SNMOT-132 SNMOT-190 \
        --model_path runs/jersey_perframe_v2_mixed/best.pt \
        --tag v2_postproc --link_fragments --split_on_switch --reassign_conflicts
"""

import sys
import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_sequence(seq, args):
    seq_dir = Path(args.data_root) / seq
    out_dir = REPO / "outputs" / seq
    det = out_dir / f"{seq}_detections.json"
    team = out_dir / f"{seq}_team_assignments.json"
    audit = out_dir / "team_audit.json"
    for p in (det, team, audit):
        if not p.exists():
            raise FileNotFoundError(f"{seq}: missing prerequisite {p}")

    jersey_json = out_dir / f"{seq}_jersey_identity_{args.tag}.json"
    eval_json = out_dir / f"jersey_e2e_eval_{args.tag}.json"

    cmd = [
        sys.executable, str(REPO / "core/identity/jersey_identity_phase.py"),
        "--sequence_dir", str(seq_dir),
        "--detections_json", str(det),
        "--team_assignments_json", str(team),
        "--model_path", args.model_path,
        "--output_json", str(jersey_json),
        "--inference_mode", "temporal",
        "--device", args.device,
        "--team_mapping", str(audit),
        "--fusion_mode", args.fusion_mode,
        "--legibility_threshold", str(args.legibility_threshold),
        "--p1_threshold", str(args.p1_threshold),
        "--margin_threshold", str(args.margin_threshold),
    ]
    if args.roster_json:
        cmd += ["--roster_json", args.roster_json]
    if args.legibility_model:
        cmd += ["--legibility_model", args.legibility_model]
    if args.link_fragments:
        cmd += ["--link_fragments"]
    if args.split_on_switch:
        cmd += ["--split_on_switch"]
    if args.reassign_conflicts:
        cmd += ["--reassign_conflicts"]
    print(f"\n=== {seq}: jersey identity ({args.tag}) ===")
    subprocess.run(cmd, check=True)

    cmd_eval = [
        sys.executable, str(REPO / "scripts/evaluate_jersey_e2e.py"),
        "--sequence_dir", str(seq_dir),
        "--detections_json", str(det),
        "--jersey_json", str(jersey_json),
        "--team_mapping", str(audit),
        "--output_json", str(eval_json),
    ]
    print(f"=== {seq}: E2E eval ===")
    subprocess.run(cmd_eval, check=True)

    with open(eval_json) as f:
        return json.load(f)["summary"]


def main():
    parser = argparse.ArgumentParser(description="Multi-sequence E2E jersey evaluation")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--data_root", type=str,
                        default="data/tracking/SoccerNet/tracking/test/test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True,
                        help="Suffix for per-sequence artifacts and the aggregate report")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--roster_json", type=str,
                        default="datasets/jersey_tracking_v1/rosters.json")
    parser.add_argument("--legibility_model", type=str,
                        default="runs/jersey_legibility_v1/best.pt")
    parser.add_argument("--fusion_mode", type=str, default="arithmetic")
    parser.add_argument("--legibility_threshold", type=float, default=0.70)
    parser.add_argument("--p1_threshold", type=float, default=0.75)
    parser.add_argument("--margin_threshold", type=float, default=0.15)
    parser.add_argument("--link_fragments", action="store_true")
    parser.add_argument("--split_on_switch", action="store_true")
    parser.add_argument("--reassign_conflicts", action="store_true")
    parser.add_argument("--no_roster", action="store_true",
                        help="Disable roster mask (raw model capability)")
    args = parser.parse_args()
    if args.no_roster:
        args.roster_json = None

    per_seq = {}
    totals = {"gt_total": 0, "raw_correct": 0, "assigned_correct": 0,
              "locked_total": 0, "locked_correct": 0, "best_frag_raw": 0}
    for seq in args.sequences:
        s = run_sequence(seq, args)
        per_seq[seq] = s
        n = s["gt_total"]
        totals["gt_total"] += n
        totals["raw_correct"] += round(s["gt_raw_top1_matched_acc"] * s["unique_gt_matched"])
        totals["assigned_correct"] += round(s["gt_assigned_matched_acc"] * s["unique_gt_matched"])
        totals["locked_total"] += s["gt_locked_total"]
        totals["locked_correct"] += round(s["gt_locked_accuracy"] * s["gt_locked_total"])
        totals["best_frag_raw"] += round(s["gt_best_frag_raw_top1_acc"] * s["unique_gt_matched"])

    n = max(totals["gt_total"], 1)
    nl = max(totals["locked_total"], 1)
    aggregate = {
        "sequences": args.sequences,
        "config": {k: v for k, v in vars(args).items() if k != "sequences"},
        "gt_total": totals["gt_total"],
        "raw_top1_acc": round(totals["raw_correct"] / n, 4),
        "best_frag_raw_top1_acc": round(totals["best_frag_raw"] / n, 4),
        "assigned_acc": round(totals["assigned_correct"] / n, 4),
        "locked_total": totals["locked_total"],
        "locked_accuracy": round(totals["locked_correct"] / nl, 4),
        "false_lock_rate": round(1.0 - totals["locked_correct"] / nl, 4) if totals["locked_total"] else 0.0,
        "locked_coverage": round(totals["locked_total"] / n, 4),
        "per_sequence": per_seq,
    }

    out_path = REPO / "outputs" / "jersey_eval" / f"e2e_multi_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"AGGREGATE E2E ({len(args.sequences)} sequences, {totals['gt_total']} GT players)")
    print(f"{'=' * 60}")
    print(f"  raw top-1:        {aggregate['raw_top1_acc']:.1%}")
    print(f"  best-frag raw:    {aggregate['best_frag_raw_top1_acc']:.1%}")
    print(f"  assigned:         {aggregate['assigned_acc']:.1%}")
    print(f"  locked:           {aggregate['locked_total']} "
          f"(cov {aggregate['locked_coverage']:.1%})")
    print(f"  locked accuracy:  {aggregate['locked_accuracy']:.1%}")
    print(f"  false lock rate:  {aggregate['false_lock_rate']:.1%}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
