# scripts/scanning_v2/run_scanning_v2.py
"""
Script principal del Scanning V2.

Flujo: game_state -> eventos pase/recepcion -> ventanas -> head pose ->
head-turn -> export (+ render + annotation pack).

Ejemplo real (SNMOT-148, rutas del proyecto):
  python scripts/scanning_v2/run_scanning_v2.py \
    --config configs/scanning_v2.yaml \
    --video_id SNMOT-148 \
    --detections   outputs/SNMOT-148/SNMOT-148_detections.json \
    --trajectory   outputs/SNMOT-148/SNMOT-148_trajectory.json \
    --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
    --calibration  outputs/SNMOT-148/calibration_hinv.json \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --output_dir   outputs/SNMOT-148/scanning \
    --head-pose-backend sixdrepnet --render --build-annotation-pack

Sin --output_dir escribe en `outputs/<video_id>/scanning/` (layout canonico).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2 import ScanningPipelineV2          # noqa: E402
from core.scanning_v2.paths import scanning_dir           # noqa: E402


def get_args():
    p = argparse.ArgumentParser(description="Scanning V2 (head-turn antes de recepcion)")
    p.add_argument("--config", default="configs/scanning_v2.yaml")
    p.add_argument("--video_id", required=True)
    p.add_argument("--sequence_dir", default=None)
    p.add_argument("--video_path", default=None, help="(no usado si hay sequence_dir/img1)")
    p.add_argument("--detections", required=True)
    p.add_argument("--tracks", default=None, help="alias de detections si difiere")
    p.add_argument("--ball_trajectory", "--trajectory", dest="trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--event_ground_truth", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--head-pose-backend", dest="head_pose_backend",
                   choices=["sixdrepnet", "mediapipe", "yolo_pose_body"], default=None,
                   help="Pone este backend primero en backend_order (en memoria).")
    p.add_argument("--render", action="store_true")
    p.add_argument("--build-annotation-pack", dest="build_annotation_pack", action="store_true")
    p.add_argument("--no_windows", action="store_true", help="no guardar windows/ crops")
    return p.parse_args()


def main():
    args = get_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.head_pose_backend:
        order = list(config.get("head_pose", {}).get(
            "backend_order", ["sixdrepnet", "mediapipe", "yolo_pose_body"]))
        order = [args.head_pose_backend] + [b for b in order if b != args.head_pose_backend]
        config.setdefault("head_pose", {})["backend_order"] = order
        print(f"[v2] head_pose backend_order={order}")

    out_dir = args.output_dir or str(scanning_dir(
        config.get("outputs", {}).get("root", "outputs"), args.video_id))
    detections = args.detections or args.tracks

    pipe = ScanningPipelineV2(config)
    summary = pipe.run(
        video_id=args.video_id, detections=detections, trajectory=args.trajectory,
        team_assignments=args.team_assignments, calibration=args.calibration,
        sequence_dir=args.sequence_dir, event_ground_truth=args.event_ground_truth,
        out_dir=out_dir, save_windows=not args.no_windows)

    print("\n=== GAME STATE ===")
    print(json.dumps(summary["game_state"], indent=2, default=str))
    print("\n=== EVENTOS ===")
    print(f"  recepciones detectadas: {summary['receptions_detected']}")
    print(f"  candidatos rechazados:  {summary['rejected_total']} "
          f"(arbitro: {summary['rejected_referee']})")
    print(f"  eventos de scanning (head-turn): {summary['scanning_events_predicted']}")
    print(f"  outputs -> {summary['out_dir']}")

    if args.render:
        from scripts.scanning_v2.render_scanning_events import render_from_outputs
        render_from_outputs(config, args.video_id, out_dir, detections, args.trajectory,
                            args.team_assignments, args.calibration, args.sequence_dir)
    if args.build_annotation_pack:
        from scripts.scanning_v2.annotate_scanning_windows import build_annotation_pack
        build_annotation_pack(out_dir, args.video_id)


if __name__ == "__main__":
    main()
