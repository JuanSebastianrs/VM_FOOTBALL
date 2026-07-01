# scripts/scanning_v2/build_game_state.py
"""Construye y exporta solo el game_state.parquet (FASE 1)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.game_state import build_game_state                 # noqa: E402
from core.scanning_v2 import exporter                        # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--fps", type=float, default=25.0)
    return p.parse_args()


def main():
    a = get_args()
    gs, report = build_game_state(a.video_id, a.detections, a.trajectory,
                                  a.team_assignments, a.calibration,
                                  sequence_dir=a.sequence_dir, fps=a.fps)
    path = exporter.export_game_state(gs, a.output_dir)
    print(json.dumps(report, indent=2, default=str))
    print(f"game_state -> {path}")


if __name__ == "__main__":
    main()
