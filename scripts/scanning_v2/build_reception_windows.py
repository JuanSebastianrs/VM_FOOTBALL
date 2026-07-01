# scripts/scanning_v2/build_reception_windows.py
"""Construye game_state + eventos + ventanas (metadata) sin head pose (FASES 1-3)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.events import PassReceptionDetector, load_event_ground_truth   # noqa: E402
from core.game_state import build_game_state                             # noqa: E402
from core.scanning.data_io import load_sequence                          # noqa: E402
from core.scanning_v2 import exporter                                    # noqa: E402
from core.scanning_v2.reception_window_extractor import ReceptionWindowExtractor  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/scanning_v2.yaml")
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", default=None)
    p.add_argument("--event_ground_truth", default=None)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    a = get_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    fps = float(cfg.get("video", {}).get("fps", 25.0))
    gs, report = build_game_state(a.video_id, a.detections, a.trajectory,
                                  a.team_assignments, a.calibration,
                                  sequence_dir=a.sequence_dir, fps=fps)
    exporter.export_game_state(gs, a.output_dir)
    seq = load_sequence(a.video_id, a.detections, a.trajectory, a.team_assignments,
                        a.calibration, sequence_dir=a.sequence_dir, fps=fps)
    attached = {fid: int(b.attached_player_id) for fid, b in seq.ball_by_frame.items()
                if b.attached_player_id is not None}
    ev_cfg = dict(cfg.get("events", {})); ev_cfg["fps"] = fps
    ev_cfg["has_homography"] = seq.homography.available
    gt = load_event_ground_truth(a.event_ground_truth, a.video_id)
    events, rejected = PassReceptionDetector(ev_cfg).detect(gs, a.video_id, attached, gt)
    exporter.export_events(events, a.output_dir)
    exporter.export_rejected(rejected, a.output_dir)
    wcfg = dict(cfg.get("windows", {})); wcfg["fps"] = fps
    specs = ReceptionWindowExtractor(wcfg).extract(events, gs)
    from pathlib import Path
    for s in specs:
        ReceptionWindowExtractor.write_metadata(Path(a.output_dir), s)
    print(json.dumps(report, indent=2, default=str))
    print(f"events={len(events)} rejected={len(rejected)} windows={len(specs)}")


if __name__ == "__main__":
    main()
