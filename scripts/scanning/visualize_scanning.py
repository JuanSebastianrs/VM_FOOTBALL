# scripts/scanning/visualize_scanning.py
"""
Genera las visualizaciones de scanning a partir de resultados ya exportados.

Lee player_orientation.* y scanning_events.* y renderiza:
  - debug_video.mp4       (overlay sobre el video)
  - minimap_scanning.mp4  (overlay sobre el minimapa)

Ejemplo:
  python scripts/scanning/visualize_scanning.py \
      --video_id SNMOT-148 \
      --detections outputs/SNMOT-148/SNMOT-148_detections.json \
      --trajectory outputs/SNMOT-148/SNMOT-148_trajectory.json \
      --calibration outputs/SNMOT-148/calibration_hinv.json \
      --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
      --orientation outputs/scanning/SNMOT-148/player_orientation.parquet \
      --scanning outputs/scanning/SNMOT-148/scanning_events.parquet
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning import load_config, load_sequence              # noqa: E402
from core.scanning.visualization import ScanningVisualizer        # noqa: E402


def _read(path):
    return pd.read_parquet(path) if str(path).endswith("parquet") else pd.read_csv(path)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", default=None)
    p.add_argument("--orientation", required=True)
    p.add_argument("--scanning", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--no_minimap", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    sc = config["scanning"]
    out_dir = args.output_dir or os.path.join(config["data"]["output_dir"], args.video_id)
    os.makedirs(out_dir, exist_ok=True)

    seq = load_sequence(args.video_id, args.detections, args.trajectory,
                        args.team_assignments, args.calibration,
                        sequence_dir=args.sequence_dir, fps=sc.get("fps", 25.0))
    df = _read(args.orientation)
    events = _read(args.scanning).to_dict("records") if args.scanning else []
    viz = ScanningVisualizer(sc)

    if not args.no_video and seq.image_paths:
        vp = viz.render_video(seq, df, events, os.path.join(out_dir, "debug_video.mp4"),
                              max_frames=args.max_frames)
        print(f"video: {vp}")
    elif not args.no_video:
        print("[aviso] sin imagenes -> se omite el overlay de video.")
    if not args.no_minimap:
        mp = viz.render_minimap(seq, df, events,
                                os.path.join(out_dir, "minimap_scanning.mp4"),
                                max_frames=args.max_frames)
        print(f"minimap: {mp}")


if __name__ == "__main__":
    main()
