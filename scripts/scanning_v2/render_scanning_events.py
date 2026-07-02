# scripts/scanning_v2/render_scanning_events.py
"""
Renderiza clips por evento (video + minimap) a partir de los outputs ya
exportados del scanning V2.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning.data_io import load_sequence                      # noqa: E402
from core.scanning_v2.paths import resolve_scanning_dir              # noqa: E402
from core.scanning_v2.visualization import ScanningV2Visualizer      # noqa: E402


def render_from_outputs(config, video_id, out_dir, detections, trajectory,
                        team_assignments, calibration, sequence_dir,
                        model_predictions=None, clips_subdir="event_clips"):
    out = Path(out_dir)
    events = pd.read_parquet(out / "pass_reception_events.parquet")
    if events.empty:
        print("[render] sin eventos que renderizar.")
        return
    head_pose = pd.read_parquet(out / "head_pose.parquet")
    scanning = pd.read_parquet(out / "scanning_events.parquet")
    fps = float(config.get("video", {}).get("fps", 25.0))
    seq = load_sequence(video_id, detections, trajectory, team_assignments,
                        calibration, sequence_dir=sequence_dir, fps=fps)
    gs = pd.read_parquet(out / "game_state.parquet")
    viz = ScanningV2Visualizer({**config, "fps": fps})
    clips_dir = out / clips_subdir
    scan_by_id = {r["event_id"]: r for r in scanning.to_dict("records")}

    # opcional: sobreescribir la etiqueta heuristica con la del MODELO entrenado
    model_by_id = {}
    if model_predictions and Path(model_predictions).exists():
        preds = pd.read_parquet(model_predictions)
        if "video_id" in preds.columns:
            preds = preds[preds["video_id"].astype(str) == str(video_id)]
        model_by_id = {r["event_id"]: r for r in preds.to_dict("records")}
        print(f"[render] usando predicciones del modelo para "
              f"{len(model_by_id)} eventos de {video_id}")

    n = 0
    for ev in events.to_dict("records"):
        scan_row = dict(scan_by_id.get(
            ev["event_id"], {"window_start": ev["frame_reception"] - int(3 * fps),
                             "scan_label_pred": 0, "source": ev.get("source")}))
        mp = model_by_id.get(ev["event_id"])
        if mp is not None:
            scan_row["scan_label_pred"] = int(mp["scan_label_model"])
            scan_row["source"] = (f"model p={float(mp['scan_probability_model']):.2f}")
        viz.render_event(seq, gs, head_pose, scan_row, ev, clips_dir)
        n += 1
        print(f"  clip {ev['event_id']} (recv #{ev['receiver_track_id']} "
              f"{ev.get('receiver_role')}, scan={scan_row.get('scan_label_pred')}"
              f"{'' if mp is None else ' [model]'})")
    print(f"[render] {n} clips -> {clips_dir}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/scanning_v2.yaml")
    p.add_argument("--video_id", required=True)
    p.add_argument("--output_dir", default=None,
                   help="default: outputs/<video_id>/scanning")
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", default=None)
    p.add_argument("--model_predictions", default=None,
                   help="scanning_model_predictions.parquet: renderiza la "
                        "etiqueta del MODELO en lugar de la heuristica")
    p.add_argument("--clips_subdir", default="event_clips")
    return p.parse_args()


def main():
    a = get_args()
    with open(a.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out_dir = a.output_dir or str(resolve_scanning_dir(
        config.get("outputs", {}).get("root", "outputs"), a.video_id))
    render_from_outputs(config, a.video_id, out_dir, a.detections, a.trajectory,
                        a.team_assignments, a.calibration, a.sequence_dir,
                        model_predictions=a.model_predictions,
                        clips_subdir=a.clips_subdir)


if __name__ == "__main__":
    main()
