# scripts/scanning/render_scanning_events.py
"""
Renderiza los N eventos de scanning mas fuertes como clips de auditoria.

Por cada evento (ordenados por scan_count y luego max_orientation_change_deg):
  - <tag>_video.mp4   : ventana [window_start .. frame_reception], bbox del
                        receptor, track_id, backend usado, flecha de orientacion
                        en IMAGEN, balon, marca SCAN/RECEPTION y frame exacto de
                        recepcion.
  - <tag>_minimap.mp4 : misma ventana en el minimapa, con la flecha y el cono de
                        orientacion en CANCHA (theta_visual_smooth_field).

Consume los productos ya exportados por run_scanning_pipeline.py:
  player_orientation.parquet/csv  +  scanning_events.parquet/csv

Ejemplo:
  python scripts/scanning/render_scanning_events.py \
      --video_id SNMOT-148 \
      --detections   outputs/SNMOT-148/SNMOT-148_detections.json \
      --trajectory   outputs/SNMOT-148/SNMOT-148_trajectory.json \
      --calibration  outputs/SNMOT-148/calibration_hinv.json \
      --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
      --orientation outputs/scanning/SNMOT-148/player_orientation.parquet \
      --scanning    outputs/scanning/SNMOT-148/scanning_events.parquet \
      --top 10
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning import load_config, load_sequence              # noqa: E402
from core.scanning.visualization import (                         # noqa: E402
    ScanningVisualizer, _arrow, _team_color, _col, _SCAN_COLOR, _BALL_COLOR)


def _read(path):
    return pd.read_parquet(path) if str(path).endswith("parquet") else pd.read_csv(path)


def render_event(viz, seq, by_frame, ev, out_dir, scale=11, margin=24):
    tid = int(ev["receiver_track_id"])
    f_rec = int(ev["frame_reception"])
    f0 = int(ev["window_start"]); f1 = f_rec
    tag = f"ev_t{tid}_f{f_rec}_sc{int(ev.get('scan_count', 0))}"

    w, h = seq.frame_size or (1920, 1080)
    vw = cv2.VideoWriter(str(out_dir / f"{tag}_video.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (w, h))
    base = viz._draw_pitch(scale, margin)
    mh, mw = base.shape[:2]
    mwr = cv2.VideoWriter(str(out_dir / f"{tag}_minimap.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (mw, mh))

    for fid in range(f0, f1 + 1):
        ipath = seq.image_paths.get(fid)
        img = cv2.imread(str(ipath)) if ipath else None
        pitch = base.copy()
        rows = [r for r in by_frame.get(fid, []) if int(r["track_id"]) == tid]
        is_rec = (fid == f_rec)
        for r in rows:
            color = _team_color(r.get("team_id"))
            # --- video (orientacion en imagen) ---
            if img is not None:
                x1, y1, x2, y2 = (int(r["bbox_x1"]), int(r["bbox_y1"]),
                                  int(r["bbox_x2"]), int(r["bbox_y2"]))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                th_i = _col(r, "theta_visual_smooth_img", "theta_visual_smooth")
                conf = r.get("orientation_confidence") or 0.0
                if th_i is not None:
                    _arrow(img, cx, cy, math.radians(float(th_i)), 30 + 40 * float(conf), color, 2)
                backend = _col(r, "keypoint_backend_used") or "?"
                lab = f"#{tid} {backend} c={float(conf):.2f}"
                lab += " RECEPTION" if is_rec else " SCAN"
                ring = (40, 255, 40) if is_rec else _SCAN_COLOR
                cv2.circle(img, (cx, cy), 26 if is_rec else 22, ring, 3 if is_rec else 2)
                cv2.putText(img, lab, (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
            # --- minimapa (orientacion en cancha) ---
            X, Y = r.get("x_field"), r.get("y_field")
            if X is not None and not (isinstance(X, float) and math.isnan(X)):
                mx, my = viz._to_minimap(float(X), float(Y), scale, margin)
                if 0 <= mx < mw and 0 <= my < mh:
                    cv2.circle(pitch, (mx, my), 6, color, -1)
                    th_f = _col(r, "theta_visual_smooth_field")
                    if th_f is not None:
                        thr = math.radians(float(th_f))
                        viz._draw_cone(pitch, mx, my, thr, color, scale)
                        _arrow(pitch, mx, my, thr, 20, color, 2)
                    ring = (40, 255, 40) if is_rec else _SCAN_COLOR
                    cv2.circle(pitch, (mx, my), 11 if is_rec else 9, ring, 2)
        # balon
        ball = seq.ball(fid)
        if img is not None and ball is not None and ball.valid:
            cv2.circle(img, (int(ball.x), int(ball.y)), 6, _BALL_COLOR, -1)
        bxy = seq.ball_field_xy(fid) if seq.homography.available else None
        if bxy is not None:
            mx, my = viz._to_minimap(bxy[0], bxy[1], scale, margin)
            if 0 <= mx < mw and 0 <= my < mh:
                cv2.circle(pitch, (mx, my), 4, _BALL_COLOR, -1)
        if img is not None:
            cv2.putText(img, f"frame {fid}  t-{(f_rec - fid) / seq.fps:.1f}s",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            vw.write(img)
        mwr.write(pitch)
    vw.release(); mwr.release()
    return tag


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", default=None)
    p.add_argument("--orientation", required=True)
    p.add_argument("--scanning", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    sc = config["scanning"]
    out_dir = Path(args.out_dir or os.path.join(config["data"]["output_dir"],
                                                args.video_id, "events"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = load_sequence(args.video_id, args.detections, args.trajectory,
                        args.team_assignments, args.calibration,
                        sequence_dir=args.sequence_dir, fps=sc.get("fps", 25.0))
    df = _read(args.orientation)
    by_frame = {}
    for row in df.to_dict("records"):
        by_frame.setdefault(int(row["frame_id"]), []).append(row)

    events = _read(args.scanning)
    events = events[events["scan_count"] > 0].copy()
    events = events.sort_values(["scan_count", "max_orientation_change_deg"],
                                ascending=False).head(args.top)
    if events.empty:
        print("[render] no hay eventos con scan_count>0.")
        return
    viz = ScanningVisualizer(sc)
    for ev in events.to_dict("records"):
        tag = render_event(viz, seq, by_frame, ev, out_dir)
        print(f"  clip -> {tag} (scan_count={int(ev['scan_count'])}, "
              f"max_change={ev['max_orientation_change_deg']:.1f} deg)")
    print(f"[render] {len(events)} eventos en {out_dir}")


if __name__ == "__main__":
    main()
