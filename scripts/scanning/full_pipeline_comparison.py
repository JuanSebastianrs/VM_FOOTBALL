# scripts/scanning/full_pipeline_comparison.py
"""
Comparacion de backends de keypoints en el PIPELINE COMPLETO (no solo crops).

Corre `ScanningPipeline` end-to-end sobre una secuencia para cada backend
(`yolo_pose`, `mediapipe`, `hybrid`) y agrega metricas a nivel de pipeline:

  - total frames procesados, total player-frame samples
  - pose_valid_rate, head_angle_rate, torso_angle_rate
  - distribucion de theta_source (head/torso/movement/previous/invalid)
  - mean_orientation_confidence
  - angular_jitter_deg (cambio medio frame-a-frame de theta_visual_img por track)
  - receptions_detected, scanning_events_detected, scan_count medio,
    max_orientation_change_deg medio
  - tiempo total y ms/frame
  - distribucion de backends realmente usados (keypoint_backend_used)
  - distribucion por bbox_height (small/medium/large)

Genera `outputs/scanning/backend_benchmark/full_pipeline_comparison.md`.

Ejemplo:
  python scripts/scanning/full_pipeline_comparison.py \
      --video_id SNMOT-148 \
      --detections   outputs/SNMOT-148/SNMOT-148_detections.json \
      --trajectory   outputs/SNMOT-148/SNMOT-148_trajectory.json \
      --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
      --calibration  outputs/SNMOT-148/calibration_hinv.json \
      --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning import load_config, load_sequence              # noqa: E402
from core.scanning.circular import circular_delta                 # noqa: E402
from core.scanning.pipeline import ScanningPipeline               # noqa: E402

BACKENDS = ["yolo_pose", "mediapipe", "hybrid"]


def _bucket(h: float) -> str:
    return "small" if h < 120 else ("medium" if h < 200 else "large")


def aggregate(records, receptions, scans, elapsed_s, n_frames):
    n = len(records)
    valid = [r for r in records if r["pose_valid"]]

    def rate(pred):
        return sum(1 for r in records if pred(r)) / n if n else 0.0

    src = Counter(r["theta_source"] for r in records)
    src_pct = {k: 100.0 * src.get(k, 0) / n for k in
               ["head", "torso", "movement", "previous", "invalid"]} if n else {}

    confs = [r["orientation_confidence"] for r in records
             if r["theta_visual_img"] is not None]

    # jitter: cambio medio frame-a-frame de theta_visual_img por track (grados)
    by_track = defaultdict(list)
    for r in records:
        if r["theta_visual_img"] is not None:
            by_track[r["track_id"]].append((r["frame_id"], r["theta_visual_img"]))
    jit = []
    for tid, seq in by_track.items():
        seq.sort()
        for (f0, a0), (f1, a1) in zip(seq[:-1], seq[1:]):
            if f1 == f0 + 1:   # solo frames consecutivos
                jit.append(abs(circular_delta(math.radians(a1), math.radians(a0))))
    jitter_deg = math.degrees(float(np.mean(jit))) if jit else float("nan")

    backend_used = Counter(r.get("keypoint_backend_used", "") for r in records)
    buckets = Counter(_bucket(r["bbox_y2"] - r["bbox_y1"]) for r in records)

    # routing real por backend_attempted (cuantos crops fueron a cada motor)
    att = [r.get("keypoint_backend_attempted", "") for r in records]
    crops_to_mp = sum(1 for a in att if "mediapipe" in a)
    crops_to_yolo = sum(1 for a in att if "yolo_pose" in a)
    fallbacks = sum(1 for a in att if "+" in a)   # mediapipe+yolo_pose => hubo fallback

    scan_counts = [m.scan_count for m in scans]
    max_changes = [m.max_orientation_change_deg for m in scans]

    return {
        "total_frames": n_frames,
        "player_frame_samples": n,
        "pose_valid_rate": rate(lambda r: r["pose_valid"]),
        "head_angle_rate": rate(lambda r: r["head_angle"] is not None),
        "torso_valid_rate": rate(lambda r: r["shoulder_angle"] is not None),
        "theta_source_pct": src_pct,
        "mean_orientation_confidence": float(np.mean(confs)) if confs else 0.0,
        "angular_jitter_deg": jitter_deg,
        "receptions_detected": len(receptions),
        "scanning_events_detected": int(sum(1 for m in scans if m.scan_count > 0)),
        "mean_scan_count": float(np.mean(scan_counts)) if scan_counts else 0.0,
        "mean_max_orientation_change_deg": float(np.mean(max_changes)) if max_changes else 0.0,
        "elapsed_s": elapsed_s,
        "ms_per_frame": 1000.0 * elapsed_s / n_frames if n_frames else 0.0,
        "crops_to_mediapipe": crops_to_mp,
        "crops_to_yolo": crops_to_yolo,
        "fallbacks_mp_to_yolo": fallbacks,
        "backend_used_dist": dict(backend_used),
        "bbox_height_dist": dict(buckets),
    }


def _fmt(v, p=3):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{p}f}"
    return str(v)


def write_md(out_path, video_id, metrics, device_note, cols=BACKENDS):

    def row(label, key, p=3, pct=False):
        cells = []
        for bk in cols:
            v = metrics[bk][key]
            if pct:
                v = v * 100
            cells.append(_fmt(v, p))
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        f"# Comparacion de pipeline completo por backend — {video_id}",
        "",
        "> Orientacion visual **aproximada** basada en keypoints. **No es gaze real.** "
        "Cada backend corre el `ScanningPipeline` end-to-end sobre la MISMA secuencia.",
        "",
        f"- Dispositivo: {device_note}",
        "",
        "| Metrica | " + " | ".join(cols) + " |",
        "|---|" + "|".join(["---"] * len(cols)) + "|",
        row("total_frames", "total_frames", p=0),
        row("player_frame_samples", "player_frame_samples", p=0),
        row("pose_valid_rate", "pose_valid_rate"),
        row("head_angle_rate", "head_angle_rate"),
        row("torso_valid_rate", "torso_valid_rate"),
        row("mean_orientation_confidence", "mean_orientation_confidence"),
        row("angular_jitter_deg", "angular_jitter_deg", p=2),
        row("receptions_detected", "receptions_detected", p=0),
        row("scanning_events_detected", "scanning_events_detected", p=0),
        row("mean_scan_count", "mean_scan_count", p=2),
        row("mean_max_orientation_change_deg", "mean_max_orientation_change_deg", p=2),
        row("crops_to_mediapipe", "crops_to_mediapipe", p=0),
        row("crops_to_yolo", "crops_to_yolo", p=0),
        row("fallbacks_mp_to_yolo", "fallbacks_mp_to_yolo", p=0),
        row("elapsed_s", "elapsed_s", p=1),
        row("ms_per_frame", "ms_per_frame", p=1),
        "",
        "## theta_source (%)",
        "",
        "| Fuente | " + " | ".join(cols) + " |",
        "|---|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for s in ["head", "torso", "movement", "previous", "invalid"]:
        cells = [_fmt(metrics[bk]["theta_source_pct"].get(s, 0.0), 1) for bk in cols]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")

    lines += ["", "## Backend realmente usado por player-frame", "",
              "| Backend usado | " + " | ".join(cols) + " |",
              "|---|" + "|".join(["---"] * len(cols)) + "|"]
    keys = sorted({k for bk in cols for k in metrics[bk]["backend_used_dist"]})
    for k in keys:
        cells = [str(metrics[bk]["backend_used_dist"].get(k, 0)) for bk in cols]
        lines.append(f"| {k or '(vacio)'} | " + " | ".join(cells) + " |")

    lines += ["", "## Distribucion por bbox_height", "",
              "| Bucket | " + " | ".join(cols) + " |",
              "|---|" + "|".join(["---"] * len(cols)) + "|"]
    for bk_size in ["small", "medium", "large"]:
        cells = [str(metrics[bk]["bbox_height_dist"].get(bk_size, 0)) for bk in cols]
        lines.append(f"| {bk_size} | " + " | ".join(cells) + " |")

    lines += ["", "## Lectura", "",
              "- `hybrid` debe igualar o mejorar la cobertura (`pose_valid_rate`) de "
              "`yolo_pose` y acercarse a la limpieza de `mediapipe` en crops grandes.",
              "- `angular_jitter_deg` menor = orientacion mas estable.",
              "- El conteo de recepciones es independiente del backend (depende del balon).",
              ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default="outputs/scanning/backend_benchmark")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--backends", nargs="+", default=BACKENDS)
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    sc = config["scanning"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = load_sequence(args.video_id, args.detections, args.trajectory,
                        args.team_assignments, args.calibration,
                        sequence_dir=args.sequence_dir, fps=sc.get("fps", 25.0))
    if not seq.image_paths:
        raise SystemExit("Se requieren imagenes (sequence_dir/img1).")
    n_frames = len(seq.frame_ids[:args.max_frames] if args.max_frames else seq.frame_ids)

    metrics = {}
    for bk in args.backends:
        print(f"[full-cmp] backend={bk} ...")
        cfg = copy.deepcopy(config)
        cfg["scanning"]["keypoint_backend"] = bk
        pipe = ScanningPipeline(cfg, enable_pose=True)
        t0 = time.perf_counter()
        records, receptions, scans = pipe.run(seq, max_frames=args.max_frames, verbose=False)
        elapsed = time.perf_counter() - t0
        if pipe.extractor is not None:
            pipe.extractor.close()
        metrics[bk] = aggregate(records, receptions, scans, elapsed, n_frames)
        m = metrics[bk]
        print(f"  pose_valid={m['pose_valid_rate']:.3f} jitter={m['angular_jitter_deg']:.2f} "
              f"recep={m['receptions_detected']} scan_ev={m['scanning_events_detected']} "
              f"ms/frame={m['ms_per_frame']:.1f} used={m['backend_used_dist']}")

    with open(out_dir / "full_pipeline_comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    import torch
    device_note = ("YOLO-Pose en GPU (CUDA), MediaPipe en CPU (XNNPACK) — el tiempo "
                   "no es comparable 1:1") if torch.cuda.is_available() else "todo en CPU"
    write_md(out_dir / "full_pipeline_comparison.md",
             args.video_id, metrics, device_note, cols=args.backends)
    print(f"[full-cmp] listo -> {out_dir / 'full_pipeline_comparison.md'}")


if __name__ == "__main__":
    main()
