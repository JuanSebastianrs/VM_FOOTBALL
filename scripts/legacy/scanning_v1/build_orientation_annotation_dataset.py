# scripts/scanning/build_orientation_annotation_dataset.py
"""
Construye el dataset de anotacion de orientacion (semi-automatico).

Produce dos artefactos:

  1. features parquet  : features POR FRAME (keypoints + contexto) para cada
     (video_id, frame_id, track_id). Lo consume OrientationSequenceDataset para
     armar las secuencias temporales.
  2. labels CSV        : plantilla `orientation_labels.csv` con UNA fila por
     muestra a anotar. Viene PRE-LLENADA con la orientacion heuristica
     (orientation_angle_deg / orientation_bin_8) como semilla; el anotador
     corrige el angulo/bin y marca scan_label. Las columnas siguen el esquema
     del modulo (ver core/scanning/README.md).

Las muestras a anotar se eligen por stride + ventanas previas a cada recepcion,
para concentrar el esfuerzo de anotacion donde el scanning importa.

Ejemplo:
  python scripts/scanning/build_orientation_annotation_dataset.py \
      --video_id SNMOT-148 \
      --detections outputs/SNMOT-148/SNMOT-148_detections.json \
      --trajectory outputs/SNMOT-148/SNMOT-148_trajectory.json \
      --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
      --calibration outputs/SNMOT-148/calibration_hinv.json \
      --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
      --stride 5 --max_frames 300
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

from core.scanning import load_config, load_sequence                    # noqa: E402
from core.scanning.circular import angle_to_bin                         # noqa: E402
from core.scanning.keypoint_extractor import KeypointExtractor          # noqa: E402
from core.scanning.models.orientation_dataset import build_feature_dict, FEATURE_COLUMNS  # noqa: E402
from core.scanning.orientation_estimator import OrientationEstimator    # noqa: E402
from core.scanning.player_cropper import PlayerCropper                  # noqa: E402
from core.scanning.reception_detector import ReceptionDetector          # noqa: E402

LABEL_COLUMNS = [
    "sample_id", "video_id", "frame_id", "track_id", "crop_path",
    "context_frame_path", "x_field", "y_field", "ball_x", "ball_y",
    "time_to_reception", "orientation_bin_8", "orientation_angle_deg",
    "scan_label", "visibility", "confidence", "annotator_notes",
]


def _vis_bucket(v: float) -> str:
    return "high" if v >= 0.6 else ("medium" if v >= 0.35 else "low")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default="data/annotations")
    p.add_argument("--features_out", default=None)
    p.add_argument("--stride", type=int, default=5, help="muestrear 1 de cada N frames")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--min_head_vis", type=float, default=0.3)
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    sc = config["scanning"]
    fps = sc.get("fps", 25.0)

    seq = load_sequence(args.video_id, args.detections, args.trajectory,
                        args.team_assignments, args.calibration,
                        sequence_dir=args.sequence_dir, fps=fps)
    if not seq.image_paths:
        raise SystemExit("Se requieren imagenes (sequence_dir/img1) para construir crops.")

    crop_dir = Path(config["data"]["crop_dir"]) / args.video_id
    cropper = PlayerCropper(expand_ratio=sc.get("crop_expand_ratio", 0.25),
                            crop_size=tuple(sc.get("crop_size", (384, 768))),
                            min_crop_height=sc.get("min_crop_height", 64),
                            low_quality_height=sc.get("low_quality_crop_height", 110),
                            save_dir=crop_dir)
    extractor = KeypointExtractor(sc)
    estimator = OrientationEstimator(sc)

    # recepciones -> ventanas de anotacion preferente
    receptions = ReceptionDetector(sc).detect(seq)
    window_frames = {}      # (track_id, frame) -> time_to_reception
    before = sc.get("scan_window_seconds_before", 3.0)
    after = sc.get("scan_window_seconds_after", 0.2)
    for ev in receptions:
        f0 = int(ev.frame_reception - before * fps)
        f1 = int(ev.frame_reception - after * fps)
        for f in range(f0, f1 + 1):
            window_frames[(ev.receiver_track_id, f)] = (ev.frame_reception - f) / fps

    frame_ids = seq.frame_ids[:args.max_frames] if args.max_frames else seq.frame_ids
    feat_rows = []
    label_rows = []
    prev_center = {}
    sid = 0
    fsize = seq.frame_size or (1920, 1080)

    for n, fid in enumerate(frame_ids):
        ipath = seq.image_paths.get(fid)
        img = cv2.imread(str(ipath)) if ipath else None
        if img is None:
            continue
        ball = seq.ball(fid)
        ball_img = (ball.x, ball.y) if (ball and ball.valid) else None
        ball_field = seq.ball_field_xy(fid) if seq.homography.available else None

        for p in seq.players(fid):
            tid = p.track_id
            center = p.center
            vel = None
            if tid in prev_center:
                vel = (center[0] - prev_center[tid][0], center[1] - prev_center[tid][1])
            prev_center[tid] = center

            crop = cropper.crop(img, p.bbox, fid, tid, save_debug=False)
            pose = extractor.extract(crop.crop) if crop.valid_crop else {"pose_valid": False, "landmarks": {}}
            bbox_exp = crop.bbox_expanded if crop.valid_crop else p.bbox

            ori = estimator.estimate(
                frame_id=fid, track_id=tid, pose=pose, bbox_exp=bbox_exp,
                velocity=vel, ball_xy=ball_img, player_xy=center,
                homography=seq.homography)

            field_xy = seq.player_field_xy(fid, tid)
            dist_ball = None
            if field_xy is not None and ball_field is not None:
                dist_ball = float(np.hypot(field_xy[0] - ball_field[0],
                                           field_xy[1] - ball_field[1]))
            ttr = window_frames.get((tid, fid), 0.0)

            feat = build_feature_dict(
                pose=pose, bbox=p.bbox, velocity=vel, run_angle=ori.run_angle,
                ball_relative_angle=ori.ball_relative_angle, field_xy=field_xy,
                dist_ball_m=dist_ball, time_to_reception=ttr, frame_size=fsize)
            feat.update(video_id=args.video_id, frame_id=fid, track_id=tid)
            feat_rows.append(feat)

            # muestrear para anotacion
            in_window = (tid, fid) in window_frames
            sample = in_window or (n % max(1, args.stride) == 0)
            if not sample or len(label_rows) >= args.max_samples:
                continue
            if not crop.valid_crop or pose.get("head_visibility", 0.0) < args.min_head_vis:
                continue
            crop_path = str(crop_dir / f"f{fid:06d}_t{tid:04d}.jpg")
            cv2.imwrite(crop_path, crop.crop)
            seed_deg = math.degrees(ori.theta_visual_img) if ori.theta_visual_img is not None else 0.0
            label_rows.append({
                "sample_id": sid, "video_id": args.video_id, "frame_id": fid,
                "track_id": tid, "crop_path": crop_path,
                "context_frame_path": str(ipath),
                "x_field": field_xy[0] if field_xy else None,
                "y_field": field_xy[1] if field_xy else None,
                "ball_x": ball_img[0] if ball_img else None,
                "ball_y": ball_img[1] if ball_img else None,
                "time_to_reception": ttr,
                "orientation_bin_8": angle_to_bin(ori.theta_visual_img, 8) if ori.theta_visual_img is not None else None,
                "orientation_angle_deg": round(seed_deg, 2),
                "scan_label": 0,
                "visibility": _vis_bucket(pose.get("head_visibility", 0.0)),
                "confidence": round(float(ori.orientation_confidence), 3),
                "annotator_notes": "SEED-heuristic; revisar angulo/bin/scan",
            })
            sid += 1
        if n % 50 == 0:
            print(f"  frame {fid} ({n + 1}/{len(frame_ids)}) feats={len(feat_rows)} labels={len(label_rows)}")

    # escribir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_out = Path(args.features_out) if args.features_out else \
        out_dir / f"{args.video_id}_orientation_features.parquet"
    pd.DataFrame(feat_rows)[["video_id", "frame_id", "track_id"] + FEATURE_COLUMNS]\
        .to_parquet(features_out, index=False)

    labels_path = out_dir / "orientation_labels.csv"
    df_labels = pd.DataFrame(label_rows)[LABEL_COLUMNS]
    if labels_path.exists():
        prev = pd.read_csv(labels_path)
        prev = prev[prev["video_id"] != args.video_id]   # reemplazar este video
        df_labels = pd.concat([prev, df_labels], ignore_index=True)
    df_labels.to_csv(labels_path, index=False)

    print(f"\nFeatures  -> {features_out}  ({len(feat_rows)} filas)")
    print(f"Labels    -> {labels_path}  ({len(label_rows)} muestras nuevas)")
    print("Plantilla pre-llenada con orientacion heuristica (SEED). "
          "Corrige orientation_angle_deg / orientation_bin_8 / scan_label.")


if __name__ == "__main__":
    main()
