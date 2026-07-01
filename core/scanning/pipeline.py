# core/scanning/pipeline.py
"""
Orquestador del modulo de scanning (no monolitico: solo coordina los modulos).

    frame/crop -> keypoints -> orientacion -> suavizado
               -> recepciones -> scanning -> (records / eventos)

Mantiene el estado temporal por track_id (centro previo para la velocidad y
orientacion suavizada previa para el fallback). Devuelve:

  - orientation_records : list[dict]  (filas de player_orientation)
  - reception_events    : list[ReceptionEvent]
  - scanning_metrics    : list[ScanningMetrics]

El render de video/minimapa es responsabilidad de visualization.py y se invoca
aparte (a partir de los records exportados).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .data_io import SequenceData
from .exporter import orientation_record
from .keypoint_extractor import KeypointExtractor
from .orientation_estimator import OrientationEstimator
from .orientation_smoother import OrientationSmoother
from .player_cropper import PlayerCropper
from .reception_detector import ReceptionDetector
from .scanning_detector import FrameOrientation, ScanningDetector


class ScanningPipeline:
    def __init__(self, config: dict, enable_pose: bool = True,
                 keypoint_extractor: Optional[KeypointExtractor] = None):
        sc = config.get("scanning", config)
        self.cfg = sc
        self.fps = float(sc.get("fps", 25.0))
        self.enable_pose = enable_pose

        self.cropper = PlayerCropper(
            expand_ratio=sc.get("crop_expand_ratio", 0.25),
            crop_size=tuple(sc.get("crop_size", (384, 768))),
            min_crop_height=sc.get("min_crop_height", 64),
            low_quality_height=sc.get("low_quality_crop_height", 110),
            save_dir=(config.get("data", {}).get("crop_dir")
                      if sc.get("export_debug_crops") else None),
        )
        self.extractor = None
        if enable_pose:
            self.extractor = keypoint_extractor or KeypointExtractor(sc)
        self.estimator = OrientationEstimator(sc)
        # Suavizado circular independiente para imagen y para cancha (no se pueden
        # mezclar marcos): el minimapa/scanning usan el de cancha cuando hay homografia.
        self.smoother_img = OrientationSmoother(sc)
        self.smoother_field = OrientationSmoother(sc)
        self.reception_detector = ReceptionDetector(sc)
        self.scanning_detector = ScanningDetector(sc)

    def run(self, seq: SequenceData, max_frames: Optional[int] = None,
            verbose: bool = True) -> Tuple[List[dict], list, list]:
        prev_center: Dict[int, Tuple[float, float]] = {}
        prev_smooth: Dict[int, float] = {}
        series: Dict[int, List[FrameOrientation]] = defaultdict(list)
        records: List[dict] = []

        frame_ids = seq.frame_ids[:max_frames] if max_frames else seq.frame_ids
        empty_pose = {"pose_valid": False, "landmarks": {}, "mean_visibility": 0.0,
                      "head_visibility": 0.0, "torso_visibility": 0.0,
                      "backend_used": "invalid", "backend_attempted": ""}
        use_field = seq.homography.available

        for n, fid in enumerate(frame_ids):
            img = None
            if self.enable_pose:
                ipath = seq.image_paths.get(fid)
                img = cv2.imread(str(ipath)) if ipath else None

            ball = seq.ball(fid)
            ball_img = (ball.x, ball.y) if (ball and ball.valid) else None

            for p in seq.players(fid):
                tid = p.track_id
                center = p.center
                vel = None
                if tid in prev_center:
                    vel = (center[0] - prev_center[tid][0],
                           center[1] - prev_center[tid][1])

                # keypoints
                pose = empty_pose
                crop_quality = 0.0
                if self.enable_pose and img is not None:
                    crop = self.cropper.crop(img, p.bbox, fid, tid,
                                             save_debug=self.cfg.get("export_debug_crops", False))
                    crop_quality = 0.0 if not crop.valid_crop else (0.5 if crop.low_quality else 1.0)
                    if crop.valid_crop:
                        pose = self.extractor.extract(crop.crop, crop_height=crop.orig_height)
                    bbox_exp = crop.bbox_expanded
                else:
                    bbox_exp = p.bbox

                ori = self.estimator.estimate(
                    frame_id=fid, track_id=tid, pose=pose, bbox_exp=bbox_exp,
                    velocity=vel, ball_xy=ball_img, player_xy=center,
                    prev_theta=prev_smooth.get(tid), homography=seq.homography,
                    crop_quality=crop_quality,
                )
                sm_img = self.smoother_img.update(tid, fid, ori.theta_visual_img,
                                                  ori.orientation_confidence)
                sm_field = self.smoother_field.update(tid, fid, ori.theta_visual_field,
                                                      ori.orientation_confidence)
                if sm_img.theta_visual_smooth is not None:
                    prev_smooth[tid] = sm_img.theta_visual_smooth

                field_xy = seq.player_field_xy(fid, tid)
                records.append(orientation_record(
                    seq.video_id, ori, sm_img, sm_field, p, self.fps,
                    field_xy=field_xy, velocity=vel, crop_quality=crop_quality,
                    keypoint_backend_used=pose.get("backend_used", ""),
                    keypoint_backend_attempted=pose.get("backend_attempted", "")))

                # Scanning en CANCHA cuando hay homografia (theta y ball_rel en el
                # mismo marco); si no, en imagen.
                if use_field:
                    theta_scan = sm_field.theta_visual_smooth
                    conf_scan = sm_field.smooth_confidence
                    ball_rel_scan = ori.ball_relative_angle_field
                else:
                    theta_scan = sm_img.theta_visual_smooth
                    conf_scan = sm_img.smooth_confidence
                    ball_rel_scan = ori.ball_relative_angle
                series[tid].append(FrameOrientation(
                    frame_id=fid,
                    theta_smooth=theta_scan,
                    confidence=conf_scan,
                    ball_relative_angle=ball_rel_scan,
                ))
                prev_center[tid] = center

            if verbose and n % 100 == 0:
                print(f"  [scanning] frame {fid} ({n + 1}/{len(frame_ids)})")

        # recepciones + scanning
        receptions = self.reception_detector.detect(seq)
        scan_metrics = []
        for ev in receptions:
            frames = series.get(ev.receiver_track_id, [])
            m = self.scanning_detector.evaluate_window(
                video_id=seq.video_id, event_id=ev.event_id,
                receiver_track_id=ev.receiver_track_id,
                frame_reception=ev.frame_reception, frames=frames)
            scan_metrics.append(m)

        return records, receptions, scan_metrics
