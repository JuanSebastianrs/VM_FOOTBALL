# core/scanning_v2/scanning_pipeline_v2.py
"""
Orquestador del scanning V2.

Flujo (cada fase es un modulo separado; ver docs/scanning_v2/methodology.md):

  game_state -> eventos pase/recepcion -> ventanas -> head crop -> head pose
            -> suavizado yaw -> head-turn -> export (+ render + annotation pack)

Decisiones de receptor SOLO vienen del game_state/eventos (sin arbitros). El
head pose NO redefine quien recibe.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from core.game_state import build_game_state
from core.events import PassReceptionDetector, load_event_ground_truth
from core.scanning.data_io import load_sequence
from core.scanning.keypoint_extractor import KeypointExtractor
from core.scanning.player_cropper import PlayerCropper
from . import exporter
from .head_cropper import HeadCropper
from .head_pose_estimator import HeadPoseEstimator
from .head_pose_smoother import HeadYawSmoother
from .head_turn_detector import HeadTurnDetector
from .reception_window_extractor import ReceptionWindowExtractor
from .schema import HEAD_POSE_COLUMNS
from .vision_map_wog import VisionMapWOG

_HEAD_LM = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


class ScanningPipelineV2:
    def __init__(self, config: dict):
        self.cfg = config
        self.fps = float(config.get("video", {}).get("fps", 25.0))
        self.events_cfg = dict(config.get("events", {})); self.events_cfg["fps"] = self.fps
        self.win_cfg = dict(config.get("windows", {})); self.win_cfg["fps"] = self.fps
        self.save_windows_cfg = bool(self.win_cfg.get("save_windows", True))
        self.save_full_frames = bool(self.win_cfg.get("save_full_frames", False))
        ht = dict(config.get("head_turn", {})); ht.update(self.win_cfg); ht["fps"] = self.fps
        self.head_turn_cfg = ht
        self.cropper = PlayerCropper()
        self.head_cropper = HeadCropper(config.get("head_crop", {}))
        # keypoints configurables (NO hardcodear cuda:0 / modelo)
        self.kp_cfg = dict(config.get("keypoints", {}))
        self.kp_cfg.setdefault("keypoint_backend", "yolo_pose")
        self.kp_cfg.setdefault("pose_model", "models/yolo11n-pose.pt")
        self.kp_cfg.setdefault("pose_device", "cpu")
        self.kp = None
        hp_cfg = dict(config.get("head_pose", {})); hp_cfg["min_pose_confidence"] = 0.3
        hp_cfg.setdefault("pose_device", self.kp_cfg["pose_device"])
        self.head_pose = HeadPoseEstimator(hp_cfg)
        self.smoother = HeadYawSmoother(config.get("smoothing", {}))
        self.head_turn = HeadTurnDetector(ht)
        self.window_extractor = ReceptionWindowExtractor(self.win_cfg)
        self.vision_map = VisionMapWOG(config.get("vision_map", {}))
        self.vision_export = bool(config.get("vision_map", {}).get("export_per_event", True))

    # ------------------------------------------------------------------
    def run(self, video_id: str, detections, trajectory=None, team_assignments=None,
            calibration=None, sequence_dir=None, event_ground_truth=None,
            out_dir: str = "outputs/scanning_v2", save_windows: bool = True) -> dict:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        save_windows = bool(save_windows and self.save_windows_cfg)

        # 1) game state
        gs, gs_report = build_game_state(
            video_id, detections, trajectory, team_assignments, calibration,
            sequence_dir=sequence_dir, fps=self.fps)
        exporter.export_game_state(gs, out)

        seq = load_sequence(video_id, detections, trajectory, team_assignments,
                            calibration, sequence_dir=sequence_dir, fps=self.fps)
        attached = {fid: int(b.attached_player_id)
                    for fid, b in seq.ball_by_frame.items()
                    if b.attached_player_id is not None}

        # 2) eventos
        gt = load_event_ground_truth(event_ground_truth, video_id)
        self.events_cfg["has_homography"] = seq.homography.available
        detector = PassReceptionDetector(self.events_cfg)
        events, rejected = detector.detect(gs, video_id, attached, gt)
        exporter.export_events(events, out)
        exporter.export_rejected(rejected, out)

        # 3) ventanas
        specs = self.window_extractor.extract(events, gs)
        if save_windows:
            for s in specs:
                self.window_extractor.write_metadata(out, s)

        # 4-6) head crop + head pose + suavizado
        head_pose = self._head_pose_all(seq, gs, specs, out, save_windows)
        exporter.export_head_pose(head_pose, out)

        # 7) head-turn / scanning
        scanning = self.head_turn.evaluate_all(events, head_pose)
        exporter.export_scanning(scanning, out)

        # 8) vision map por evento (recepcion); no inventa orientacion si no hay cancha
        vision = self._vision_map_all(seq, gs, events, head_pose)
        if self.vision_export:
            exporter.export_vision_map(vision, out)

        def _count(reason):
            return int((rejected["reason"] == reason).sum()) if len(rejected) else 0

        summary = {
            "video_id": video_id,
            "game_state": gs_report,
            "receptions_detected": int(len(events)),
            "rejected_total": int(len(rejected)),
            "rejected_referee": _count("referee"),
            "rejected_unknown_role": _count("unknown_role"),
            "rejected_missing_receiver_in_gt": _count("missing_receiver_in_gt"),
            "rejected_invalid_attached": _count("invalid_attached_player"),
            "rejected_by_reason": ({str(k): int(v) for k, v in
                                    rejected["reason"].value_counts().items()}
                                   if len(rejected) else {}),
            "scanning_events_predicted": int((scanning["scan_label_pred"] == 1).sum()) if len(scanning) else 0,
            "events": events, "scanning": scanning, "head_pose": head_pose,
            "vision_map": vision, "out_dir": str(out),
        }
        return summary

    # ------------------------------------------------------------------
    def _vision_map_all(self, seq, gs, events, head_pose) -> pd.DataFrame:
        """Vision map en el frame de recepcion de cada evento. Usa head_field si
        existe, si no body_field (conf baja); si no hay cancha -> available=False."""
        rows: List[dict] = []
        if not len(events):
            return pd.DataFrame(rows)
        hp_idx = head_pose.set_index(["event_id", "frame_id"]) if len(head_pose) else None
        for ev in events.to_dict("records"):
            eid = ev["event_id"]; tid = int(ev["receiver_track_id"])
            f_rec = int(ev["frame_reception"])
            gframe = gs[(gs["frame_id"] == f_rec)]
            recv = gframe[gframe["track_id"] == tid]
            player_xy = None
            if not recv.empty and pd.notna(recv["x_field"].iloc[0]):
                player_xy = (float(recv["x_field"].iloc[0]), float(recv["y_field"].iloc[0]))
            # orientacion: el frame de recepcion suele estar fuera de la ventana de
            # head_pose (que termina ~0.2s antes); usamos el ultimo frame valido.
            theta_head = theta_body = None
            if hp_idx is not None:
                sub = head_pose[(head_pose["event_id"] == eid)
                                & head_pose["theta_body_field"].notna()]
                if not sub.empty:
                    last = sub.sort_values("frame_id").iloc[-1]
                    theta_body = (math.radians(float(last["theta_body_field"]))
                                  if pd.notna(last["theta_body_field"]) else None)
                    if pd.notna(last.get("theta_head_field")):
                        theta_head = math.radians(float(last["theta_head_field"]))
            others = []
            for _, r in gframe.iterrows():
                if int(r["track_id"]) == tid or pd.isna(r.get("x_field")):
                    continue
                team = r.get("team_id")
                others.append((int(r["track_id"]), float(r["x_field"]),
                               float(r["y_field"]),
                               int(team) if pd.notna(team) else None))
            ball_xy = None
            bxy = seq.ball_field_xy(f_rec) if seq.homography.available else None
            if bxy is not None:
                ball_xy = bxy
            res = self.vision_map.compute(
                player_xy, theta_head, theta_body, other_players=others,
                ball_xy=ball_xy, player_team=ev.get("receiver_team_id"))
            rows.append({
                "event_id": eid, "video_id": ev["video_id"],
                "receiver_track_id": tid, "frame_reception": f_rec,
                "vision_map_available": bool(res.available),
                "used_orientation": res.used_orientation,
                "vision_map_confidence": round(float(res.vision_map_confidence), 3),
                "n_visible_teammates": len(res.visible_teammates),
                "n_visible_opponents": len(res.visible_opponents),
                "visible_ball": bool(res.visible_ball),
                "observed_space_score": round(float(res.observed_space_score), 4),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _head_pose_all(self, seq, gs, specs, out, save_windows) -> pd.DataFrame:
        if self.kp is None:
            # backend / modelo / device vienen de configs/scanning_v2.yaml (keypoints:)
            self.kp = KeypointExtractor(dict(self.kp_cfg))
        rows: List[dict] = []
        gs_idx = gs.set_index(["frame_id", "track_id"])
        for s in specs:
            ev = s.event
            tid = int(ev["receiver_track_id"])
            f_rec = int(ev["frame_reception"])
            prev_center = None
            prev_theta = None
            raw_series: List[Tuple[Optional[float], float]] = []
            frame_rows: List[dict] = []
            for fid in s.frames:
                key = (fid, tid)
                if key not in gs_idx.index:
                    continue
                r = gs_idx.loc[key]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                bbox = np.array([r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]], float)
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                vel = None if prev_center is None else (center[0] - prev_center[0],
                                                        center[1] - prev_center[1])
                prev_center = center
                ipath = seq.image_paths.get(fid)
                img = cv2.imread(str(ipath)) if ipath else None

                pose = {"pose_valid": False, "landmarks": {}}
                bbox_exp = bbox
                crop_quality = 0.0
                head_valid = False
                head_q = 0.0
                head_src = "invalid"
                head_img = None
                wdir = out / "windows" / str(ev["event_id"])
                if img is not None:
                    crop = self.cropper.crop(img, bbox, fid, tid)
                    if save_windows and self.save_full_frames:
                        cv2.imwrite(str(wdir / "frames" / f"f{fid:06d}.jpg"), img)
                    if crop.valid_crop:
                        crop_quality = 0.5 if crop.low_quality else 1.0
                        if save_windows and crop.crop is not None and crop.crop.size:
                            cv2.imwrite(str(wdir / "player_crops" / f"f{fid:06d}.jpg"),
                                        crop.crop)
                        pose = self.kp.extract(crop.crop, crop_height=crop.orig_height)
                        bbox_exp = crop.bbox_expanded
                        # head points -> frame
                        hp_pts = self._head_points_frame(pose, bbox_exp)
                        hc = self.head_cropper.crop(img, bbox, hp_pts)
                        head_valid, head_q, head_src = hc.valid, hc.quality, hc.source
                        head_img = hc.crop if hc.valid else None
                        if save_windows and hc.valid:
                            cv2.imwrite(str(wdir / "head_crops" / f"f{fid:06d}.jpg"),
                                        hc.crop)

                ball = seq.ball(fid)
                ball_xy = (ball.x, ball.y) if (ball and ball.valid) else None
                hp = self.head_pose.estimate(
                    head_img, pose, bbox_exp, fid, player_xy=center, velocity=vel,
                    ball_xy=ball_xy, prev_theta=prev_theta,
                    homography=seq.homography, crop_quality=max(crop_quality, head_q))
                if hp.theta_body_img is not None:
                    prev_theta = hp.theta_body_img

                raw_series.append((hp.yaw_raw, hp.confidence))
                frame_rows.append({
                    "event_id": ev["event_id"], "video_id": ev["video_id"],
                    "frame_id": fid, "track_id": tid,
                    "yaw_raw": hp.yaw_raw, "pitch_raw": hp.pitch_raw, "roll_raw": hp.roll_raw,
                    "yaw_smooth": None, "yaw_confidence_smooth": None,
                    "head_pose_confidence": hp.confidence,
                    "head_pose_backend_used": hp.backend_used,
                    "head_pose_backend_attempted": hp.backend_attempted,
                    "head_crop_valid": head_valid, "head_crop_quality": round(head_q, 3),
                    "head_crop_source": head_src,
                    "theta_body_img": _deg(hp.theta_body_img),
                    "theta_body_field": _deg(hp.theta_body_field),
                    "theta_head_img": _deg(hp.theta_head_img),
                    "theta_head_field": _deg(hp.theta_head_field),
                    "ball_relative_angle_field": _deg(hp.ball_relative_angle_field),
                    "distance_to_ball": r.get("distance_to_ball"),
                    "time_to_reception": (f_rec - fid) / self.fps,
                })
            # suavizado del yaw (circular) por evento; se persiste tambien la
            # confianza suavizada (yaw_confidence_smooth) para el head-turn.
            sm = self.smoother.smooth(raw_series)
            for fr, (ys, cs) in zip(frame_rows, sm):
                fr["yaw_smooth"] = ys
                fr["yaw_confidence_smooth"] = round(float(cs), 4)
            rows.extend(frame_rows)
        return pd.DataFrame(rows, columns=HEAD_POSE_COLUMNS)

    @staticmethod
    def _head_points_frame(pose, bbox_exp):
        if not pose.get("pose_valid"):
            return None
        x1, y1, x2, y2 = bbox_exp
        w, h = (x2 - x1), (y2 - y1)
        lm = pose.get("landmarks", {})
        pts = {}
        for n in _HEAD_LM:
            if n in lm and lm[n][3] >= 0.3:
                pts[n] = (x1 + lm[n][0] * w, y1 + lm[n][1] * h)
        return pts or None


def _deg(x):
    return None if x is None else math.degrees(x)
