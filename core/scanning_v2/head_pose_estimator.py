# core/scanning_v2/head_pose_estimator.py
"""
FASE 5 — Head pose estimation (orientacion visual APROXIMADA, NO gaze real).

Cadena de backends (configurable `backend_order`):
  1. sixdrepnet   : 6DRepNet sobre el head crop -> yaw/pitch/roll (camara). Lazy
                    import; si no esta instalado, se omite (fallback documentado).
  2. mediapipe    : FaceLandmarker (Tasks API) -> matriz de transformacion facial
                    -> yaw/pitch/roll. Falla en cabezas muy pequenas; se omite.
  3. yolo_pose_body : proxy de yaw a partir de la geometria de los keypoints de
                    cabeza (nariz vs orejas/ojos) ya extraidos del cuerpo. Es el
                    fallback robusto en broadcast.

SISTEMA DE REFERENCIA (importante):
  - `yaw_raw` es CAMARA-RELATIVO (no es orientacion en cancha). Se usa para
    detectar *cambios* de cabeza (head-turn), no como orientacion absoluta.
  - `theta_body_img/field` (orientacion de cuerpo) viene del estimador V1 ya
    validado y SI tiene marco imagen/cancha. Nno se mezcla con el yaw de cabeza.
  - `theta_head_field` queda None salvo que exista una conversion explicita
    (no disponible de forma fiable en broadcast).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.scanning.data_io import HomographyLookup
from core.scanning.orientation_estimator import OrientationEstimator

_HEAD_LM = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


def _default_gpu_id(c: dict) -> int:
    """Deriva gpu_id de `pose_device` ('cuda:0'->0, 'cpu'->-1). Si no se da, -1.
    No se asume CUDA disponible: el default es CPU."""
    dev = str(c.get("pose_device", "cpu")).lower()
    if dev.startswith("cuda"):
        try:
            return int(dev.split(":")[1]) if ":" in dev else 0
        except (ValueError, IndexError):
            return 0
    return -1


@dataclass
class HeadPose:
    yaw_raw: Optional[float]            # grados, camara-relativo
    pitch_raw: Optional[float]
    roll_raw: Optional[float]
    confidence: float
    backend_used: str
    backend_attempted: str
    theta_body_img: Optional[float]     # rad
    theta_body_field: Optional[float]   # rad
    theta_head_img: Optional[float]     # rad (None si no es fiable)
    theta_head_field: Optional[float]   # rad (None: requiere conversion explicita)
    ball_relative_angle_field: Optional[float] = None   # rad (marco cancha)


# --------------------------------------------------------------------------
# Backends de yaw
# --------------------------------------------------------------------------

class _SixDRepNet:
    available = False

    def __init__(self, gpu_id: int = -1):
        """gpu_id >= 0 usa esa GPU; gpu_id < 0 fuerza CPU. Configurable (no se
        hardcodea cuda:0)."""
        try:
            from sixdrepnet import SixDRepNet  # type: ignore
            try:
                self.model = SixDRepNet(gpu_id=gpu_id)
            except TypeError:
                self.model = SixDRepNet()   # versiones sin gpu_id
            self.available = True
        except Exception:
            self.available = False

    def yaw(self, head_crop) -> Optional[Tuple[float, float, float, float]]:
        if not self.available or head_crop is None or head_crop.size == 0:
            return None
        try:
            pitch, yaw, roll = self.model.predict(head_crop)
            return float(yaw), float(pitch), float(roll), 0.8
        except Exception:
            return None


class _MediaPipeFace:
    available = False

    def __init__(self, model_path: str = "models/face_landmarker.task",
                 allow_download: bool = False):
        """Carga FaceLandmarker desde `model_path`. Solo descarga el modelo si
        `allow_download=True` (no se descargan modelos en runtime por defecto)."""
        try:
            import os
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            dst = model_path
            if not os.path.exists(dst):
                if not allow_download:
                    self.available = False
                    return
                import urllib.request
                url = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                       "face_landmarker/float16/latest/face_landmarker.task")
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                urllib.request.urlretrieve(url, dst)
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=dst),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True)
            self._mp = mp
            self.landmarker = vision.FaceLandmarker.create_from_options(opts)
            self.available = True
        except Exception:
            self.available = False

    def yaw(self, head_crop) -> Optional[Tuple[float, float, float, float]]:
        if not self.available or head_crop is None or head_crop.size == 0:
            return None
        try:
            import cv2
            rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(img)
            if not res.facial_transformation_matrixes:
                return None
            m = np.array(res.facial_transformation_matrixes[0]).reshape(4, 4)
            R = m[:3, :3]
            yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
            pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            return float(yaw), float(pitch), float(roll), 0.7
        except Exception:
            return None


def _yaw_from_head_keypoints(landmarks: Dict[str, list], min_vis: float
                             ) -> Optional[Tuple[float, float]]:
    """Proxy de yaw camara-relativo desde nariz vs orejas/ojos. (yaw_deg, conf)."""
    def ok(n):
        return n in landmarks and landmarks[n][3] >= min_vis
    nose = landmarks.get("nose")
    if not nose or nose[3] < min_vis:
        return None
    if ok("left_ear") and ok("right_ear"):
        a, b = landmarks["left_ear"], landmarks["right_ear"]
    elif ok("left_eye") and ok("right_eye"):
        a, b = landmarks["left_eye"], landmarks["right_eye"]
    else:
        return None
    midx = (a[0] + b[0]) / 2.0
    half = abs(b[0] - a[0]) / 2.0
    if half < 1e-3:
        return None
    ratio = float(np.clip((nose[0] - midx) / half, -1.6, 1.6))
    yaw = ratio * 60.0   # proxy en grados (positivo = nariz hacia +x imagen)
    conf = float(min(nose[3], a[3], b[3]))
    return yaw, conf


class HeadPoseEstimator:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.order: List[str] = list(c.get("backend_order",
                                            ["sixdrepnet", "mediapipe", "yolo_pose_body"]))
        self.min_conf = float(c.get("min_confidence", 0.35))
        self.min_kp_vis = float(c.get("min_pose_confidence", 0.3))
        self.allow_body_fallback = bool(c.get("allow_body_fallback", True))
        # device / modelos configurables (no se hardcodea cuda:0)
        self.sixdrepnet_gpu_id = int(c.get("sixdrepnet_gpu_id", _default_gpu_id(c)))
        self.face_model_path = str(c.get("face_model_path", "models/face_landmarker.task"))
        self.allow_model_download = bool(c.get("allow_model_download", False))
        self._body = OrientationEstimator(c)
        self._six = None
        self._face = None

    def _get_six(self):
        if self._six is None:
            self._six = _SixDRepNet(gpu_id=self.sixdrepnet_gpu_id)
        return self._six

    def _get_face(self):
        if self._face is None:
            self._face = _MediaPipeFace(self.face_model_path, self.allow_model_download)
        return self._face

    def estimate(
        self, head_crop, body_pose: dict, bbox_exp: np.ndarray,
        frame_id: int, player_xy: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
        ball_xy: Optional[Tuple[float, float]] = None,
        prev_theta: Optional[float] = None,
        homography: Optional[HomographyLookup] = None,
        crop_quality: float = 1.0,
    ) -> HeadPose:
        # --- orientacion de cuerpo (reusa estimador V1: imagen + cancha) ---
        body = self._body.estimate(
            frame_id=frame_id, track_id=0, pose=body_pose, bbox_exp=bbox_exp,
            velocity=velocity, ball_xy=ball_xy, player_xy=player_xy,
            prev_theta=prev_theta, homography=homography, crop_quality=crop_quality)

        attempted: List[str] = []
        yaw = pitch = roll = None
        conf = 0.0
        used = "invalid"

        for bk in self.order:
            if bk == "sixdrepnet":
                attempted.append(bk)
                r = self._get_six().yaw(head_crop)
                if r:
                    yaw, pitch, roll, conf = r
                    used = bk
                    break
            elif bk == "mediapipe":
                attempted.append(bk)
                r = self._get_face().yaw(head_crop)
                if r:
                    yaw, pitch, roll, conf = r
                    used = bk
                    break
            elif bk == "yolo_pose_body":
                attempted.append(bk)
                if body_pose.get("pose_valid"):
                    r = _yaw_from_head_keypoints(body_pose.get("landmarks", {}),
                                                 self.min_kp_vis)
                    if r:
                        yaw, conf = r
                        used = bk
                        break

        # penalizacion por calidad de crop / pocos landmarks
        conf *= float(np.clip(0.4 + 0.6 * crop_quality, 0.0, 1.0))

        # fallback final: si no hay yaw pero hay cuerpo, derivar un yaw relativo
        # del cambio de orientacion corporal (marcado con baja confianza)
        if yaw is None and self.allow_body_fallback and body.theta_visual_img is not None:
            yaw = math.degrees(body.theta_visual_img)
            used = "body_orientation"
            attempted.append("body_orientation")
            conf = max(conf, 0.2 * body.orientation_confidence)

        return HeadPose(
            yaw_raw=yaw, pitch_raw=pitch, roll_raw=roll,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            backend_used=used, backend_attempted="+".join(attempted),
            theta_body_img=body.theta_visual_img,
            theta_body_field=body.theta_visual_field,
            theta_head_img=None,            # yaw es camara-relativo: no marco imagen fiable
            theta_head_field=None,          # requiere conversion explicita (no disponible)
            ball_relative_angle_field=body.ball_relative_angle_field,
        )
