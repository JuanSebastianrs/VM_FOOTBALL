# core/scanning/keypoint_extractor.py
"""
Extraccion de keypoints corporales sobre crops de jugadores.

Backends intercambiables (patron estrategia) para no acoplar el modulo a un
modelo concreto:

  - "yolo_pose"  : Ultralytics YOLO-Pose (COCO-17). Es el backend por defecto
                   porque ultralytics ya es dependencia del proyecto y no
                   requiere instalar nada nuevo.
  - "mediapipe"  : MediaPipe Pose Landmarker (import perezoso; requiere instalar
                   `mediapipe`).

Ambos producen el MISMO dict canonico de landmarks para que el estimador de
orientacion sea agnostico al backend. Las coordenadas se devuelven NORMALIZADAS
en [0, 1] relativas al crop (x/crop_w, y/crop_h); asi son invariantes al resize
y el estimador puede remapearlas al frame completo sin distorsion de aspecto.

Landmarks canonicos:
  nose, left_eye, right_eye, left_ear, right_ear,
  left_shoulder, right_shoulder, left_hip, right_hip
Cada valor es [x, y, z, visibility].

Diseno extensible: para anadir MoveNet / ViTPose / RTMPose basta con crear una
subclase de `PoseBackend` que rellene `CANONICAL_LANDMARKS`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

CANONICAL_LANDMARKS: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
]
HEAD_LANDMARKS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
TORSO_LANDMARKS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

# COCO-17 (orden de Ultralytics YOLO-Pose) -> landmark canonico
_COCO_TO_CANONICAL = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 11: "left_hip", 12: "right_hip",
}

# Indices de MediaPipe Pose (33 landmarks) -> landmark canonico
_MP_TO_CANONICAL = {
    0: "nose", 2: "left_eye", 5: "right_eye", 7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder", 23: "left_hip", 24: "right_hip",
}


def _empty_landmarks() -> Dict[str, list]:
    return {name: [0.0, 0.0, 0.0, 0.0] for name in CANONICAL_LANDMARKS}


def _aggregate(result: dict) -> dict:
    """Calcula mean/head/torso visibility a partir de los landmarks."""
    lm = result["landmarks"]
    vis = {k: lm[k][3] for k in CANONICAL_LANDMARKS}
    head = [vis[k] for k in HEAD_LANDMARKS]
    torso = [vis[k] for k in TORSO_LANDMARKS]
    result["mean_visibility"] = float(np.mean(list(vis.values())))
    result["head_visibility"] = float(np.mean(head))
    result["torso_visibility"] = float(np.mean(torso))
    return result


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class PoseBackend:
    """Interfaz base de un backend de pose."""

    def infer(self, crop: np.ndarray) -> dict:
        raise NotImplementedError


class YoloPoseBackend(PoseBackend):
    def __init__(self, model_path: str = "yolo11n-pose.pt",
                 device: str = "cuda:0", conf: float = 0.25):
        from ultralytics import YOLO  # import local: dependencia ya presente
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def infer(self, crop: np.ndarray) -> dict:
        out = _result_template("yolo_pose")
        h, w = crop.shape[:2]
        res = self.model.predict(crop, device=self.device, conf=self.conf,
                                 verbose=False)
        if not res:
            return out
        r = res[0]
        if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
            return out
        # Elegir la persona de mayor confianza (la caja del detector)
        if r.boxes is not None and len(r.boxes) > 0:
            idx = int(np.argmax(r.boxes.conf.cpu().numpy()))
        else:
            idx = 0
        xy = r.keypoints.xy[idx].cpu().numpy()              # (17, 2) px
        conf = (r.keypoints.conf[idx].cpu().numpy()
                if r.keypoints.conf is not None else np.ones(len(xy)))
        for ci, name in _COCO_TO_CANONICAL.items():
            if ci < len(xy):
                x, y = xy[ci]
                out["landmarks"][name] = [float(x) / w, float(y) / h, 0.0,
                                          float(conf[ci])]
        out["pose_valid"] = True
        return _aggregate(out)


# URLs oficiales de los modelos Pose Landmarker (MediaPipe Tasks)
_MP_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def _resolve_mediapipe_model(model_path: Optional[str], complexity: str,
                             models_dir: str = "models") -> str:
    """Devuelve la ruta a un .task; lo descarga a models/ si hace falta."""
    import os
    import urllib.request
    if model_path and os.path.exists(model_path):
        return model_path
    complexity = (complexity or "full").lower()
    if complexity not in _MP_MODEL_URLS:
        complexity = "full"
    os.makedirs(models_dir, exist_ok=True)
    dst = os.path.join(models_dir, f"pose_landmarker_{complexity}.task")
    if not os.path.exists(dst):
        print(f"[mediapipe] descargando pose_landmarker_{complexity}.task ...")
        urllib.request.urlretrieve(_MP_MODEL_URLS[complexity], dst)
    return dst


class MediaPipeBackend(PoseBackend):
    """
    Backend basado en el Pose Landmarker de MediaPipe Tasks (full/heavy).

    Corre sobre CROPS individuales (no sobre el frame completo). Devuelve los
    mismos campos normalizados que YOLO-Pose y, ademas, los `world_landmarks`
    (coords 3D en metros, centradas en la cadera) que entrega MediaPipe.
    """

    def __init__(self, model_path: Optional[str] = None,
                 complexity: str = "full", min_confidence: float = 0.3):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "El backend 'mediapipe' requiere `pip install mediapipe` "
                "(Tasks API). Alternativa: keypoint_backend 'yolo_pose'."
            ) from e
        self._mp = mp
        self.complexity = complexity
        task_path = _resolve_mediapipe_model(model_path, complexity)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def infer(self, crop: np.ndarray) -> dict:
        import cv2
        out = _result_template("mediapipe")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_image)
        if not res.pose_landmarks:
            return out
        lms = res.pose_landmarks[0]
        for mi, name in _MP_TO_CANONICAL.items():
            lm = lms[mi]
            # MediaPipe da visibility y presence; usamos visibility como confianza
            out["landmarks"][name] = [float(lm.x), float(lm.y), float(lm.z),
                                      float(lm.visibility)]
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks[0]
            for mi, name in _MP_TO_CANONICAL.items():
                w = wl[mi]
                out["world_landmarks"][name] = [float(w.x), float(w.y),
                                                float(w.z), float(w.visibility)]
        out["pose_valid"] = True
        return _aggregate(out)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def _result_template(backend: str = "") -> dict:
    return {
        "pose_valid": False,
        "backend": backend,                      # backend que PRODUJO landmarks
        "backend_attempted": "",                 # backend(s) intentado(s), p.ej. "mediapipe+yolo_pose"
        "backend_used": "invalid",               # backend efectivo (o "invalid" si nada detecto)
        "landmarks": _empty_landmarks(),         # normalizados al crop [0,1]
        "world_landmarks": {},                   # metros (hip-centered); solo si el backend los da
        "mean_visibility": 0.0,
        "head_visibility": 0.0,
        "torso_visibility": 0.0,
    }


# ---------------------------------------------------------------------------
# Fachada
# ---------------------------------------------------------------------------

class KeypointExtractor:
    """
    Fachada que selecciona el backend y expone `extract(crop) -> dict`.

    Config esperada (subdict `scanning` de configs/scanning.yaml):
      keypoint_backend ("yolo_pose" | "mediapipe"), min_pose_confidence
      yolo_pose:   pose_model, pose_device
      mediapipe:   mediapipe_model_complexity ("full"|"heavy"|"lite"),
                   mediapipe_model_path (.task opcional)

    `keypoint_backend` reemplaza al antiguo `pose_backend` (que se sigue
    aceptando por retrocompatibilidad).
    """

    def __init__(self, config: Optional[dict] = None, backend: Optional[PoseBackend] = None,
                 backend_name: Optional[str] = None):
        config = config or {}
        self.config = config
        if backend is not None:
            self.backend = backend
            self.name = getattr(backend, "name", "custom")
            return
        name = backend_name or config.get("keypoint_backend",
                                          config.get("pose_backend", "yolo_pose"))
        self.name = name
        self._mp_threshold = float(config.get("mediapipe_min_crop_height", 110))
        if name == "yolo_pose":
            self.backend = self._make_yolo(config)
        elif name == "mediapipe":
            self.backend = self._make_mediapipe(config)
        elif name == "hybrid":
            # Recomendado por el benchmark: MediaPipe (cabeza mas limpia, menos
            # jitter) en crops grandes; YOLO-Pose (mejor cobertura) en pequenos.
            self._yolo = self._make_yolo(config)
            self._mp = self._make_mediapipe(config)
            self.backend = self._yolo
        else:
            raise ValueError(f"keypoint_backend desconocido: {name!r}")

    @staticmethod
    def _make_yolo(config: dict) -> "YoloPoseBackend":
        return YoloPoseBackend(
            model_path=config.get("pose_model", "yolo11n-pose.pt"),
            device=config.get("pose_device", "cuda:0"),
            conf=config.get("min_pose_confidence", 0.25))

    @staticmethod
    def _make_mediapipe(config: dict) -> "MediaPipeBackend":
        return MediaPipeBackend(
            model_path=config.get("mediapipe_model_path"),
            complexity=config.get("mediapipe_model_complexity", "full"),
            min_confidence=config.get("min_pose_confidence", 0.3))

    def extract(self, crop: np.ndarray, crop_height: Optional[float] = None) -> dict:
        """Extrae keypoints y anota `backend_attempted` / `backend_used`.

        En modo "hybrid":
          - si `crop_height >= mediapipe_min_crop_height` intenta MediaPipe;
          - si MediaPipe devuelve pose_valid=False, REINTENTA con YOLO-Pose;
          - crops por debajo del umbral (o sin `crop_height`) van directo a YOLO;
          - si nada detecta, backend_used="invalid".
        """
        if crop is None or crop.size == 0:
            t = _result_template()
            t["backend_attempted"] = ""
            return t

        if self.name == "hybrid":
            attempted = []
            use_mp = crop_height is not None and crop_height >= self._mp_threshold
            if use_mp:
                res = self._mp.infer(crop); attempted.append("mediapipe")
                if not res.get("pose_valid"):
                    res = self._yolo.infer(crop); attempted.append("yolo_pose")
            else:
                res = self._yolo.infer(crop); attempted.append("yolo_pose")
            res["backend_attempted"] = "+".join(attempted)
        else:
            res = self.backend.infer(crop)
            res["backend_attempted"] = self.name

        res["backend_used"] = res["backend"] if res.get("pose_valid") else "invalid"
        return res

    def close(self) -> None:
        """Libera recursos del backend (relevante para MediaPipe)."""
        for b in {id(self.backend): self.backend,
                  **({id(self._mp): self._mp} if self.name == "hybrid" else {})}.values():
            close = getattr(b, "close", None)
            if callable(close):
                close()
