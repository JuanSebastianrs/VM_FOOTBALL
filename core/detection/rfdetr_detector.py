# rfdetr_detector.py
"""
Detector de fútbol basado en RF-DETR (Real-Time DETR de Roboflow).

Compatible con checkpoints de:
    - 1 clase: ["player"]
    - 2 clases: ["player", "goalkeeper"]

RF-DETR maneja preprocesamiento y postprocesamiento internamente
a través de model.predict(), sin necesidad de NMS.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from .detector import FootballDetector, Detection


class RFDETRDetector(FootballDetector):
    """
    Detector de fútbol usando RF-DETR.

    RF-DETR es un detector basado en transformadores desarrollado por Roboflow.
    Ventajas sobre YOLO:
        - Mejor rendimiento en objetos pequeños
        - Sin necesidad de NMS (Hungarian matching)
        - Mejor transferencia a nuevos dominios

    El modelo se entrena con 1 clase unificada ("player") que incluye
    player_left, player_right, goalkeeper_left, goalkeeper_right.

    Uso:
        detector = RFDETRDetector("models/rfdetr_player.pth")
        detector.load_model()
        detections = detector.detect(image)
    """

    # RF-DETR class names (trained with 1 unified class)
    RFDETR_CLASSES = ["player"]

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cuda:0",
        resolution: int = 480,
    ):
        """
        Args:
            weights_path: Ruta a pesos .pth de RF-DETR
            device: Dispositivo de inferencia
            resolution: Resolución de entrada (debe coincidir con entrenamiento)
        """
        super().__init__(weights_path, device)
        self.resolution = resolution
        self.class_names = list(self.RFDETR_CLASSES)

    def load_model(self) -> None:
        """Cargar modelo RF-DETR con pesos entrenados."""
        try:
            from rfdetr import RFDETRBase
        except ImportError:
            raise ImportError(
                "rfdetr no está instalado. Instalar con: pip install rfdetr>=1.4.0"
            )

        self.model = RFDETRBase(
            pretrain_weights=str(self.weights_path),
            resolution=self.resolution,
        )

        # RF-DETR puede exponer los nombres de clase según el checkpoint.
        classes = getattr(self.model, "classes", None)
        if isinstance(classes, (list, tuple)) and classes:
            self.class_names = [str(c) for c in classes]

        print(f"[RF-DETR] Modelo cargado: {self.weights_path}")
        print(f"[RF-DETR] Resolución: {self.resolution}px, Clases: {self.class_names}")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,  # No se usa en DETR, se mantiene por interfaz
    ) -> List[Detection]:
        """
        Detectar jugadores con RF-DETR.

        RF-DETR no requiere NMS (usa Hungarian matching).
        El parámetro iou_threshold se ignora pero se mantiene por compatibilidad.

        Args:
            image: Imagen BGR (numpy array)
            conf_threshold: Umbral de confianza mínimo
            iou_threshold: Ignorado (DETR no usa NMS)

        Returns:
            Lista de detecciones de jugadores
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llamar load_model() primero.")

        from PIL import Image
        import cv2

        # RF-DETR espera PIL Image en RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Inferencia — devuelve supervision.Detections
        sv_detections = self.model.predict(pil_img, threshold=conf_threshold)

        # Convertir a List[Detection]
        detections = []
        for i in range(len(sv_detections)):
            rfdetr_class_id = int(sv_detections.class_id[i])
            class_name = (
                self.class_names[rfdetr_class_id]
                if 0 <= rfdetr_class_id < len(self.class_names)
                else "unknown"
            )

            detections.append(Detection(
                bbox=sv_detections.xyxy[i],     # [x1, y1, x2, y2]
                class_id=rfdetr_class_id,
                class_name=class_name,
                confidence=float(sv_detections.confidence[i]),
            ))

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[List[Detection]]:
        """Detectar en batch de imágenes (secuencial, RF-DETR no soporta batch nativo)."""
        return [self.detect(img, conf_threshold, iou_threshold) for img in images]
