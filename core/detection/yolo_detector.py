# yolo_detector.py
"""
Detector basado en YOLO (v8/v11) usando Ultralytics.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np

# TODO: Descomentar cuando se implemente
# from ultralytics import YOLO
# import torch

from .detector import FootballDetector, Detection


class YOLODetector(FootballDetector):
    """
    Detector de fútbol usando YOLO (Ultralytics).
    
    Soporta YOLOv8 y YOLOv11.
    
    Uso:
        detector = YOLODetector("models/yolo11n_football.pt")
        detector.load_model()
        detections = detector.detect(image)
    """
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cuda:0",
        imgsz: int = 640,
    ):
        """
        Args:
            weights_path: Ruta a pesos .pt de YOLO
            device: Dispositivo ("cuda:0", "cpu")
            imgsz: Tamaño de imagen para inferencia
        """
        super().__init__(weights_path, device)
        self.imgsz = imgsz
        
    def load_model(self) -> None:
        """Cargar modelo YOLO."""
        # TODO: Implementar
        # self.model = YOLO(str(self.weights_path))
        # self.model.to(self.device)
        raise NotImplementedError("TODO: Cargar modelo YOLO")
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """
        Detectar objetos con YOLO.
        
        TODO:
            - Llamar a self.model.predict()
            - Convertir resultados a List[Detection]
            - Manejar caso sin detecciones
        """
        # TODO: Implementar
        # results = self.model.predict(
        #     source=image,
        #     imgsz=self.imgsz,
        #     conf=conf_threshold,
        #     iou=iou_threshold,
        #     device=self.device,
        #     verbose=False
        # )
        # return self._parse_results(results[0])
        raise NotImplementedError("TODO: Implementar detección YOLO")
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[List[Detection]]:
        """Detectar en batch de imágenes."""
        # TODO: Implementar batch processing
        raise NotImplementedError("TODO: Implementar batch detection")
    
    def _parse_results(self, result) -> List[Detection]:
        """
        Convertir resultado de Ultralytics a List[Detection].
        
        TODO:
            - Extraer boxes.xyxy, boxes.conf, boxes.cls
            - Crear objetos Detection
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Parsear resultados YOLO")
    
    def track(
        self,
        source: Union[str, List[np.ndarray]],
        tracker_config: str = "bytetrack.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        persist: bool = True,
    ):
        """
        Tracking con YOLO integrado.
        
        TODO:
            - Usar model.track() de Ultralytics
            - Mantener IDs persistentes
            - Retornar generador de resultados
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Implementar tracking YOLO")
