# rfdetr_detector.py
"""
Detector basado en RF-DETR (Real-Time DETR de Roboflow).
Detector basado en transformadores, estado del arte.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np

# TODO: Descomentar cuando se instale RF-DETR
# from rfdetr import RFDETR
# import torch

from .detector import FootballDetector, Detection


class RFDETRDetector(FootballDetector):
    """
    Detector de fútbol usando RF-DETR.
    
    RF-DETR es un detector basado en transformadores desarrollado por Roboflow.
    Ventajas sobre YOLO:
        - Mejor rendimiento en objetos pequeños (balón)
        - Sin necesidad de NMS (Hungarian matching)
        - Mejor transferencia a nuevos dominios
    
    Instalación:
        pip install rfdetr
        
    Uso:
        detector = RFDETRDetector("models/rfdetr_football.pt")
        detector.load_model()
        detections = detector.detect(image)
    
    Referencia:
        https://github.com/roboflow/rf-detr
    """
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cuda:0",
        imgsz: int = 640,
    ):
        """
        Args:
            weights_path: Ruta a pesos .pt de RF-DETR
            device: Dispositivo de inferencia
            imgsz: Tamaño de imagen
        """
        super().__init__(weights_path, device)
        self.imgsz = imgsz
        
    def load_model(self) -> None:
        """
        Cargar modelo RF-DETR.
        
        TODO:
            - Inicializar modelo RF-DETR
            - Cargar pesos entrenados
            - Mover a dispositivo
        """
        # TODO: Implementar
        # self.model = RFDETR.from_pretrained(str(self.weights_path))
        # self.model.to(self.device)
        raise NotImplementedError("TODO: Cargar modelo RF-DETR")
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,  # No se usa en DETR, pero se mantiene por interfaz
    ) -> List[Detection]:
        """
        Detectar objetos con RF-DETR.
        
        Nota: RF-DETR no requiere NMS ya que usa Hungarian matching.
        El parámetro iou_threshold se ignora pero se mantiene por compatibilidad.
        
        TODO:
            - Preprocesar imagen
            - Ejecutar inferencia
            - Postprocesar y filtrar por confianza
            - Convertir a List[Detection]
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Implementar detección RF-DETR")
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[List[Detection]]:
        """Detectar en batch de imágenes."""
        # TODO: Implementar
        raise NotImplementedError("TODO: Implementar batch detection RF-DETR")
    
    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """
        Preprocesar imagen para RF-DETR.
        
        TODO:
            - Redimensionar a imgsz
            - Normalizar (ImageNet mean/std)
            - BGR a RGB
            - HWC a CHW
            - Convertir a tensor
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Preprocesamiento RF-DETR")
    
    def _postprocess(self, outputs, conf_threshold: float) -> List[Detection]:
        """
        Postprocesar salidas de RF-DETR.
        
        TODO:
            - Extraer predicciones de bboxes y clases
            - Filtrar por confianza
            - Convertir coordenadas normalizadas a píxeles
            - Crear objetos Detection
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Postprocesamiento RF-DETR")


# Notas de entrenamiento RF-DETR:
# 
# Para entrenar RF-DETR con el dataset SoccerNet:
# 
# 1. Convertir dataset a formato COCO:
#    - Estructura: images/, annotations.json
#    - Formato: {"images": [...], "annotations": [...], "categories": [...]}
# 
# 2. Entrenar:
#    from rfdetr import RFDETR
#    model = RFDETR(num_classes=6)
#    model.train(
#        train_annots="path/to/train.json",
#        val_annots="path/to/val.json", 
#        epochs=50,
#        batch_size=8,
#        lr=1e-4
#    )
# 
# 3. Guardar:
#    model.save("models/rfdetr_football.pt")
