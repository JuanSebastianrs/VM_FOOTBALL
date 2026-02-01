# detector.py
"""
Clase base abstracta para detectores de fútbol.
Permite usar YOLO o RF-DETR de forma intercambiable.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """Representa una detección individual."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    class_id: int             # 0=player_left, 1=player_right, 2=gk_left, 3=gk_right, 4=ref, 5=ball
    class_name: str           # Nombre de la clase
    confidence: float         # Score de confianza
    track_id: Optional[int] = None   # ID de tracking (si aplica)
    mask: Optional[np.ndarray] = None  # Máscara de segmentación (si aplica)


class FootballDetector(ABC):
    """
    Clase base para detectores de fútbol.
    
    Clases detectadas (6):
        0: player_left
        1: player_right
        2: goalkeeper_left
        3: goalkeeper_right
        4: referee
        5: ball
    """
    
    CLASS_NAMES = [
        "player_left", "player_right",
        "goalkeeper_left", "goalkeeper_right",
        "referee", "ball"
    ]
    
    def __init__(self, weights_path: Union[str, Path], device: str = "cuda:0"):
        """
        Args:
            weights_path: Ruta al archivo de pesos .pt
            device: Dispositivo de inferencia ("cuda:0", "cpu")
        """
        self.weights_path = Path(weights_path)
        self.device = device
        self.model = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Cargar el modelo en memoria."""
        # TODO: Implementar en subclases
        pass
    
    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """
        Detectar objetos en una imagen.
        
        Args:
            image: Imagen BGR (numpy array)
            conf_threshold: Umbral de confianza mínimo
            iou_threshold: Umbral IoU para NMS
            
        Returns:
            Lista de detecciones
        """
        # TODO: Implementar en subclases
        pass
    
    @abstractmethod
    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[List[Detection]]:
        """
        Detectar objetos en múltiples imágenes.
        
        Args:
            images: Lista de imágenes BGR
            conf_threshold: Umbral de confianza
            iou_threshold: Umbral IoU
            
        Returns:
            Lista de listas de detecciones
        """
        # TODO: Implementar en subclases
        pass
    
    def get_class_name(self, class_id: int) -> str:
        """Obtener nombre de clase dado su ID."""
        return self.CLASS_NAMES[class_id] if 0 <= class_id < len(self.CLASS_NAMES) else "unknown"
    
    def filter_by_class(
        self, 
        detections: List[Detection], 
        class_ids: List[int]
    ) -> List[Detection]:
        """Filtrar detecciones por clase."""
        return [d for d in detections if d.class_id in class_ids]
    
    def get_players(self, detections: List[Detection]) -> List[Detection]:
        """Obtener solo jugadores (excluyendo porteros)."""
        return self.filter_by_class(detections, [0, 1])
    
    def get_goalkeepers(self, detections: List[Detection]) -> List[Detection]:
        """Obtener solo porteros."""
        return self.filter_by_class(detections, [2, 3])
    
    def get_referees(self, detections: List[Detection]) -> List[Detection]:
        """Obtener solo árbitros."""
        return self.filter_by_class(detections, [4])
    
    def get_ball(self, detections: List[Detection]) -> Optional[Detection]:
        """Obtener detección del balón (la de mayor confianza si hay varias)."""
        balls = self.filter_by_class(detections, [5])
        if not balls:
            return None
        return max(balls, key=lambda d: d.confidence)
