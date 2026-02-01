# team_classifier.py
"""
Clasificador de equipos usando clustering K-Means.
Combina embeddings de SigLIP2 con descriptores HSV.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import cv2

# TODO: Descomentar cuando se implemente
# from sklearn.cluster import KMeans


@dataclass
class TeamAssignment:
    """Asignación de equipo para un jugador."""
    track_id: int
    team_id: int          # 0 o 1
    confidence: float     # Confianza de la asignación
    embedding: Optional[np.ndarray] = None


class TeamClassifier:
    """
    Clasificador de equipos para jugadores de fútbol.
    
    Estrategia:
        1. Extraer crop del torso (zona de camiseta)
        2. Generar embedding visual (SigLIP2 o HSV)
        3. Clustering K-Means con k=2
        4. Heurística left/right para normalizar etiquetas
    
    Uso:
        classifier = TeamClassifier(mode="siglip")  # o "hsv"
        classifier.fit(player_crops)
        team_id = classifier.predict(crop)
    """
    
    MODES = ["hsv", "siglip", "hybrid"]
    
    def __init__(
        self,
        mode: str = "hsv",
        siglip_model: Optional[str] = None,
        n_clusters: int = 2,
        random_state: int = 42,
    ):
        """
        Args:
            mode: Modo de embedding ("hsv", "siglip", "hybrid")
            siglip_model: Nombre del modelo SigLIP para modo siglip/hybrid
            n_clusters: Número de clusters (default 2 equipos)
            random_state: Semilla para reproducibilidad
        """
        if mode not in self.MODES:
            raise ValueError(f"Mode debe ser uno de {self.MODES}")
            
        self.mode = mode
        self.siglip_model = siglip_model
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.kmeans = None
        self.embedder = None
        self.track_to_team: Dict[int, int] = {}
        
    def fit(
        self,
        crops: List[np.ndarray],
        track_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Ajustar clasificador con crops de jugadores.
        
        TODO:
            - Generar embeddings para cada crop
            - Ajustar K-Means
            - Normalizar etiquetas (left/right heuristic)
        
        Args:
            crops: Lista de crops BGR de torsos
            track_ids: IDs de track correspondientes (opcional)
        """
        # TODO: Implementar
        # embeddings = [self._get_embedding(c) for c in crops]
        # X = np.vstack(embeddings)
        # self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        # labels = self.kmeans.fit_predict(X)
        raise NotImplementedError("TODO: Implementar fit de TeamClassifier")
    
    def predict(self, crop: np.ndarray) -> int:
        """
        Predecir equipo para un crop.
        
        Args:
            crop: Crop BGR del torso
            
        Returns:
            ID de equipo (0 o 1)
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Implementar predict")
    
    def predict_batch(
        self,
        crops: List[np.ndarray],
        track_ids: Optional[List[int]] = None,
    ) -> List[TeamAssignment]:
        """
        Predecir equipos para múltiples crops.
        
        Args:
            crops: Lista de crops BGR
            track_ids: IDs de track correspondientes
            
        Returns:
            Lista de asignaciones de equipo
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Implementar predict_batch")
    
    def _get_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Obtener embedding para un crop según el modo.
        
        Args:
            crop: Crop BGR del torso
            
        Returns:
            Vector de embedding
        """
        if self.mode == "hsv":
            return self._hsv_descriptor(crop)
        elif self.mode == "siglip":
            return self._siglip_embedding(crop)
        else:  # hybrid
            hsv = self._hsv_descriptor(crop)
            siglip = self._siglip_embedding(crop)
            return np.concatenate([hsv, siglip])
    
    def _hsv_descriptor(
        self,
        crop: np.ndarray,
        bins: int = 12,
    ) -> Optional[np.ndarray]:
        """
        Generar descriptor HSV con máscara anti-verde.
        
        Basado en el código existente en team_clustering.py:
            - Suprimir verde del pasto
            - Calcular histogramas H y S
            - Normalizar
        
        Args:
            crop: Crop BGR
            bins: Número de bins por canal
            
        Returns:
            Vector de descriptor HSV normalizado
        """
        if crop is None or crop.size == 0:
            return None
            
        # Máscara para quitar verde del pasto
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(green_mask)
        
        if cv2.countNonZero(mask) < 16:
            return None
            
        # Histogramas H y S
        h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
        
        feat = np.concatenate([h_hist.ravel(), s_hist.ravel()]).astype(np.float32)
        s = float(feat.sum())
        if s > 0.0:
            feat /= s
            return feat
        return None
    
    def _siglip_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Generar embedding usando SigLIP2.
        
        TODO:
            - Cargar modelo SigLIP2 si no está cargado
            - Preprocesar imagen
            - Obtener embedding del encoder visual
            - Normalizar embedding
        
        Args:
            crop: Crop BGR
            
        Returns:
            Vector de embedding SigLIP2
        """
        # TODO: Implementar
        # from transformers import AutoProcessor, AutoModel
        # if self.embedder is None:
        #     self.embedder = AutoModel.from_pretrained(self.siglip_model)
        # ...
        raise NotImplementedError("TODO: Implementar embedding SigLIP2")
    
    def _normalize_team_labels(
        self,
        labels: np.ndarray,
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Normalizar etiquetas para que team 0 sea el de la izquierda.
        
        Heurística: usando las posiciones X de los primeros frames,
        asignar team 0 al equipo con menor X promedio.
        
        Args:
            labels: Etiquetas raw de K-Means
            bboxes: Bounding boxes correspondientes (opcional)
            
        Returns:
            Etiquetas normalizadas
        """
        # TODO: Implementar heurística left/right
        # (similar al código en team_clustering.py)
        raise NotImplementedError("TODO: Normalizar etiquetas de equipo")
    
    @staticmethod
    def crop_torso(
        image: np.ndarray,
        bbox: np.ndarray,
        top_ratio: float = 0.15,
        bottom_ratio: float = 0.60,
        left_ratio: float = 0.20,
        right_ratio: float = 0.80,
    ) -> Optional[np.ndarray]:
        """
        Recortar zona del torso (camiseta) de un bbox.
        
        Args:
            image: Imagen completa BGR
            bbox: Bounding box [x1, y1, x2, y2]
            top_ratio: Ratio desde arriba para inicio
            bottom_ratio: Ratio desde arriba para fin
            left_ratio: Ratio desde izquierda para inicio
            right_ratio: Ratio desde izquierda para fin
            
        Returns:
            Crop de la zona del torso
        """
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        
        if w <= 0 or h <= 0:
            return None
            
        top = y1 + int(top_ratio * h)
        bottom = y1 + int(bottom_ratio * h)
        left = x1 + int(left_ratio * w)
        right = x1 + int(right_ratio * w)
        
        # Clamp to valid range
        top = max(y1, min(top, y2 - 1))
        bottom = max(top + 1, min(bottom, y2))
        left = max(x1, min(left, x2 - 1))
        right = max(left + 1, min(right, x2))
        
        return image[top:bottom, left:right]
    
    def evaluate(
        self,
        ground_truth: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Evaluar clasificación contra ground truth.
        
        TODO:
            - Calcular accuracy
            - Calcular Silhouette Coefficient
            - Retornar métricas
        
        Args:
            ground_truth: Mapeo track_id -> team_id real
            
        Returns:
            Diccionario de métricas
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Evaluar clasificador")
