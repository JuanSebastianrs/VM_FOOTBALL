# tracker.py
"""
Multi-Object Tracker para fútbol.
Integra ByteTrack para asociación de detecciones.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..detection.detector import Detection


@dataclass
class Track:
    """Representa una trayectoria de un objeto."""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    class_id: int = -1
    is_active: bool = True
    age: int = 0                    # Frames desde creación
    time_since_update: int = 0      # Frames sin detección
    
    def add_detection(self, detection: Detection) -> None:
        """Añadir nueva detección a la trayectoria."""
        self.detections.append(detection)
        self.time_since_update = 0
        self.age += 1
        
    def get_last_bbox(self) -> Optional[np.ndarray]:
        """Obtener última bbox conocida."""
        if self.detections:
            return self.detections[-1].bbox
        return None
    
    def get_trajectory(self) -> np.ndarray:
        """Obtener array de centros de bbox."""
        # TODO: Implementar
        raise NotImplementedError("TODO: Calcular trayectoria")


class MultiObjectTracker:
    """
    Tracker multiobject para fútbol usando ByteTrack.
    
    Mantiene trayectorias de:
        - Jugadores (con ID persistente)
        - Porteros
        - Árbitros
        - Balón
    
    Uso:
        tracker = MultiObjectTracker()
        for frame in video:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
    
    Referencia:
        ByteTrack: https://github.com/ifzhang/ByteTrack
    """
    
    def __init__(
        self,
        max_age: int = 30,           # Frames para eliminar track inactivo
        min_hits: int = 3,           # Detecciones mínimas para confirmar track
        iou_threshold: float = 0.3,  # Umbral IoU para asociación
    ):
        """
        Args:
            max_age: Frames máximos sin detección antes de eliminar track
            min_hits: Detecciones mínimas para track confirmado
            iou_threshold: Umbral IoU para asociar detección con track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 1
        self.frame_count: int = 0
        
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Actualizar tracks con nuevas detecciones.
        
        Algoritmo ByteTrack:
            1. Dividir detecciones en alta/baja confianza
            2. Asociar alta confianza con tracks activos (matching cascade)
            3. Asociar baja confianza con tracks no matcheados
            4. Crear nuevos tracks para detecciones sin match
            5. Eliminar tracks muy antiguos sin detección
        
        TODO:
            - Implementar asociación húngara por IoU
            - Manejar alta/baja confianza
            - Predicción Kalman para tracks perdidos
            
        Args:
            detections: Lista de detecciones del frame actual
            
        Returns:
            Lista de tracks activos
        """
        # TODO: Implementar algoritmo ByteTrack completo
        raise NotImplementedError("TODO: Implementar update de tracker")
    
    def _associate_detections(
        self,
        detections: List[Detection],
        tracks: List[Track],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Asociar detecciones con tracks existentes.
        
        TODO:
            - Calcular matriz de IoU
            - Resolver asignación húngara
            - Retornar matches, unmatched_detections, unmatched_tracks
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Asociación de detecciones")
    
    def _iou_matrix(
        self,
        detections: List[Detection],
        tracks: List[Track],
    ) -> np.ndarray:
        """
        Calcular matriz de IoU entre detecciones y tracks.
        
        TODO:
            - Calcular IoU para cada par (detección, track)
            - Retornar matriz NxM
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Calcular matriz IoU")
    
    def _create_track(self, detection: Detection) -> Track:
        """Crear nuevo track a partir de detección."""
        track = Track(
            track_id=self.next_id,
            class_id=detection.class_id,
        )
        track.add_detection(detection)
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track
    
    def get_active_tracks(self) -> List[Track]:
        """Obtener lista de tracks activos confirmados."""
        return [
            t for t in self.tracks.values()
            if t.is_active and t.age >= self.min_hits
        ]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Obtener track por ID."""
        return self.tracks.get(track_id)
    
    def reset(self) -> None:
        """Reiniciar tracker."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
