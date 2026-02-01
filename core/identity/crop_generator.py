# crop_generator.py
"""
Generador de crops de dorsales y torsos desde videos procesados.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
import numpy as np
import cv2
import json
from dataclasses import dataclass


@dataclass
class CropMetadata:
    """Metadatos de un crop generado."""
    crop_path: Path
    source_frame: int
    track_id: int
    bbox: List[float]         # [x1, y1, x2, y2]
    crop_type: str            # "torso" o "dorsal"
    team_id: Optional[int] = None
    jersey_number: Optional[int] = None  # Ground truth si existe


class DorsalCropGenerator:
    """
    Generador de crops de dorsales para entrenamiento y análisis.
    
    Extrae crops de:
        - Torso: para clustering de equipos
        - Dorsal: para OCR de números
    
    Uso:
        generator = DorsalCropGenerator(output_dir="crops/")
        generator.process_sequence(
            frames_dir="datasets/images/train",
            frame_assignments="outputs/team_clustering/SNMOT-116_frame_assignments.json"
        )
    """
    
    def __init__(
        self,
        output_dir: Path,
        min_crop_size: Tuple[int, int] = (32, 32),
        target_size: Tuple[int, int] = (64, 64),
    ):
        """
        Args:
            output_dir: Directorio de salida para crops
            min_crop_size: Tamaño mínimo para considerar un crop válido
            target_size: Tamaño de redimensionado de los crops
        """
        self.output_dir = Path(output_dir)
        self.min_crop_size = min_crop_size
        self.target_size = target_size
        
        # Crear subdirectorios
        self.torso_dir = self.output_dir / "torso"
        self.dorsal_dir = self.output_dir / "dorsal"
        
    def setup_directories(self) -> None:
        """Crear estructura de directorios."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.torso_dir.mkdir(exist_ok=True)
        self.dorsal_dir.mkdir(exist_ok=True)
        
    def process_sequence(
        self,
        frames_dir: Path,
        frame_assignments_json: Path,
        sequence_id: str,
        sample_rate: int = 1,
    ) -> List[CropMetadata]:
        """
        Procesar una secuencia y generar crops.
        
        TODO:
            - Cargar frame assignments
            - Para cada frame:
                - Cargar imagen
                - Para cada jugador:
                    - Generar crop de torso
                    - Generar crop de dorsal (zona trasera)
                    - Guardar con metadatos
        
        Args:
            frames_dir: Directorio con los frames
            frame_assignments_json: JSON con asignaciones de tracking
            sequence_id: ID de la secuencia (e.g., "SNMOT-116")
            sample_rate: Procesar 1 de cada N frames
            
        Returns:
            Lista de metadatos de crops generados
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Procesar secuencia para crops")
    
    def extract_torso_crop(
        self,
        image: np.ndarray,
        bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        Extraer crop de torso (zona de camiseta frontal).
        
        Región: 15-60% vertical, 20-80% horizontal del bbox.
        
        Args:
            image: Imagen completa BGR
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Crop de torso o None si muy pequeño
        """
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        
        if w <= 0 or h <= 0:
            return None
            
        # Zona del torso
        top = y1 + int(0.15 * h)
        bottom = y1 + int(0.60 * h)
        left = x1 + int(0.20 * w)
        right = x1 + int(0.80 * w)
        
        crop = image[top:bottom, left:right]
        
        if crop.shape[0] < self.min_crop_size[1] or crop.shape[1] < self.min_crop_size[0]:
            return None
            
        return cv2.resize(crop, self.target_size)
    
    def extract_dorsal_crop(
        self,
        image: np.ndarray,
        bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        Extraer crop de dorsal (zona de espalda).
        
        Región similar al torso pero ajustada para números.
        En la práctica, sin orientación, es igual al torso.
        Con pose estimation, se podría seleccionar solo espaldas.
        
        Args:
            image: Imagen completa BGR
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Crop de dorsal o None si muy pequeño
        """
        # Por ahora igual que torso
        # TODO: Integrar pose estimation para distinguir frente/espalda
        return self.extract_torso_crop(image, bbox)
    
    def save_crop(
        self,
        crop: np.ndarray,
        crop_type: str,
        sequence_id: str,
        frame_idx: int,
        track_id: int,
    ) -> Path:
        """
        Guardar crop a disco.
        
        Formato nombre: {sequence}_{frame:06d}_{track:04d}.jpg
        
        Args:
            crop: Imagen a guardar
            crop_type: "torso" o "dorsal"
            sequence_id: ID de secuencia
            frame_idx: Índice del frame
            track_id: ID del track
            
        Returns:
            Path del archivo guardado
        """
        if crop_type == "torso":
            output_dir = self.torso_dir
        else:
            output_dir = self.dorsal_dir
            
        filename = f"{sequence_id}_{frame_idx:06d}_{track_id:04d}.jpg"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), crop)
        return output_path
    
    def generate_metadata_csv(
        self,
        crops_metadata: List[CropMetadata],
        output_path: Path,
    ) -> None:
        """
        Generar CSV con metadatos de todos los crops.
        
        TODO:
            - Escribir CSV con columnas:
              crop_path, source_frame, track_id, bbox, crop_type, team_id, jersey_number
        
        Args:
            crops_metadata: Lista de metadatos
            output_path: Path del CSV de salida
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Generar CSV de metadatos")
    
    def stream_crops(
        self,
        frames_dir: Path,
        frame_assignments_json: Path,
        sequence_id: str,
    ) -> Generator[Tuple[np.ndarray, CropMetadata], None, None]:
        """
        Generador streaming de crops (no guarda a disco).
        
        Útil para procesamiento en memoria o pipeline.
        
        TODO:
            - Yield (crop, metadata) para cada detección
        
        Yields:
            Tuple (crop_image, metadata)
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Streaming de crops")


# Uso recomendado:
#
# 1. Generar crops offline para entrenamiento:
#    generator = DorsalCropGenerator(output_dir=Path("outputs/crops"))
#    generator.setup_directories()
#    metadata = generator.process_sequence(
#        frames_dir=Path("datasets/reorganized_dataset/images/train"),
#        frame_assignments_json=Path("outputs/team_clustering/SNMOT-116_frame_assignments.json"),
#        sequence_id="SNMOT-116"
#    )
#    generator.generate_metadata_csv(metadata, Path("outputs/crops/metadata.csv"))
#
# 2. Streaming para inference:
#    for crop, meta in generator.stream_crops(...):
#        prediction = jersey_reader.predict(crop, track_id=meta.track_id)
