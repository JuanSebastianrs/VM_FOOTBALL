# segmenter.py
"""
Segmentador basado en SAM2 (Segment Anything Model 2).
Segmentación con memoria temporal para video.
"""

from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple
import numpy as np

from ..detection.detector import Detection


class SAM2Segmenter:
    """
    Segmentador usando SAM2 (Meta).
    
    SAM2 extiende SAM con memoria temporal para video:
        - Mantiene coherencia de máscaras entre frames
        - Propaga identidades automáticamente
        - Robusto a oclusiones temporales
    
    Modo de uso en VM_FOOTBALL:
        - Recibe bboxes de detección como prompts
        - Genera máscaras de segmentación por instancia
        - Mantiene IDs consistentes en el tiempo
    
    Instalación:
        pip install segment-anything-2
        # O desde source para última versión
        
    Referencia:
        https://github.com/facebookresearch/sam2
    """
    
    def __init__(
        self,
        model_type: str = "sam2_hiera_large",
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cuda:0",
    ):
        """
        Args:
            model_type: Tipo de modelo SAM2 ("sam2_hiera_tiny/small/base/large")
            weights_path: Ruta a pesos fine-tuneados (opcional)
            device: Dispositivo de inferencia
        """
        self.model_type = model_type
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device
        self.model = None
        self.predictor = None
        
        # Estado de memoria para video
        self._memory_bank = {}
        self._current_frame_idx = 0
        
    def load_model(self) -> None:
        """
        Cargar modelo SAM2.
        
        TODO:
            - Cargar checkpoint SAM2
            - Inicializar predictor de video
            - Cargar pesos fine-tuneados si existen
        """
        # TODO: Implementar
        # from sam2 import build_sam2_video_predictor
        # self.predictor = build_sam2_video_predictor(
        #     self.model_type,
        #     self.weights_path,
        #     device=self.device
        # )
        raise NotImplementedError("TODO: Cargar modelo SAM2")
    
    def segment_frame(
        self,
        image: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
    ) -> List[Detection]:
        """
        Segmentar objetos en un frame usando bboxes como prompts.
        
        TODO:
            - Inicializar estado si es primer frame
            - Usar bboxes como prompts para SAM2
            - Generar máscaras de segmentación
            - Propagar IDs desde memoria
            - Actualizar detecciones con máscaras
        
        Args:
            image: Frame BGR
            detections: Detecciones con bboxes
            frame_idx: Índice del frame en la secuencia
            
        Returns:
            Detecciones con máscaras añadidas
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Segmentación de frame")
    
    def segment_video(
        self,
        frames: List[np.ndarray],
        detections_per_frame: List[List[Detection]],
    ) -> List[List[Detection]]:
        """
        Segmentar video completo con memoria temporal.
        
        TODO:
            - Procesar frames secuencialmente
            - Mantener memoria entre frames
            - Propagar máscaras e IDs
        
        Args:
            frames: Lista de frames BGR
            detections_per_frame: Detecciones por frame
            
        Returns:
            Detecciones con máscaras por frame
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Segmentación de video")
    
    def _init_video_state(self, first_frame: np.ndarray) -> None:
        """
        Inicializar estado para nuevo video.
        
        TODO:
            - Resetear memoria
            - Inicializar predictor con primer frame
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Inicializar estado video")
    
    def _propagate_masks(
        self,
        current_masks: Dict[int, np.ndarray],
        previous_masks: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        Propagar máscaras del frame anterior al actual.
        
        Usa la memoria temporal de SAM2 para mantener
        consistencia de IDs y formas.
        
        TODO:
            - Matching de máscaras por IoU
            - Usar memoria de SAM2 para propagación
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Propagar máscaras")
    
    def get_mask_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> float:
        """Calcular IoU entre dos máscaras binarias."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union
    
    def reset(self) -> None:
        """Resetear estado de memoria."""
        self._memory_bank.clear()
        self._current_frame_idx = 0


# Notas de Fine-tuning SAM2:
#
# Para adaptar SAM2 al dominio del fútbol:
#
# 1. Preparar dataset con máscaras:
#    - Usar SoccerNet o anotar manualmente
#    - Formato: image + mask por clase
#
# 2. Fine-tuning del decoder (mantener encoder frozen):
#    from sam2 import SAM2
#    model = SAM2.from_pretrained("sam2_hiera_large")
#    model.image_encoder.requires_grad_(False)  # Freeze encoder
#    # Solo entrenar decoder
#    trainer.fit(model, train_dataloader)
#
# 3. Guardar pesos:
#    model.save_pretrained("models/sam2_football")
