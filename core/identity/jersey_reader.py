# jersey_reader.py
"""
Lector de números de camiseta (dorsales).
Combina SmolVLM2 (generativo) con ResNet (discriminativo).
"""

from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class JerseyPrediction:
    """Predicción de número de camiseta."""
    number: int               # Número predicho (0-99)
    confidence: float         # Confianza de la predicción
    source: str               # "smolvlm", "resnet", o "ensemble"
    track_id: Optional[int] = None
    alternatives: Optional[List[Tuple[int, float]]] = None  # Top-k alternativas


class JerseyReader:
    """
    Lector de números de camiseta usando ensemble de modelos.
    
    Estrategia:
        1. SmolVLM2: modelo visión-lenguaje que "lee" el número
        2. ResNet: clasificador CNN entrenado para dígitos (0-99)
        3. Ensemble: combina ambas predicciones + contexto temporal
    
    Ventajas del ensemble:
        - SmolVLM2 maneja tipografías y ángulos variados
        - ResNet es más rápido y estable
        - Temporal: mantiene consistencia del dorsal por track
    
    Uso:
        reader = JerseyReader(mode="ensemble")
        reader.load_models()
        prediction = reader.predict(dorsal_crop)
    """
    
    MODES = ["smolvlm", "resnet", "ensemble"]
    
    def __init__(
        self,
        mode: str = "ensemble",
        smolvlm_model: str = "HuggingFaceTB/SmolVLM2-1.7B-Instruct",
        resnet_weights: Optional[Union[str, Path]] = None,
        device: str = "cuda:0",
    ):
        """
        Args:
            mode: Modo de predicción ("smolvlm", "resnet", "ensemble")
            smolvlm_model: Modelo SmolVLM2 a usar
            resnet_weights: Pesos del clasificador ResNet
            device: Dispositivo de inferencia
        """
        if mode not in self.MODES:
            raise ValueError(f"Mode debe ser uno de {self.MODES}")
            
        self.mode = mode
        self.smolvlm_model_name = smolvlm_model
        self.resnet_weights = Path(resnet_weights) if resnet_weights else None
        self.device = device
        
        self.smolvlm = None
        self.resnet = None
        
        # Caché temporal para estabilizar predicciones
        self.track_history: Dict[int, List[int]] = {}
        
    def load_models(self) -> None:
        """
        Cargar modelos según el modo.
        
        TODO:
            - Cargar SmolVLM2 si mode != "resnet"
            - Cargar ResNet si mode != "smolvlm"
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Cargar modelos de JerseyReader")
    
    def predict(
        self,
        crop: np.ndarray,
        track_id: Optional[int] = None,
    ) -> JerseyPrediction:
        """
        Predecir número de camiseta.
        
        TODO:
            - Llamar a modelo(s) según mode
            - Combinar predicciones si ensemble
            - Aplicar filtro temporal si hay track_id
        
        Args:
            crop: Crop BGR de la zona del dorsal
            track_id: ID del track para consistencia temporal
            
        Returns:
            Predicción del número de camiseta
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Predecir número de camiseta")
    
    def predict_batch(
        self,
        crops: List[np.ndarray],
        track_ids: Optional[List[int]] = None,
    ) -> List[JerseyPrediction]:
        """
        Predecir números para múltiples crops.
        
        Args:
            crops: Lista de crops BGR
            track_ids: IDs de track correspondientes
            
        Returns:
            Lista de predicciones
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Batch prediction de dorsales")
    
    def _predict_smolvlm(self, crop: np.ndarray) -> Tuple[int, float]:
        """
        Predecir usando SmolVLM2.
        
        TODO:
            - Preprocesar imagen
            - Crear prompt: "What is the jersey number in this image?"
            - Generar respuesta
            - Parsear número de la respuesta
            - Estimar confianza
        
        Args:
            crop: Crop BGR del dorsal
            
        Returns:
            Tuple (número, confianza)
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Predicción SmolVLM2")
    
    def _predict_resnet(self, crop: np.ndarray) -> Tuple[int, float, List[Tuple[int, float]]]:
        """
        Predecir usando ResNet clasificador.
        
        TODO:
            - Preprocesar imagen (resize, normalize)
            - Forward pass
            - Softmax para probabilidades
            - Retornar top-1 y top-k
        
        Args:
            crop: Crop BGR del dorsal
            
        Returns:
            Tuple (número, confianza, top_k_alternativas)
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Predicción ResNet")
    
    def _ensemble_predictions(
        self,
        smolvlm_pred: Tuple[int, float],
        resnet_pred: Tuple[int, float, List],
    ) -> Tuple[int, float]:
        """
        Combinar predicciones de ambos modelos.
        
        Estrategia:
            - Si ambos coinciden: alta confianza
            - Si difieren: preferir el de mayor confianza
            - Si ResNet tiene baja confianza pero SmolVLM alto: usar SmolVLM
        
        Args:
            smolvlm_pred: (número, confianza)
            resnet_pred: (número, confianza, alternativas)
            
        Returns:
            (número_final, confianza_final)
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Ensemble de predicciones")
    
    def _apply_temporal_filter(
        self,
        prediction: int,
        confidence: float,
        track_id: int,
        history_length: int = 5,
    ) -> Tuple[int, float]:
        """
        Aplicar filtro temporal para estabilizar predicciones.
        
        Mantiene historial de predicciones por track y
        retorna el modo (más frecuente) si hay consenso.
        
        Args:
            prediction: Predicción actual
            confidence: Confianza actual
            track_id: ID del track
            history_length: Longitud del historial a considerar
            
        Returns:
            (predicción_filtrada, confianza_ajustada)
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Filtro temporal")
    
    def evaluate(
        self,
        crops: List[np.ndarray],
        ground_truth: List[int],
    ) -> Dict[str, float]:
        """
        Evaluar rendimiento del lector.
        
        Métricas:
            - Accuracy: predicciones correctas
            - Top-3 Accuracy: correcto en top-3
            - CAR (Character Accuracy Rate): por dígito
        
        TODO:
            - Calcular métricas
            - Retornar diccionario
        
        Args:
            crops: Lista de crops de validación
            ground_truth: Números reales correspondientes
            
        Returns:
            Diccionario de métricas
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Evaluar JerseyReader")
    
    def clear_history(self) -> None:
        """Limpiar historial temporal."""
        self.track_history.clear()


# Notas de entrenamiento del ResNet para dorsales:
#
# Dataset:
#   - Crops de dorsales de SoccerNet
#   - Augmentations: rotación, brillo, blur
#   - Balance de clases (algunos números son raros)
#
# Arquitectura:
#   - ResNet-18 o ResNet-34
#   - Modificar última capa: fc(512, 100)  # 0-99
#
# Entrenamiento:
#   model = torchvision.models.resnet18(pretrained=True)
#   model.fc = nn.Linear(512, 100)
#   criterion = nn.CrossEntropyLoss()
#   optimizer = Adam(model.parameters(), lr=1e-4)
