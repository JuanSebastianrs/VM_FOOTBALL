# siglip_embedder.py
"""
Generador de embeddings visuales usando SigLIP2.
"""

from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np

# TODO: Descomentar cuando se instale transformers
# import torch
# from transformers import AutoProcessor, AutoModel
# from PIL import Image


class SigLIPEmbedder:
    """
    Generador de embeddings visuales usando SigLIP2.
    
    SigLIP2 es un modelo vision-language de última generación:
        - Pérdida sigmoide en lugar de softmax
        - Preentrenamiento con captioning
        - Mejor generalización a nuevos dominios
    
    Uso para clustering de equipos:
        - Genera embeddings ricos de uniformes
        - Captura textura, color, logos
        - Más robusto que histogramas simples
    
    Modelos disponibles:
        - google/siglip2-base-patch16-224
        - google/siglip2-large-patch16-256
        - google/siglip2-so400m-patch14-384
    
    Referencia:
        https://huggingface.co/docs/transformers/model_doc/siglip
    """
    
    DEFAULT_MODEL = "google/siglip2-base-patch16-224"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        normalize: bool = True,
    ):
        """
        Args:
            model_name: Nombre del modelo en HuggingFace
            device: Dispositivo de inferencia
            normalize: Si normalizar embeddings a norma unitaria
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        
        self.model = None
        self.processor = None
        self._embedding_dim = None
        
    def load_model(self) -> None:
        """
        Cargar modelo y procesador SigLIP2.
        
        TODO:
            - Cargar AutoProcessor
            - Cargar AutoModel (solo encoder visual)
            - Mover a dispositivo
            - Poner en modo eval
        """
        # TODO: Implementar
        # self.processor = AutoProcessor.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name)
        # self.model.to(self.device)
        # self.model.eval()
        raise NotImplementedError("TODO: Cargar modelo SigLIP2")
    
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generar embedding para una imagen.
        
        TODO:
            - Convertir BGR a RGB
            - Preprocesar con processor
            - Forward pass por vision encoder
            - Extraer embedding [CLS] o pooled
            - Normalizar si corresponde
        
        Args:
            image: Imagen BGR (numpy array)
            
        Returns:
            Vector de embedding
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Generar embedding SigLIP2")
    
    def embed_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generar embeddings para múltiples imágenes.
        
        TODO:
            - Procesar en batches
            - Stack resultados
        
        Args:
            images: Lista de imágenes BGR
            batch_size: Tamaño de batch
            
        Returns:
            Array de embeddings (N x D)
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Batch embedding SigLIP2")
    
    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """
        Preprocesar imagen para SigLIP2.
        
        TODO:
            - BGR a RGB
            - Numpy a PIL Image
            - Aplicar processor
            - Retornar tensor
        """
        # TODO: Implementar
        raise NotImplementedError("TODO: Preprocesamiento SigLIP2")
    
    def get_embedding_dim(self) -> int:
        """Obtener dimensión del embedding."""
        if self._embedding_dim is None:
            # TODO: Obtener de la configuración del modelo
            raise NotImplementedError("TODO: Obtener dimensión de embedding")
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        """Verificar si el modelo está cargado."""
        return self.model is not None


# Ejemplo de uso para clustering de equipos:
#
# embedder = SigLIPEmbedder()
# embedder.load_model()
#
# # Generar embeddings para crops de jugadores
# crops = [crop_torso(img, bbox) for bbox in player_bboxes]
# embeddings = embedder.embed_batch(crops)
#
# # Clustering
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=2)
# labels = kmeans.fit_predict(embeddings)
