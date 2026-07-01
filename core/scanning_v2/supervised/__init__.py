# core/scanning_v2/supervised/__init__.py
"""
Scanning V2 — Supervised Phase.

Capa de aprendizaje/evaluacion sobre la V2 heuristica. Entrena/refina un detector
SUPERVISADO de head-turn / visual scanning APROXIMADO antes de la recepcion, usando
ventanas de recepcion anotadas por humanos y features derivadas de orientaciones
aproximadas de cabeza/cuerpo.

NO es deteccion de gaze real ni mirada exacta. El modelo supervisado opera SOLO
sobre eventos de recepcion ya validados (no redefine quien recibe). Si no hay GT
humano suficiente, NO se entrena y se mantiene la heuristica V2 como fallback.
"""

from .schema import FEATURE_VERSION, LABEL_COLUMNS, training_feature_columns
from .feature_extractor import FeatureExtractor
from .dataset_builder import DatasetBuilder
from .models import build_model, SUPPORTED_MODELS
from .trainer import ScanningTrainer
from .predictor import ScanningPredictor
from .evaluator import ScanningEvaluator
from .annotation_validator import validate_annotations
from .readiness import compute_readiness

__all__ = [
    "FEATURE_VERSION", "LABEL_COLUMNS", "training_feature_columns",
    "FeatureExtractor", "DatasetBuilder", "build_model", "SUPPORTED_MODELS",
    "ScanningTrainer", "ScanningPredictor", "ScanningEvaluator",
    "validate_annotations", "compute_readiness",
]
