# VM_FOOTBALL - Core IA

"""
Core de IA para análisis de fútbol.

Módulos:
    - detection: Detección de objetos (YOLO26 balón, RF-DETR jugadores)
    - tracking: Viterbi HMM, CMC y evaluación de tracking
    - clustering: Clasificación de equipos (K-Means sobre HSV, GK role-aware)
    - identity: Dorsales (per-frame EfficientNet + PARSeq, fusión temporal)
    - mapping: Calibración PnLCalib y mapeo 2D métrico
    - segmentation: SAM2
    - analytics: Métricas físicas, forma de equipo, comparación vs GT
    - events / game_state: Recepciones de pase y estado de juego
    - scanning / scanning_v2: Visual scanning (orientación + head-turn)

El orquestador del pipeline es `src/tactical_vision_pipeline.py`.
"""

__version__ = "0.1.0"
