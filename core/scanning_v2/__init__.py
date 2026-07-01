# core/scanning_v2/__init__.py
"""
Scanning V2 — estimacion de orientacion visual APROXIMADA y deteccion heuristica
de head-turn/scanning antes de recepcion. NO es deteccion exacta de mirada.
"""
from .scanning_pipeline_v2 import ScanningPipelineV2
from .schema import HEAD_POSE_COLUMNS, SCANNING_V2_COLUMNS

__all__ = ["ScanningPipelineV2", "HEAD_POSE_COLUMNS", "SCANNING_V2_COLUMNS"]
