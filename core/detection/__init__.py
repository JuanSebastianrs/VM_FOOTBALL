# VM_FOOTBALL - Core IA
# Módulo de detección de objetos

"""
Detectores de objetos para fútbol.
Implementaciones: YOLO y RF-DETR.
"""

from .detector import FootballDetector, Detection

# RF-DETR detector (requires rfdetr package)
try:
    from .rfdetr_detector import RFDETRDetector
except ImportError:
    RFDETRDetector = None  # rfdetr not installed

# YOLO detector (requires ultralytics package)
try:
    from .yolo_detector import YOLODetector
except (ImportError, ModuleNotFoundError):
    YOLODetector = None  # ultralytics not installed

__all__ = ["FootballDetector", "Detection", "RFDETRDetector", "YOLODetector"]
