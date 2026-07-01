# core/events/__init__.py
"""Eventos de pase/recepcion (V2): GT-first, heuristica conservadora, sin arbitros."""
from .event_ground_truth_adapter import load_event_ground_truth
from .pass_reception_detector import PassReceptionDetector
from .possession_estimator import PossessionEstimator, PossessionFrame
from .schema import EVENT_COLUMNS, REJECTION_COLUMNS, REJECTION_REASONS, SOURCE_PRIORITY

__all__ = [
    "load_event_ground_truth", "PassReceptionDetector", "PossessionEstimator",
    "PossessionFrame", "EVENT_COLUMNS", "REJECTION_COLUMNS", "REJECTION_REASONS",
    "SOURCE_PRIORITY",
]
