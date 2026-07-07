# VM_FOOTBALL - Core IA
# Módulo de identificación (OCR de dorsales)

"""
Reconocimiento de dorsales de jugadores.
Implementación en producción: EfficientNet-B0 per-frame + PARSeq
fine-tuneado con fusión temporal (`jersey_identity_phase.py`,
`jersey_model.py`, `jersey_assignment.py`, `tracklet_linking.py`).
"""
