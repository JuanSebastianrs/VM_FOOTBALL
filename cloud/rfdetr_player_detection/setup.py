"""
RF-DETR Player Detection - Vertex AI Package Setup

Dependencies ordered to prevent conflicts:
- pydantic V2 listed early (CRITICAL for rfdetr)
- opencv-python-headless listed LAST (prevents GUI version override)
"""

from setuptools import find_packages, setup

setup(
    name="rfdetr-player-detection",
    version="1.0.0",
    description="RF-DETR Player Detection Training for Vertex AI",
    packages=find_packages(),
    install_requires=[
        # RF-DETR framework
        "rfdetr>=1.4.0",
        # CRITICAL: pydantic V2 required by rfdetr internals
        "pydantic>=2.5.0",
        # Transformer backbone deps
        "transformers>=4.30.0",
        "accelerate>=0.27.0",
        "timm>=0.9.0",
        # Visualization & evaluation
        "supervision>=0.25.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.0.0",
        # Config
        "PyYAML>=6.0",
        # Cloud
        "google-cloud-storage>=2.14.0",
        # Progress
        "tqdm>=4.65.0",
        # MUST be last: prevents rfdetr/ultralytics from pulling opencv-python (GUI)
        "opencv-python-headless>=4.8.0",
    ],
    python_requires=">=3.10",
)
