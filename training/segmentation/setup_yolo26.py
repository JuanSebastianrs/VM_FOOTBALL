"""
YOLO26 Ball Detection - Vertex AI Package Setup

Dependencies for training YOLO26 on Vertex AI A100 GPUs.
ultralytics>=8.3.0 is REQUIRED for YOLO26 model support
(MuSGD optimizer, ProgLoss, STAL, yolo26n-p2 architecture).

opencv-python-headless is listed LAST so ultralytics doesn't
overwrite it with the GUI version (crashes in headless containers).
"""

from setuptools import find_packages, setup

setup(
    name="yolo26-ball-detection",
    version="1.0.0",
    description="YOLO26 Ball Detection Training for Vertex AI (A100 GPU)",
    packages=find_packages(),
    install_requires=[
        # Framework — YOLO26 requires latest ultralytics
        "ultralytics>=8.3.0",
        # Config
        "PyYAML>=6.0",
        # Cloud
        "google-cloud-storage>=2.14.0",
        "python-json-logger",  # Required by Vertex AI container logging
        # Visualization
        "matplotlib>=3.7.0",
        "Pillow>=9.0.0",
        # MUST be last: prevents ultralytics from pulling opencv-python (GUI)
        "opencv-python-headless>=4.8.0",
    ],
    python_requires=">=3.10",
)
