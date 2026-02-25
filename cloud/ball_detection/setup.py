"""
YOLO Ball Detection - Vertex AI Package Setup

This file tells Vertex AI what dependencies to install when running
the training job. Dependencies are ordered to prevent conflicts:
opencv-python-headless is listed LAST so that ultralytics doesn't
overwrite it with the GUI version.
"""

from setuptools import find_packages, setup

setup(
    name="yolo-ball-detection",
    version="1.0.0",
    description="YOLO v11 Ball Detection Training for Vertex AI",
    packages=find_packages(),
    install_requires=[
        # Framework
        "ultralytics>=8.1.0",
        # Config
        "PyYAML>=6.0",
        # Cloud
        "google-cloud-storage>=2.14.0",
        # Visualization
        "matplotlib>=3.7.0",
        "Pillow>=9.0.0",
        # MUST be last: prevents ultralytics from pulling opencv-python (GUI)
        "opencv-python-headless>=4.8.0",
    ],
    python_requires=">=3.10",
)
