from ultralytics import YOLO
from pathlib import Path
import os

# === Construir ruta al archivo data.yaml ===
root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
yaml_path = root_path / "data.yaml"  # Ajusta si guardaste el archivo en otra subcarpeta

# === Verificación del archivo YAML ===
if not yaml_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo YAML en: {yaml_path}")

# === Cargar el modelo base YOLOv8n (nano) ===
model = YOLO("yolov8n.pt")  # Usa yolov8n.pt para entrenamiento rápido

# === Entrenamiento del modelo ===
model.train(
    data=str(yaml_path),   # Convertimos a string por compatibilidad con YOLO
    epochs=100,
    imgsz=640,
    batch=16,
    name="modelo_yolo_futbol",
    project="runs/train",
    exist_ok=True
)
