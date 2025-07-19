from ultralytics import YOLO
from pathlib import Path
import os

def main():
    root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
    yaml_path = root_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo YAML en: {yaml_path}")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=16,
        name="modelo_yolo_futbol",
        project="runs/train",
        exist_ok=True,
        device=0
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
