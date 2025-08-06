from ultralytics import YOLO
from pathlib import Path
import os

def main():
    # === Ruta raíz al dataset reorganizado ===
    root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
    yaml_path = root_path / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo YAML en: {yaml_path}")

    # === Cargar modelo YOLOv11n ===
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=832,               
        batch=16,                
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        optimizer="adamw",
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        translate=0.1,
        scale=0.5,
        iou=0.7,
        patience=45,
        close_mosaic=10,
        name="modelo_yolo11vn_4class",
        project="runs/train",
        device=0,
        exist_ok=True,
        amp=True               
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
