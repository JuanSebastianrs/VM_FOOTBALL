from ultralytics import YOLO
from pathlib import Path
import os
import yaml

def main():
    # === Ruta raíz al dataset reorganizado ===
    root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
    yaml_path = root_path / "data.yaml"
    hyp_path = root_path / "hyp_mot_soft.yaml"

    # === Verificación de archivos ===
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo YAML en: {yaml_path}")
    if not hyp_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de hiperparámetros en: {hyp_path}")

    # === Cargar hiperparámetros manualmente ===
    with open(hyp_path, 'r') as f:
        hyp_dict = yaml.safe_load(f)

    # === Cargar modelo YOLOv11n ===
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=960,
        batch=16,
        patience=45,
        name="modelo_yolo11vn_mot_soft",
        project="runs/train",
        device=0,
        exist_ok=True,
        amp=True,
        **hyp_dict  # <-- esto es suficiente
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
