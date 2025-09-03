from ultralytics import YOLO
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
    ball_yaml = root / "ball_only" / "ball_only.yaml"
    if not ball_yaml.exists():
        raise FileNotFoundError(f"No existe el YAML del subset balón: {ball_yaml}\n"
                                f"Primero ejecuta make_ball_only_dataset.py")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(ball_yaml),
        epochs=80,
        imgsz=1280,            
        batch=12,               
        optimizer="adamw",
        lr0=0.004, lrf=0.01,
        weight_decay=5e-4,
        patience=30,
        mosaic=0.2, mixup=0.1, copy_paste=0.1, close_mosaic=10,
        hsv_h=0.005, hsv_s=0.3, hsv_v=0.3,
        fliplr=0.0, translate=0.05, scale=0.2,
        iou=0.7,
        amp=True,
        cache=False,
        workers=8,
        name="ball_det_yolo11n",
        project="runs/train",
        device=0,
        exist_ok=True,
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
