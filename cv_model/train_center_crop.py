from ultralytics import YOLO
from pathlib import Path
import yaml

def main():
    # === Ruta raíz al dataset recortado ===
    root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "center_crop_dataset"
    yaml_path = root_path / "data.yaml"
    hyp_path = root_path / "hyp_center_crop.yaml"

    # === Verificación de archivos ===
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset recortado en: {yaml_path}\n"
                              f"Primero ejecuta make_center_crop_dataset.py")

    # === Crear archivo de hiperparámetros optimizados para el dataset recortado ===
    hyp_config = {
        'lr0': 0.004,          # learning rate inicial
        'lrf': 0.01,           # learning rate final
        'momentum': 0.937,      # SGD momentum/Adam beta1
        'weight_decay': 0.0005, # factor de weight decay
        'warmup_epochs': 3.0,   # epochs de warmup
        'warmup_momentum': 0.8, # warmup initial momentum
        'box': 7.5,            # box loss gain
        'cls': 0.5,            # cls loss gain
        'hsv_h': 0.015,        # ajuste de tono (menor por ser zona central)
        'hsv_s': 0.4,          # ajuste de saturación
        'hsv_v': 0.3,          # ajuste de brillo
        'degrees': 2.0,        # rotación máxima (+/- deg) (reducida por ser zona central)
        'translate': 0.05,     # traducción máxima (+/- fracción)
        'scale': 0.2,          # escala máxima (+/- ganancia)
        'shear': 0.0,          # shear máximo (+/- deg) (desactivado por ser zona central)
        'perspective': 0.0,     # perspectiva (+/- fracción) (desactivada por ser zona central)
        'flipud': 0.0,         # probabilidad de flip vertical (desactivado)
        'fliplr': 0.3,         # probabilidad de flip horizontal (reducida)
        'mosaic': 0.3,         # probabilidad de mosaico (reducida)
        'mixup': 0.1,          # probabilidad de mixup
        'copy_paste': 0.0      # probabilidad de copy-paste (desactivado por ser zona central)
    }

    # Guardar hiperparámetros
    with open(hyp_path, 'w') as f:
        yaml.dump(hyp_config, f, sort_keys=False)

    # === Cargar modelo YOLOv11n ===
    model = YOLO("yolo11n.pt")

    # === Configurar y ejecutar entrenamiento ===
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=960,            # Podemos usar una resolución menor ya que las imágenes están recortadas
        batch=24,             # Batch size mayor por tener imágenes más pequeñas
        optimizer="adamw",    # Optimizador AdamW
        hyp=str(hyp_path),   # Archivo de hiperparámetros personalizado
        patience=20,         # Early stopping
        amp=True,           # Mixed precision training
        device=0,           # GPU
        workers=8,          # Número de workers para carga de datos
        project="runs/train",
        name="ball_det_center_crop",  # Nombre distintivo para este experimento
        exist_ok=True,
        close_mosaic=10,    # Desactivar mosaico en las últimas épocas
        cache=True,         # Cachear imágenes en RAM
        save=True,          # Guardar checkpoints
        save_period=10,     # Guardar cada 10 épocas
        plots=True,         # Generar gráficas de entrenamiento
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()