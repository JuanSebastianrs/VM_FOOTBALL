import os
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Rutas a los mejores pesos de cada modelo (asumiendo que ya se instalaron localmente)
models_to_evaluate = {
    "Nano_640": "cloud/ball_detection/results/exp1_nano_640/weights/last.pt",
    "Nano_1280": "cloud/ball_detection/results/exp2_nano_1280/weights/last.pt",
    "Small_640": "cloud/ball_detection/results/exp3_small_640/weights/best.pt"
}

data_yaml = "datasets/reorganized_dataset/data_local.yaml"

plt.figure(figsize=(8, 6))

for name, model_path in models_to_evaluate.items():
    print(f"Validando {name} desde {model_path}...")
    model = YOLO(model_path)
    
    # Validacion forzando unicamente la clase 5
    metrics = model.val(data=data_yaml, split="val", imgsz=640 if "640" in name else 1280, 
                        batch=16 if "640" in name else 8, workers=4, classes=5, verbose=False, plots=False)
    
    # metrics.box.curves_results es una lista. A veces index 0 es PR
    # Extraigamos la curva desde el objeto nativo
    # Internamente YOLO guarda x (Recall) y y (Precision) en metrics.box.curves
    
    try:
        # En ultalytics 8.3+, pr_curve suele estar en la posicion 0 o 1
        # La forma segura sugerida es mirar los arreglos
        recall = metrics.box.curves[0][0] # x_axis
        precision = metrics.box.curves[0][1] # y_axis
        # Graficamos la curva promedio de todas las clases (en este caso solo hay 1 clase: la 5)
        mean_precision = precision.mean(axis=0) if len(precision.shape) > 1 else precision
        
        plt.plot(recall, mean_precision, linewidth=2, label=f"{name} (mAP50: {metrics.box.map50:.3f})")
    except Exception as e:
        print(f"Error parseando curvas para {name}: {e}")
        # Fallback a plot basico
        pass

plt.title("Curva Precision-Recall (Solo Balón)", fontsize=14)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.02)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower left", fontsize=11)

save_path = "cloud/ball_detection/results/combined_pr_curve.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Grafica combinada guardada con exito en: {save_path}")
