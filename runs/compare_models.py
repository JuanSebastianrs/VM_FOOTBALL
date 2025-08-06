import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSVs de resultados
y8 = pd.read_csv("runs/train/modelo_yolo_futbol/results.csv")
y11 = pd.read_csv("runs/train/modelo_yolo11vn_mot_futbol/results.csv")

# Comparar mAP@0.5
plt.figure(figsize=(10, 5))
plt.plot(y8["epoch"], y8["metrics/mAP50(B)"], label="YOLOv8n - mAP@0.5", linewidth=2)
plt.plot(y11["epoch"], y11["metrics/mAP50(B)"], label="YOLOv11n - mAP@0.5", linewidth=2)
plt.xlabel("Época")
plt.ylabel("mAP@0.5")
plt.title("Comparación de mAP@0.5 entre YOLOv8n y YOLOv11n")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
