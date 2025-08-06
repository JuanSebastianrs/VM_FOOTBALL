import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from natsort import natsorted

# === CONFIGURACIÓN ===
secuencia = "SNMOT-116"  # Cambia aquí para otra secuencia
split = "test"  # o "train"
root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
images_path = root_path / "images" / split

# Cargar modelo YOLO entrenado
model = YOLO("runs/train/modelo_yolo_futbol/weights/best.pt")  # Ajusta si es necesario

# Filtrar y ordenar imágenes de la secuencia
image_files = natsorted([f for f in images_path.glob(f"{secuencia}_*.jpg")])

# Crear el video de salida
output_video_path = f"{secuencia}_predictionsyolov11_6class.mp4"
frame = cv2.imread(str(image_files[0]))
height, width, _ = frame.shape
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

# Dibujar predicciones en cada frame
for img_path in image_files:
    results = model.predict(source=str(img_path), save=False, conf=0.4, verbose=False)[0]
    frame = cv2.imread(str(img_path))

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = box.conf[0]
        label = f"{model.names[class_id]} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    video_writer.write(frame)

video_writer.release()
print(f"✅ Video generado: {output_video_path}")

