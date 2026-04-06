"""
Evaluación visual de RF-DETR entrenado sobre test sequence SNMOT-116.
Genera un video MP4 con bounding boxes de detecciones RF-DETR.

Replica el approach de test_sequence_video.py (YOLO, branch feauture/organization)
pero usando RF-DETR + supervision para anotación.
"""

import os
import glob
import sys
import time

import cv2
from PIL import Image
import supervision as sv

# ── Configuration ────────────────────────────────────────────────────────
model_path = "results_final/rfdetr/checkpoint_best_ema_3class_gcs.pth"
sequence_folder = "data/test_sequences/SNMOT-116"
output_video_path = "outputs/visualizations/SNMOT-116_rfdetr_boxes.mp4"

RESOLUTION = 448       # must match training resolution
CONF_THRESHOLD = 0.40
FPS = 25

CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
}

CLASS_COLORS = {
    0: (219, 152, 52),
    1: (37, 37, 213),
    2: (70, 190, 80),
    -1: (160, 160, 160),
}


def make_inference_video():
    from rfdetr.detr import RFDETRBase

    print(f"Cargando modelo RF-DETR desde {model_path}...")
    model = RFDETRBase(pretrain_weights=model_path, resolution=RESOLUTION)
    print(f"Modelo cargado. Resolución: {RESOLUTION}px")
    classes = list(getattr(model, "classes", []) or [])

    if not os.path.exists(sequence_folder):
        print(f"No se encontró la carpeta de secuencia en {sequence_folder}")
        return

    image_files = sorted(glob.glob(os.path.join(sequence_folder, "*.jpg")))
    if not image_files:
        print("No se encontraron imágenes en la carpeta de secuencia.")
        return

    print(f"Se encontraron {len(image_files)} frames. Iniciando inferencia...")

    # Leer primer frame para dimensiones
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"No se pudo leer el primer frame: {image_files[0]}")
        return
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), FPS, (width, height))

    total_dets = 0
    total_time = 0.0

    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"No se pudo leer el frame: {img_path}")
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Inferencia RF-DETR → supervision.Detections
        t0 = time.perf_counter()
        detections = model.predict(pil_img, threshold=CONF_THRESHOLD)
        if isinstance(detections, list):
            detections = detections[0] if detections else sv.Detections.empty()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        n_dets = len(detections)
        total_dets += n_dets
        total_time += elapsed_ms

        annotated = frame.copy()
        class_ids = detections.class_id if detections.class_id is not None else [None] * len(detections)
        confidences = detections.confidence if detections.confidence is not None else [0.0] * len(detections)

        # Dibujar cajas y etiquetas manualmente para mantener color estable por clase.
        for bbox, class_id, conf in zip(detections.xyxy, class_ids, confidences):
            cid = int(class_id) if class_id is not None else -1
            label_name = classes[cid] if 0 <= cid < len(classes) else CLASS_NAMES.get(cid, "object")
            color = CLASS_COLORS.get(cid, CLASS_COLORS[-1])

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{label_name} {float(conf):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        out.write(annotated)

        if i % 50 == 0:
            avg_ms = total_time / (i + 1)
            print(f"  Procesando frame {i}/{len(image_files)}... "
                  f"{n_dets} det | {elapsed_ms:.0f}ms | avg {avg_ms:.0f}ms")

    out.release()

    # Summary
    avg_dets = total_dets / len(image_files)
    avg_ms = total_time / len(image_files)
    print(f"\n¡Video completado con éxito! Guardado en: {output_video_path}")
    print(f"  Frames procesados: {len(image_files)}")
    print(f"  Detecciones totales: {total_dets} ({avg_dets:.1f}/frame)")
    print(f"  Tiempo promedio inferencia: {avg_ms:.1f}ms ({1000/avg_ms:.1f} FPS)")


if __name__ == "__main__":
    make_inference_video()
