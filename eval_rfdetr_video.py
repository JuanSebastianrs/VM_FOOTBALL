"""
Evaluación visual de RF-DETR entrenado sobre test sequence SNMOT-116.
Genera un video MP4 con bounding boxes de jugadores detectados.

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
model_path = "results_final/rfdetr/checkpoint_best_ema.pth"
sequence_folder = "data/test_sequences/SNMOT-116"
output_video_path = "outputs/visualizations/SNMOT-116_rfdetr_eval.mp4"

RESOLUTION = 448       # must match training resolution
CONF_THRESHOLD = 0.40
FPS = 25


def make_inference_video():
    from rfdetr import RFDETRBase

    print(f"Cargando modelo RF-DETR desde {model_path}...")
    model = RFDETRBase(pretrain_weights=model_path, resolution=RESOLUTION)
    print(f"Modelo cargado. Resolución: {RESOLUTION}px")

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
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))

    # Supervision annotators (equivalente a results[0].plot() de YOLO)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    total_dets = 0
    total_time = 0.0

    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Inferencia RF-DETR → supervision.Detections
        t0 = time.perf_counter()
        detections = model.predict(pil_img, threshold=CONF_THRESHOLD)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        n_dets = len(detections)
        total_dets += n_dets
        total_time += elapsed_ms

        # Generar labels con confianza
        labels = [f"player {conf:.2f}" for conf in detections.confidence]

        # Anotar frame con supervision (como YOLO results[0].plot())
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Info overlay
        info = f"RF-DETR {RESOLUTION}px | Frame {i+1}/{len(image_files)} | {n_dets} det | {elapsed_ms:.0f}ms"
        cv2.rectangle(annotated, (0, 0), (width, 35), (0, 0, 0), -1)
        cv2.putText(annotated, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

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
