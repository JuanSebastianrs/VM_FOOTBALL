import cv2
import os
from ultralytics import YOLO

model_path = "models/yolo_ball_data_centric.pt"
video_path = "datasets/test_match.mp4"
output_path = "cloud/ball_detection/results/inference_video/test_match_bytetrack.avi"

def run_bytetrack_baseline():
    print("==================================================")
    print(" BASELINE: YOLO Data-Centric + ByteTrack")
    print("==================================================")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return
        
    if not os.path.exists(video_path):
        print(f"[ERROR] Video no encontrado: {video_path}")
        return

    print(f"[INFO] Cargando modelo: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"[INFO] Iniciando Tracking (ByteTrack) sobre: {video_path}")
    print(f"[INFO] Espere... esto procesará todo el video.")
    
    # Run tracking using the built-in bytetrack.yaml configuration
    # We use a lower confidence to allow ByteTrack to associate blurry frames
    results = model.track(
        source=video_path,
        conf=0.15,          # Bajar conf para que ByteTrack atrape balones borrosos
        iou=0.45,           # NMS IOU
        imgsz=1280,         # Resolución original
        tracker="bytetrack.yaml", # Usar ByteTrack en lugar de BoT-SORT
        persist=True,       # Mantener IDs entre frames
        save=False,         # Lo guardaremos manualmente para controlar el codec
        verbose=False
    )
    
    print(f"[INFO] Inferencia completada. Compilando video de salida...")
    
    # Custom video writing to ensure compatibility
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    for r in results:
        annotated_frame = r.plot()
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    
    print(f"\n[OK] Video de Tracking guardado en: {output_path}")

if __name__ == "__main__":
    run_bytetrack_baseline()
