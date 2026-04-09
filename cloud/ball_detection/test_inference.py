import cv2
from ultralytics import YOLO
import os

model_path = "models/yolo_ball_data_centric.pt"
output_path = "cloud/ball_detection/results/inference_test.mp4"

def test_inference(video_path="datasets/test_match.mp4"):
    print(f"Cargando modelo especializado desde {model_path}...")
    model = YOLO(model_path)
    
    if not os.path.exists(video_path):
        print(f"No se encontro archivo de video en {video_path}")
        return

    print(f"Iniciando inferencia en: {video_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = model.predict(source=video_path, 
                            save=True, 
                            project="cloud/ball_detection/results", 
                            name="inference_video", 
                            exist_ok=True,
                            conf=0.25, 
                            imgsz=1280)
    
    print(f"Inferencia completada. Video guardado en: cloud/ball_detection/results/inference_video")

if __name__ == "__main__":
    test_inference()
