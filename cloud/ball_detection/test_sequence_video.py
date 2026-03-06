import cv2
import os
import glob
from ultralytics import YOLO

model_path = "models/yolo_ball_data_centric.pt"
sequence_folder = "datasets/test_seq_116"
output_video_path = "cloud/ball_detection/results/inference_video/test_seq_116_result.mp4"

def make_inference_video():
    print(f"Cargando modelo especializado desde {model_path}...")
    model = YOLO(model_path)
    
    if not os.path.exists(sequence_folder):
        print(f"No se encontro la carpeta de secuencia en {sequence_folder}")
        return

    # Buscar todas las imagenes
    image_files = sorted(glob.glob(os.path.join(sequence_folder, "*.jpg")))
    if not image_files:
        print("No se encontraron imagenes en la carpeta de secuencia.")
        return

    print(f"Se encontraron {len(image_files)} frames. Iniciando compilación del video...")
    
    # Leer el primer frame para obtener las dimensiones
    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape
    size = (width, height)
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Crear VideoWriter a 25 fps (estandar)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    
    for i, img_path in enumerate(image_files):
        if i % 50 == 0:
            print(f"Procesando frame {i}/{len(image_files)}...")
            
        frame = cv2.imread(img_path)
        
        # Realizar inferencia
        results = model.predict(source=frame, conf=0.25, imgsz=1280, verbose=False)
        
        # Plotear los bounding boxes
        annotated_frame = results[0].plot()
        
        # Escribir frame al video
        out.write(annotated_frame)

    out.release()
    print(f"\n¡Video completado con exito! Guardado en: {output_video_path}")

if __name__ == "__main__":
    make_inference_video()
