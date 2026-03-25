import os
import cv2
import glob
from ultralytics import YOLO

model_path = "models/yolo26.pt"
sequence_folder = "data/tracking/SoccerNet/tracking/test/test/SNMOT-197"
output_video_path = "cloud/ball_detection/results/inference_video/SNMOT-197_result.mp4"

def make_inference_video_and_evaluate():
    print(f"Cargando modelo YOLO26 desde {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Modelo no encontrado en {model_path}")
        return
        
    model = YOLO(model_path)
    
    img_dir = os.path.join(sequence_folder, "img1")
    if not os.path.exists(img_dir):
        print(f"No se encontro la carpeta de imagenes en {img_dir}")
        return

    # Buscar todas las imagenes
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not image_files:
        print("No se encontraron imagenes en la carpeta de secuencia.")
        return

    print(f"Se encontraron {len(image_files)} frames. Iniciando compilación del video y validacion...")
    
    # 1. Crear el video de inferencia frame por frame
    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape
    size = (width, height)
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    
    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"Procesando frame para video {i}/{len(image_files)}...")
            
        frame = cv2.imread(img_path)
        # Inferencia rapida
        results = model.predict(source=frame, conf=0.25, imgsz=1280, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    out.release()
    print(f"\n¡Video completado con exito! Guardado en: {output_video_path}")

    # 2. Calcular metricas mAP, Precision, Recall, F1
    print("\nPreparando dataset temporal para evaluación (filtrando solo el balón)...")
    temp_dataset_dir = "temp_snmot197"
    temp_img_dir = os.path.join(temp_dataset_dir, "images", "val")
    temp_lbl_dir = os.path.join(temp_dataset_dir, "labels", "val")
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_lbl_dir, exist_ok=True)
    
    # Copiar/crear links simbolicos a las imagenes y filtrar los labels para que clase 5 (ball) pase a ser clase 0
    import shutil
    gt_labels_dir = os.path.join(sequence_folder, "labels")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # Link o copia de imagen (para Windows copia rápida)
        dest_img = os.path.join(temp_img_dir, filename)
        if not os.path.exists(dest_img):
            shutil.copy2(img_path, dest_img)
            
        # Filtrar label
        src_lbl = os.path.join(gt_labels_dir, f"{base_name}.txt")
        dest_lbl = os.path.join(temp_lbl_dir, f"{base_name}.txt")
        
        ball_lines = []
        if os.path.exists(src_lbl):
            with open(src_lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "5": # El balon es clase 5 en SoccerNet
                        # Lo convertimos a clase 0 para el YOLO de balon
                        ball_lines.append(f"0 {' '.join(parts[1:5])}\n")
                        
        with open(dest_lbl, "w") as f:
            f.writelines(ball_lines)

    # Crear temp.yaml
    yaml_path = os.path.join(temp_dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(temp_dataset_dir)}\n")
        f.write("val: images/val\n") 
        f.write("names:\n")
        f.write("  0: ball\n")
    
    print("\nIniciando validación para extraer métricas...")
    # Ejecutar validacion usando el set como val
    try:
        metrics = model.val(data=yaml_path, imgsz=1280, split='val', verbose=False, plots=False)
        
        # Extraer
        p = metrics.results_dict.get('metrics/precision(B)', 0.0)
        r = metrics.results_dict.get('metrics/recall(B)', 0.0)
        map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        print("\n========================================")
        print("RESULTADOS DE EVALUACION YOLO26 (SNMOT-197)")
        print("========================================")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"mAP@50:    {map50:.4f}")
        print(f"mAP@50-95: {map50_95:.4f}")
        print("========================================")
        
        # Limpieza opcional (descomentar si se desea)
        # shutil.rmtree(temp_dataset_dir)
        
    except Exception as e:
        print(f"\nError al calcular metricas: {e}")

if __name__ == "__main__":
    make_inference_video_and_evaluate()
