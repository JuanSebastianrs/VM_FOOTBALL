import os
import glob
from ultralytics import YOLO

model_path = "models/yolo26.pt"
sequence_folder = "data/tracking/SoccerNet/tracking/test/test/SNMOT-197"
temp_dataset_dir = "temp_snmot197"

def ensure_correct_labels():
    print("\nPreparando dataset temporal (MANTENIENDO clase 5 para el balón)...")
    temp_lbl_dir = os.path.join(temp_dataset_dir, "labels", "val")
    os.makedirs(temp_lbl_dir, exist_ok=True)
    
    gt_labels_dir = os.path.join(sequence_folder, "labels")
    lbl_files = glob.glob(os.path.join(gt_labels_dir, "*.txt"))
    
    for src_lbl in lbl_files:
        base_name = os.path.basename(src_lbl)
        dest_lbl = os.path.join(temp_lbl_dir, base_name)
        
        ball_lines = []
        with open(src_lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                # Mantener la clase 5 (ball) tal cual
                if len(parts) >= 5 and parts[0] == "5":
                    ball_lines.append(f"5 {' '.join(parts[1:5])}\n")
                    
        with open(dest_lbl, "w") as f:
            f.writelines(ball_lines)
            
    # Crear temp.yaml con todas las clases originales
    yaml_path = os.path.join(temp_dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(temp_dataset_dir)}\n")
        f.write("train: images/val\n") 
        f.write("val: images/val\n") 
        f.write("names:\n")
        f.write("  0: player_left\n")
        f.write("  1: player_right\n")
        f.write("  2: goalkeeper_left\n")
        f.write("  3: goalkeeper_right\n")
        f.write("  4: referee\n")
        f.write("  5: ball\n")
    return yaml_path

def evaluate_only():
    print(f"Cargando modelo YOLO26 desde {model_path}...")
    model = YOLO(model_path)
    
    yaml_path = ensure_correct_labels()
    
    print("\nIniciando validación para extraer métricas del balón (clase 5)...")
    try:
        # classes=[5] asegura que solo evaluemos el rendimiento en el balon
        metrics = model.val(data=yaml_path, imgsz=1280, split='val', verbose=False, plots=False, classes=[5])
        
        # Extraer metricas para la clase evaluada
        p = metrics.results_dict.get('metrics/precision(B)', 0.0)
        r = metrics.results_dict.get('metrics/recall(B)', 0.0)
        map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        print("\n========================================")
        print("RESULTADOS DE EVALUACION YOLO26 (SNMOT-197 BALON)")
        print("========================================")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"mAP@50:    {map50:.4f}")
        print(f"mAP@50-95: {map50_95:.4f}")
        print("========================================")
        
    except Exception as e:
        print(f"\nError al calcular metricas: {e}")

if __name__ == "__main__":
    evaluate_only()
