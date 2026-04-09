import os, glob
from ultralytics import YOLO

def calculate_metrics():
    model_path = "models/yolo26.pt"
    sequence_folder = "data/tracking/SoccerNet/tracking/test/test/SNMOT-197"
    img_files = sorted(glob.glob(os.path.join(sequence_folder, "img1", "*.jpg")))
    lbl_folder = os.path.join(sequence_folder, "labels")

    print(f"Cargando {model_path}...")
    model = YOLO(model_path)
    
    TP = 0
    FP = 0
    FN = 0

    print(f"Evaluando {len(img_files)} frames...")
    
    for i, img_path in enumerate(img_files):
        if i % 100 == 0:
            print(f"Procesando frame {i}/{len(img_files)}")
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_folder, f"{base_name}.txt")
        
        gt_ball = None
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "5":
                        gt_ball = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        break
        
        results = model.predict(source=img_path, conf=0.25, imgsz=1280, verbose=False)
        boxes = results[0].boxes
        
        pred_ball = None
        if len(boxes) > 0:
            # Seleccionar la prediccion con mayor confianza si hay multiples
            conf, idx = boxes.conf.max(0)
            box = boxes.xywhn[idx]
            pred_ball = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            
        # Tolerancia de 0.02 norm dist (~38 pixels en 1920 ancho) para True Positive
        if gt_ball is not None and pred_ball is not None:
            dist = ((gt_ball[0] - pred_ball[0])**2 + (gt_ball[1] - pred_ball[1])**2)**0.5
            if dist <= 0.02: 
                TP += 1
            else:
                FP += 1
                FN += 1
        elif gt_ball is not None and pred_ball is None:
            FN += 1
        elif gt_ball is None and pred_ball is not None:
            FP += 1

    eps = 1e-6
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    print("\n========================================")
    print("MÉTRICAS CUSTOM YOLO26 (SNMOT-197 BALON)")
    print("========================================")
    print(f"Total Frames: {len(img_files)}")
    print(f"TP (Aciertos): {TP}")
    print(f"FP (Falsas Alarmas): {FP}")
    print(f"FN (No Detectados): {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("========================================")

if __name__ == "__main__":
    calculate_metrics()
