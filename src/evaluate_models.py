"""
Evaluación de Modelos RF-DETR con pycocotools.

Este script evalúa los checkpoints de RF-DETR generados sobre el subconjunto de prueba 
utilizando las métricas oficiales de COCO.

Métricas extraídas:
- mAP@0.5:0.95 (General)
- mAP@0.50     (General)
- mAP@0.75     (Estricta)
- mAP_S        (Objetos pequeños, Área < 32^2 px)
- mAP_M        (Objetos medianos, 32^2 < Área < 96^2 px)
- mAP_L        (Objetos grandes, Área > 96^2 px)
- AR@1         (Avg Recall, maxDets=1)
- AR@10        (Avg Recall, maxDets=10)
- AR@100       (Avg Recall, maxDets=100)
- Per-class AP (AP desglosado por clase: player, goalkeeper, referee)

Uso:
  python scripts/evaluate_models.py
  python scripts/evaluate_models.py --max-images 50    # dry-run rápido
"""

import os
import sys
import glob
import json
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

from rfdetr.detr import RFDETRBase

# ==========================================
# CONFIGURACIONES
# ==========================================
RESOLUTION = 448
CONF_THRESHOLD = 0.01   # Muy bajo para permitir curva Recall-Precision completa
IMAGES_DIR = "datasets/reorganized_dataset/images/test"
LABELS_DIR = "datasets/reorganized_dataset/labels/test"
OUTPUT_DIR = "results_final/evaluation"

# Checkpoints a evaluar
MODELS_TO_EVALUATE = {
    "Best_EMA": "models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_ema.pth",
    "Best_Regular": "models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_regular.pth",
    "Best_Total": "models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth"
}

# Mapeo de taxonomía 6-clases original a 3-clases (las que predice RF-DETR)
# 0 (player_left), 1 (player_right) -> 0 (player)
# 2 (gk_left), 3 (gk_right) -> 1 (goalkeeper)
# 4 (referee) -> 2 (referee)
# 5 (ball) -> None (Ignorar)
CLASS_MAPPING = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: None
}

COCO_CATEGORIES = [
    {"id": 0, "name": "player", "supercategory": "person"},
    {"id": 1, "name": "goalkeeper", "supercategory": "person"},
    {"id": 2, "name": "referee", "supercategory": "person"}
]

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee"}


def load_dataset_to_coco_format(max_images=None):
    """Lee las anotaciones YOLO y las convierte al formato COCO en un solo dict JSON."""
    images_info = []
    annotations_info = []
    
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not img_paths:
        # Intentar también con .png
        img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {IMAGES_DIR}")
    
    if max_images and max_images < len(img_paths):
        img_paths = img_paths[:max_images]
        print(f"[DRY-RUN] Limitado a {max_images} imágenes.")
        
    print(f"Cargando ground truth para {len(img_paths)} imágenes...")
    
    # Leer una imagen para cachear dimensiones (asume todas iguales en el dataset de fútbol)
    sample_img = cv2.imread(img_paths[0])
    if sample_img is None:
        raise FileNotFoundError(f"No se pudo leer {img_paths[0]}")
    default_h, default_w = sample_img.shape[:2]
    print(f"Dimensiones detectadas: {default_w}x{default_h}")
    
    annot_id = 0
    skipped = 0
    for img_id, img_path in enumerate(img_paths):
        base_name = os.path.basename(img_path)
        
        # Usar dimensiones cacheadas en lugar de leer cada imagen (todas son del mismo video)
        w, h = default_w, default_h
        
        images_info.append({
            "id": img_id,
            "file_name": base_name,
            "width": w,
            "height": h
        })
        
        # Leer anotaciones asociadas
        label_name = os.path.splitext(base_name)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_name)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    orig_cls = int(parts[0])
                    mapped_cls = CLASS_MAPPING.get(orig_cls)
                    
                    # Ignorar clases que no entrenamos en este modelo (balón)
                    if mapped_cls is None:
                        continue
                        
                    # YOLO normalizado: x_center, y_center, width, height
                    x_c_n, y_c_n, w_n, h_n = map(float, parts[1:5])
                    
                    w_px = w_n * w
                    h_px = h_n * h
                    x_tl = (x_c_n * w) - (w_px / 2)
                    y_tl = (y_c_n * h) - (h_px / 2)
                    
                    area = w_px * h_px
                    
                    annotations_info.append({
                        "id": annot_id,
                        "image_id": img_id,
                        "category_id": mapped_cls,
                        "bbox": [round(x_tl, 2), round(y_tl, 2), round(w_px, 2), round(h_px, 2)],
                        "area": round(float(area), 2),
                        "iscrowd": 0
                    })
                    annot_id += 1
        else:
            skipped += 1

    if skipped:
        print(f"[WARN] {skipped} imágenes sin archivo de anotaciones.")
    
    print(f"Ground truth: {len(images_info)} imágenes, {annot_id} anotaciones.")
    
    coco_dataset = {
        "images": images_info,
        "annotations": annotations_info,
        "categories": COCO_CATEGORIES
    }
    
    gt_file = os.path.join(OUTPUT_DIR, "ground_truth.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(gt_file, 'w') as f:
        json.dump(coco_dataset, f)
        
    return gt_file, images_info


def evaluate_model(model_name, model_path, images_info, coco_gt):
    """Realiza inferencias y evalúa contra COCO GT."""
    print(f"\n{'='*60}")
    print(f"  Evaluando: {model_name}")
    print(f"  Pesos: {model_path}")
    print(f"  Tamaño: {os.path.getsize(model_path) / 1e6:.1f} MB")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en {model_path}.")
        return None
    
    print("Inicializando modelo...")
    t_load = time.perf_counter()
    model = RFDETRBase(pretrain_weights=model_path, resolution=RESOLUTION)
    print(f"Modelo cargado en {time.perf_counter() - t_load:.1f}s")
    
    predictions = []
    n_images = len(images_info)
    t_start = time.perf_counter()
    inference_times = []
    
    # Inferencia en todas las imágenes
    for i, img_dict in enumerate(images_info):
        img_path = os.path.join(IMAGES_DIR, img_dict["file_name"])
        pil_img = Image.open(img_path).convert("RGB")
        img_id = img_dict["id"]
        
        # Inferencia 
        try:
            t_inf = time.perf_counter()
            dets = model.predict(pil_img, threshold=CONF_THRESHOLD)
            inference_times.append(time.perf_counter() - t_inf)
            
            if isinstance(dets, list):
                if len(dets) > 0:
                    dets = dets[0]
                else: 
                    continue
        except Exception as e:
            print(f"Error procesando {img_dict['file_name']}: {e}")
            continue
            
        # Si no hay detecciones
        if not hasattr(dets, 'xyxy') or len(dets.xyxy) == 0:
            continue
            
        # Convertir sv.Detections a COCO
        for j in range(len(dets.xyxy)):
            box = dets.xyxy[j]
            conf = float(dets.confidence[j]) if dets.confidence is not None else 1.0
            cls_id = int(dets.class_id[j]) if dets.class_id is not None else 0
            
            x_tl, y_tl, x_br, y_br = box
            w_px = max(0.0, float(x_br - x_tl))
            h_px = max(0.0, float(y_br - y_tl))
            
            predictions.append({
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(float(x_tl), 2), round(float(y_tl), 2), round(w_px, 2), round(h_px, 2)],
                "score": round(conf, 5)
            })
        
        # Progreso
        if (i + 1) % 500 == 0 or (i + 1) == n_images:
            elapsed = time.perf_counter() - t_start
            avg_ms = (elapsed / (i + 1)) * 1000
            eta_s = avg_ms * (n_images - i - 1) / 1000
            print(f"  [{i+1}/{n_images}] {len(predictions)} dets | "
                  f"{avg_ms:.0f}ms/img | ETA: {eta_s/60:.1f}min")
            
    # Guardar predicciones en JSON
    dt_file = os.path.join(OUTPUT_DIR, f"predictions_{model_name}.json")
    with open(dt_file, 'w') as f:
        json.dump(predictions, f)
        
    total_time = time.perf_counter() - t_start
    avg_inf_ms = np.mean(inference_times) * 1000 if inference_times else 0
    fps = 1000 / avg_inf_ms if avg_inf_ms > 0 else 0
    
    print(f"\nInferencia completada:")
    print(f"  Total predicciones: {len(predictions)}")
    print(f"  Tiempo total: {total_time:.1f}s")
    print(f"  Promedio: {avg_inf_ms:.1f}ms/img ({fps:.1f} FPS)")
    
    if len(predictions) == 0:
        print("El modelo no hizo ninguna detección.")
        return None
        
    # pycocotools Eval
    coco_dt = coco_gt.loadRes(dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # COCOEval.stats contiene la info resumida (12 valores estándar):
    # 0  = AP @ IoU=0.50:0.95, area=all, maxDets=100
    # 1  = AP @ IoU=0.50,      area=all, maxDets=100
    # 2  = AP @ IoU=0.75,      area=all, maxDets=100
    # 3  = AP @ IoU=0.50:0.95, area=small, maxDets=100
    # 4  = AP @ IoU=0.50:0.95, area=medium, maxDets=100
    # 5  = AP @ IoU=0.50:0.95, area=large, maxDets=100
    # 6  = AR @ IoU=0.50:0.95, area=all, maxDets=1
    # 7  = AR @ IoU=0.50:0.95, area=all, maxDets=10
    # 8  = AR @ IoU=0.50:0.95, area=all, maxDets=100
    # 9  = AR @ IoU=0.50:0.95, area=small, maxDets=100
    # 10 = AR @ IoU=0.50:0.95, area=medium, maxDets=100
    # 11 = AR @ IoU=0.50:0.95, area=large, maxDets=100
    stats = coco_eval.stats
    
    # Per-class AP
    per_class_ap = {}
    for cat_id, cat_name in CLASS_NAMES.items():
        coco_eval_cls = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cls.params.catIds = [cat_id]
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        per_class_ap[cat_name] = {
            "AP@0.5:0.95": coco_eval_cls.stats[0],
            "AP@0.50": coco_eval_cls.stats[1],
        }
    
    return {
        "mAP@0.5:0.95": stats[0],
        "mAP@0.50": stats[1],
        "mAP@0.75": stats[2],
        "mAP_S": stats[3],
        "mAP_M": stats[4],
        "mAP_L": stats[5],
        "AR@1": stats[6],
        "AR@10": stats[7],
        "AR@100": stats[8],
        "AR_S": stats[9],
        "AR_M": stats[10],
        "AR_L": stats[11],
        "per_class": per_class_ap,
        "avg_inference_ms": avg_inf_ms,
        "fps": fps,
        "total_predictions": len(predictions),
        "model_size_mb": os.path.getsize(model_path) / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluación RF-DETR con métricas COCO")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limitar número de imágenes (dry-run)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  EVALUACIÓN EXTENSIVA RF-DETR - 3 CHECKPOINTS")
    print("  Métricas: pycocotools (estándar COCO oficial)")
    print("=" * 60)
    
    gt_file, images_info = load_dataset_to_coco_format(max_images=args.max_images)
    coco_gt = COCO(gt_file)
    
    results = {}
    for name, path in MODELS_TO_EVALUATE.items():
        res = evaluate_model(name, path, images_info, coco_gt)
        if res:
            results[name] = res
    
    if not results:
        print("ERROR: Ningún modelo fue evaluado exitosamente.")
        return
            
    # ============================================================
    # REPORTE FINAL
    # ============================================================
    print("\n" + "=" * 90)
    print("                           REPORTE COMPARATIVO FINAL")
    print("=" * 90)
    
    # Tabla principal
    metrics_main = ["mAP@0.5:0.95", "mAP@0.50", "mAP@0.75", "mAP_S", "mAP_M", "mAP_L"]
    metrics_recall = ["AR@1", "AR@10", "AR@100", "AR_S", "AR_M", "AR_L"]
    
    header = f"{'Métrica':<18}"
    for name in results:
        header += f" | {name:<14}"
    print(header)
    print("-" * len(header))
    
    # Métricas de precision
    for m in metrics_main:
        row = f"{m:<18}"
        vals = []
        for name in results:
            v = results[name][m]
            val_str = f"{v:.4f}" if v != -1 else "N/A"
            vals.append(v)
            row += f" | {val_str:<14}"
        # Marcar el mejor
        print(row)
    
    print("-" * len(header))
    
    # Métricas de recall
    for m in metrics_recall:
        row = f"{m:<18}"
        for name in results:
            v = results[name][m]
            val_str = f"{v:.4f}" if v != -1 else "N/A"
            row += f" | {val_str:<14}"
        print(row)
    
    print("-" * len(header))
    
    # Métricas de rendimiento
    for m_label, m_key in [("Inference (ms)", "avg_inference_ms"), ("FPS", "fps"), 
                            ("Tamaño (MB)", "model_size_mb"), ("Total Preds", "total_predictions")]:
        row = f"{m_label:<18}"
        for name in results:
            v = results[name][m_key]
            if m_key == "total_predictions":
                row += f" | {int(v):<14}"
            else:
                row += f" | {v:<14.1f}"
        print(row)
    
    # Per-class AP
    print(f"\n{'='*90}")
    print("                           AP POR CLASE")
    print(f"{'='*90}")
    for cls_name in CLASS_NAMES.values():
        row_50_95 = f"  {cls_name} AP@0.5:0.95 "
        row_50 = f"  {cls_name} AP@0.50     "
        for name in results:
            pc = results[name]["per_class"].get(cls_name, {})
            v1 = pc.get("AP@0.5:0.95", -1)
            v2 = pc.get("AP@0.50", -1)
            row_50_95 += f" | {v1:.4f}" if v1 != -1 else " | N/A   "
            row_50 += f" | {v2:.4f}" if v2 != -1 else " | N/A   "
        print(row_50_95)
        print(row_50)
    
    # Determinar ganador
    print(f"\n{'='*90}")
    print("                           VEREDICTO")
    print(f"{'='*90}")
    best_model = max(results, key=lambda k: results[k]["mAP@0.5:0.95"])
    best_score = results[best_model]["mAP@0.5:0.95"]
    print(f"  >> MEJOR MODELO (mAP@0.5:0.95): {best_model} con {best_score:.4f}")
    
    best_50 = max(results, key=lambda k: results[k]["mAP@0.50"])
    print(f"  >> MEJOR MODELO (mAP@0.50):     {best_50} con {results[best_50]['mAP@0.50']:.4f}")
    
    best_strict = max(results, key=lambda k: results[k]["mAP@0.75"])
    print(f"  >> MEJOR MODELO (mAP@0.75):     {best_strict} con {results[best_strict]['mAP@0.75']:.4f}")
    
    # Preparar el guardado a CSV
    csv_lines = ["Model,mAP@0.5:0.95,mAP@0.50,mAP@0.75,mAP_S,mAP_M,mAP_L,AR@1,AR@10,AR@100,FPS,Size_MB"]
    for name, res in results.items():
        csv_lines.append(
            f"{name},{res['mAP@0.5:0.95']:.4f},{res['mAP@0.50']:.4f},{res['mAP@0.75']:.4f},"
            f"{res['mAP_S']:.4f},{res['mAP_M']:.4f},{res['mAP_L']:.4f},"
            f"{res['AR@1']:.4f},{res['AR@10']:.4f},{res['AR@100']:.4f},"
            f"{res['fps']:.1f},{res['model_size_mb']:.1f}"
        )
        
    csv_out = os.path.join(OUTPUT_DIR, "rfdetr_evaluation_report.csv")
    with open(csv_out, "w") as f:
        f.write("\n".join(csv_lines))
        
    md_out = os.path.join(OUTPUT_DIR, "rfdetr_evaluation_report.md")
    with open(md_out, "w") as f:
        f.write("# RF-DETR Evaluation Report\n\n")
        f.write("## Métricas Principales\n\n")
        f.write("| Modelo | mAP@0.5:0.95 | mAP@0.50 | mAP@0.75 | mAP_S | mAP_M | mAP_L |\n")
        f.write("|--------|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['mAP@0.5:0.95']:.4f} | {res['mAP@0.50']:.4f} | "
                    f"{res['mAP@0.75']:.4f} | {res['mAP_S']:.4f} | {res['mAP_M']:.4f} | {res['mAP_L']:.4f} |\n")
        f.write(f"\n## Recall\n\n")
        f.write("| Modelo | AR@1 | AR@10 | AR@100 | AR_S | AR_M | AR_L |\n")
        f.write("|--------|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['AR@1']:.4f} | {res['AR@10']:.4f} | "
                    f"{res['AR@100']:.4f} | {res['AR_S']:.4f} | {res['AR_M']:.4f} | {res['AR_L']:.4f} |\n")
        f.write(f"\n## Per-class AP\n\n")
        f.write("| Modelo | Clase | AP@0.5:0.95 | AP@0.50 |\n")
        f.write("|--------|-------|:---:|:---:|\n")
        for name, res in results.items():
            for cls_name, vals in res["per_class"].items():
                f.write(f"| {name} | {cls_name} | {vals['AP@0.5:0.95']:.4f} | {vals['AP@0.50']:.4f} |\n")
        f.write(f"\n## Rendimiento\n\n")
        f.write("| Modelo | FPS | Tamaño (MB) |\n")
        f.write("|--------|:---:|:---:|\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['fps']:.1f} | {res['model_size_mb']:.1f} |\n")
        f.write(f"\n## Veredicto\n\n")
        f.write(f"**Mejor modelo (mAP@0.5:0.95):** `{best_model}` con **{best_score:.4f}**\n")
            
    # JSON completo con todos los resultados
    json_out = os.path.join(OUTPUT_DIR, "rfdetr_evaluation_full.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReportes guardados en: {OUTPUT_DIR}/")
    print(f"  - rfdetr_evaluation_report.csv")
    print(f"  - rfdetr_evaluation_report.md")
    print(f"  - rfdetr_evaluation_full.json")


if __name__ == "__main__":
    main()
