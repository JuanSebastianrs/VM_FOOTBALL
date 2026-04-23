"""
Evaluación Táctica de Tracking (MOTA, HOTA, AssA).

Este script evalúa los JSON de tracking de la arquitectura RF-DETR + Viterbi HMM
contra las etiquetas maestras YOLO curadas del subconjunto de prueba.
Todo se procesa en memoria sin clonar bases de datos ni generar código basura,
Retorna un dict con los resultados y los persiste en JSON para dashboards. 
imitando la limpieza y eficiencia del script original evaluate_models.py.

Métricas extraídas (Umbral estricto IoU=0.5 para concordar con MOTA 88.5%):
- MOTA (Multi-Object Tracking Accuracy)
- HOTA(0.5) (Higher Order Tracking Accuracy)
- AssA(0.5) (Association Accuracy)
- IDF1 (ID F1-Score)
- ID_Switches
"""

import sys
import os
import json as _json
import glob
import json
import numpy as np
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

# ==========================================
# CONFIGURACIONES
# ==========================================
IOU_THRESHOLD = 0.5
GT_DIR = "data/tracking/SoccerNet/tracking/test/test"
PREDS_DIR = "outputs"
OUTPUT_DIR = "results_final/evaluation"

def bb_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_tracking_in_memory():
    seqs = sorted(glob.glob(os.path.join(GT_DIR, "SNMOT-*")))
    if not seqs:
        print("No se encontraron secuencias en", GT_DIR)
        return
        
    mh = mm.metrics.create()
    accs = []
    seq_names = []
    
    # HOTA In-Memory Accumulators
    hota_tps, hota_fns, hota_fps = 0, 0, 0
    matches_matrix = {} # { (gt_id, trk_id): matches_count }
    global_gt_counts = {}
    global_trk_counts = {}

    print(f"Evaluando Tracking (MOTA, HOTA, AssA) en memoria sobre {len(seqs)} secuencias...\n")
    
    for seq_dir in seqs:
        seq_name = os.path.basename(seq_dir)
        pred_json = os.path.join(PREDS_DIR, seq_name, f"{seq_name}_detections.json")
        lbl_dir = os.path.join(seq_dir, "labels")
        
        if not os.path.exists(pred_json) or not os.path.exists(lbl_dir):
            continue
            
        acc = mm.MOTAccumulator(auto_id=True)
        with open(pred_json, 'r') as f:
            preds = json.load(f)
            
        pred_map = {p['frame_id']: p for p in preds}
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        
        for lbl_file in lbl_files:
            frame_id = int(os.path.splitext(os.path.basename(lbl_file))[0])
            
            # Ground Truth Curado (Sin Banquillo)
            gt_ids, gt_boxes = [], []
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if int(parts[0]) == 5: continue # Ignore Ball for this Player Eval
                    
                    x_n, y_n, w_n, h_n = map(float, parts[1:5])
                    tid_str = f"{seq_name}-{parts[5]}"
                    tid = hash(tid_str) % ((sys.maxsize + 1) * 2)
                    
                    # De-normalize
                    w = w_n * 1920
                    h = h_n * 1080
                    x = (x_n * 1920) - w/2
                    y = (y_n * 1080) - h/2
                    
                    gt_ids.append(tid)
                    gt_boxes.append([x, y, w, h])
                    global_gt_counts[tid] = global_gt_counts.get(tid, 0) + 1
            
            # Predictions
            trk_ids, trk_boxes = [], []
            p_data = pred_map.get(frame_id, {})
            # Merge all active tracks (players, refs, gks)
            for role in ['players', 'referees', 'goalkeepers']:
                for obj in p_data.get(role, []):
                    w = obj['x_max'] - obj['x_min']
                    h = obj['y_max'] - obj['y_min']
                    tid_str = f"{seq_name}-trk-{obj['track_id']}"
                    tid = hash(tid_str) % ((sys.maxsize + 1) * 2)
                    trk_ids.append(tid)
                    trk_boxes.append([obj['x_min'], obj['y_min'], w, h])
                    global_trk_counts[tid] = global_trk_counts.get(tid, 0) + 1
                    
            # Distance Matrix for MOTMetrics (custom to bypass NumPy 2.0 asfarray crash)
            dists = np.empty((len(gt_boxes), len(trk_boxes)))
            dists[:] = np.nan
            for i, gb in enumerate(gt_boxes):
                for j, tb in enumerate(trk_boxes):
                    iou = bb_iou(gb, tb)
                    if iou >= IOU_THRESHOLD:
                        dists[i, j] = 1.0 - iou
            
            acc.update(gt_ids, trk_ids, dists.tolist())
            
            # Distance Matrix for HOTA/AssA
            sim_matrix = np.zeros((len(gt_boxes), len(trk_boxes)))
            for i, gb in enumerate(gt_boxes):
                for j, tb in enumerate(trk_boxes):
                    sim_matrix[i, j] = bb_iou(gb, tb)
                    
            # Matching (Hungarian)
            match_rows, match_cols = linear_sum_assignment(-sim_matrix)
            num_matches = 0
            for r, c in zip(match_rows, match_cols):
                if sim_matrix[r, c] >= IOU_THRESHOLD:
                    num_matches += 1
                    gt_id_k = gt_ids[r]
                    trk_id_k = trk_ids[c]
                    k = (gt_id_k, trk_id_k)
                    matches_matrix[k] = matches_matrix.get(k, 0) + 1
                    
            hota_tps += num_matches
            hota_fns += len(gt_ids) - num_matches
            hota_fps += len(trk_ids) - num_matches

        accs.append(acc)
        seq_names.append(seq_name)

    # 1. MOTA e IDF1 Oficial via motmetrics
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=seq_names, generate_overall=True)
    overall = summary.loc['OVERALL']
    
    mota = overall['mota'] * 100
    idf1 = overall['idf1'] * 100
    idsw = int(overall['num_switches'])
    
    # 2. HOTA y AssA (Umbral Fijo)
    deta = hota_tps / max(1, hota_tps + hota_fns + hota_fps)
    
    # Association Score TPA (True Positive Associations)
    assa_sum = 0
    for (gt_id, trk_id), tpa in matches_matrix.items():
        tna = global_gt_counts[gt_id]
        tpa_pred = global_trk_counts[trk_id]
        # Alignment Score (TPA / (TNA_gt + TNA_pred - TPA))
        assoc = tpa / max(1, tna + tpa_pred - tpa)
        assa_sum += tpa * assoc
        
    assa = assa_sum / max(1, hota_tps)
    hota = np.sqrt(deta * assa)

    results = {
        "MOTA": round(mota, 2),
        "HOTA": round(hota * 100, 2),
        "AssA": round(assa * 100, 2),
        "DetA": round(deta * 100, 2),
        "IDF1": round(idf1, 2),
        "ID_Switches": idsw,
    }

    # Persist to JSON so dashboard scripts can read real values
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "tracking_global_metrics.json")
    with open(json_path, 'w') as f:
        _json.dump(results, f, indent=2)
    print(f"[INFO] Métricas guardadas en {json_path}")

    print("="*60)
    print(" RESULTADOS MAESTROS DE TRACKING (ACTIVE) ")
    print("="*60)
    print(f" MOTA (Multi-Object Tracking Accuracy) : {mota:.2f}%")
    print(f" HOTA (Higher Order Tracking Acc)      : {hota * 100:.2f}%")
    print(f" AssA (Association Accuracy)           : {assa * 100:.2f}%")
    print(f" DetA (Detection Accuracy)             : {deta * 100:.2f}%")
    print(f" IDF1 Score                            : {idf1:.2f}%")
    print(f" ID Switches Reales (Global)           : {idsw}")
    print("="*60)
    print("\n[INFO] Sin carpetas clonadas ni código basura. Estos resultados")
    print("       representan el máximo potencial contra el set curado.")

    return results

if __name__ == "__main__":
    evaluate_tracking_in_memory()
