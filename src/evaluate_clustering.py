"""
Evaluación de Clustering (Métricas de Calidad Interna).

Calcula el Silhouette Score y el Davies-Bouldin Index para los clusters de
equipos, basándose en descriptores HSV extraídos en memoria.
"""

import os
import glob
import json
import cv2
import numpy as np

from core.clustering.team_classifier import TeamClassifier

GT_DIR = r"d:\sebastian\Tesis\VM_FOOTBALL\data\tracking\SoccerNet\tracking\test\test"
PREDS_DIR = r"d:\sebastian\Tesis\VM_FOOTBALL\outputs"

def run_clustering_eval():
    seqs = sorted(glob.glob(os.path.join(GT_DIR, "SNMOT-*")))
    if not seqs:
        print("No sequences found in", GT_DIR)
        return
        
    print("=" * 60)
    print(" EVALUACIÓN DE CLUSTERING (Silhouette & Davies-Bouldin) ")
    print("=" * 60)
    
    global_sil = []
    global_db = []
    seq_names_list = []
    
    classifier = TeamClassifier(mode="hsv", n_clusters=2)

    for seq_dir in seqs:
        seq_name = os.path.basename(seq_dir)
        pred_json = os.path.join(PREDS_DIR, seq_name, f"{seq_name}_detections.json")
        if not os.path.exists(pred_json):
            continue
            
        with open(pred_json, 'r') as f:
            preds = json.load(f)
            
        # Tomar una muestra representativa de frames para no saturar la memoria
        # (p.ej. 1 frame por cada segundo = cada 25 frames)
        frame_interval = 25
        
        crops = []
        for p_data in preds:
            if p_data["frame_id"] % frame_interval != 0:
                continue
                
            img_path = os.path.join(seq_dir, "img1", f"{p_data['frame_id']:06d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(seq_dir, "img1", f"{str(p_data['frame_id']).zfill(6)}.jpg")
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Extraer jugadores activos (se agrupan ambos equipos)
            for role in ['players', 'goalkeepers']:
                for obj in p_data.get(role, []):
                    bbox = np.array([obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']])
                    crop = TeamClassifier.crop_torso(img, bbox)
                    if crop is not None and crop.size > 0:
                        crops.append(crop)
                        
        if not crops:
            continue
            
        # Fit clustering (K-Means) on these crops
        try:
            classifier.fit(crops)
            scores = classifier.cluster_quality(crops)
            sil, db = scores["silhouette"], scores["davies_bouldin"]
            
            global_sil.append(sil)
            global_db.append(db)
            seq_names_list.append(seq_name)
            print(f"[{seq_name}] Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}")
        except Exception as e:
            print(f"[{seq_name}] Error en clustering eval: {e}")

    avg_sil = np.mean(global_sil) if global_sil else 0.0
    avg_db = np.mean(global_db) if global_db else 0.0
    
    print("=" * 60)
    print(f" PROMEDIO GLOBAL SILHOUETTE SCORE     : {avg_sil:.3f}")
    print(f" PROMEDIO GLOBAL DAVIES-BOULDIN INDEX : {avg_db:.3f}")
    print("=" * 60)
    print("Nota: Silhouette de 1.0 es el mejor. Davies-Bouldin cercano a 0.0 es el mejor.")
    
    # Persist to JSON for dashboard scripts
    import json as _json
    out_dir = r"d:\sebastian\Tesis\VM_FOOTBALL\results_final\evaluation"
    os.makedirs(out_dir, exist_ok=True)
    clustering_data = {
        "sequences": seq_names_list,
        "silhouette_scores": global_sil,
        "davies_bouldin_scores": global_db,
        "avg_silhouette": float(avg_sil),
        "avg_davies_bouldin": float(avg_db),
    }
    json_path = os.path.join(out_dir, "clustering_metrics.json")
    with open(json_path, 'w') as f:
        _json.dump(clustering_data, f, indent=2)
    print(f"[INFO] Clustering metrics saved to {json_path}")
    
    return seq_names_list, global_sil, global_db

if __name__ == "__main__":
    run_clustering_eval()
