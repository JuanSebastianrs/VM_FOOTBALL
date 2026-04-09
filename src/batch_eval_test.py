import os
import glob
import argparse
import subprocess
import matplotlib
matplotlib.use('Agg')
import re
import math
import json

def bb_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[0]+box1[2], box2[0]+box2[2]), min(box1[1]+box1[3], box2[1]+box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    iou = interArea / float(box1[2]*box1[3] + box2[2]*box2[3] - interArea + 1e-6)
    return iou
    
def dist_pt(x1, y1, x2, y2):
    return math.hypot(x1-x2, y1-y2)

def get_args():
    parser = argparse.ArgumentParser(description="Batch TacticalVision Evaluator")
    parser.add_argument("--force", action="store_true", help="Force re-processing of all phases even if JSONs exist")
    parser.add_argument("--params_yaml", type=str, default=r"d:\sebastian\Tesis\VM_FOOTBALL\best_hmm_params.yaml", help="Path to optimized parameters YAML")
    return parser.parse_args()

def run_batch():
    args_batch = get_args()
    base_test_dir = r"d:\sebastian\Tesis\VM_FOOTBALL\data\tracking\SoccerNet\tracking\test\test"
    test_sequences = sorted(glob.glob(os.path.join(base_test_dir, "SNMOT-*")))
    
    # Adicionando secuencia de train 164
    seq_164 = r"d:\sebastian\Tesis\VM_FOOTBALL\data\tracking\SoccerNet\tracking\train\train\SNMOT-164"
    if os.path.exists(seq_164):
        test_sequences.append(seq_164)

    results_dir = r"d:\sebastian\Tesis\VM_FOOTBALL\tactical_results\batch_eval"
    os.makedirs(results_dir, exist_ok=True)
    
    log_file = os.path.join(results_dir, "batch_evaluation_log.txt")
    
    global_metrics = {
        "TP": 0, "FP": 0, "FN": 0, "TP_50": 0, "FP_50": 0,
        "RMSE_sum": 0, "MAE_sum": 0, "Frag_sum": 0, "seq_count": 0,
        "Raw_TP": 0, "Raw_FP": 0, "Raw_FN": 0, "Total_Frames": 0,
        "yolo_all_detections": [], "cle_yolo_global": [], "cle_viterbi_global": [],
        "iou_yolo_global": [], "iou_viterbi_global": [],
        "occluded_frames_total": 0, "viterbi_orr_recovered": 0, "total_gt_boxes": 0
    }
    evaluated_sequences = []

    sequence_f1s = []
    sequence_raw_f1s = []
    sequence_precisions = []
    sequence_recalls = []
    sequence_raw_precisions = []
    sequence_raw_recalls = []
    all_rmses = []
    
    with open(log_file, "w") as out_txt:
        out_txt.write("BATCH EVALUATION LOG - COMPARATIVE ANALYSIS\n==========================================\n\n")

    for seq_dir in test_sequences:
        seq_name = os.path.basename(seq_dir)
        print(f"\n[ {seq_name} ] Checking previous results...")
        
        # Paths for outputs
        det_json = os.path.join(results_dir, f"{seq_name}_detections.json")
        cmc_json = os.path.join(results_dir, f"{seq_name}_cmc.json")
        traj_json = os.path.join(results_dir, f"{seq_name}_trajectory.json")
        
        # Check if GT labels have ball (class 5)
        has_gt = False
        lbl_dir = os.path.join(seq_dir, "labels")
        if os.path.exists(lbl_dir):
            for lbl in glob.glob(os.path.join(lbl_dir, "*.txt")):
                with open(lbl, "r") as f:
                    if any(line.startswith("5 ") for line in f):
                        has_gt = True
                        break
        
        if not has_gt:
            print(f"Skipping {seq_name} (No GT class 5 found)")
            with open(log_file, "a") as out_txt:
                out_txt.write(f"Sequence: {seq_name} - NO BALL GT LABELS FOUND - SKIPPED\n")
            continue
            
        # Phase 1: Extractor
        if not os.path.exists(det_json) or args_batch.force:
            print(f"Executing Phase 1 (Extractor) for {seq_name}...")
            subprocess.run(["python", r"d:\sebastian\Tesis\VM_FOOTBALL\core\detection\tactical_vision_extractor.py",
                            "--sequence_dir", seq_dir,
                            "--yolo_weights", r"models\yolo26.pt",
                            "--rtdetr_weights", "models/rtdetr-l.pt",
                            "--output_json", det_json], check=True)
        
        # Phase 2: CMC
        if not os.path.exists(cmc_json) or args_batch.force:
            subprocess.run(["python", r"d:\sebastian\Tesis\VM_FOOTBALL\core\tracking\tactical_vision_cmc.py",
                            "--sequence_dir", seq_dir,
                            "--output_json", cmc_json,
                            "--fast"], check=True)
        
        # Phase 3-5: Viterbi
        if not os.path.exists(traj_json) or args_batch.force:
            cmd_hmm = ["python", r"d:\sebastian\Tesis\VM_FOOTBALL\core\tracking\tactical_vision_hmm.py",
                       "--detections_json", det_json,
                       "--cmc_json", cmc_json,
                       "--output_json", traj_json]
            if args_batch.params_yaml and os.path.exists(args_batch.params_yaml):
                cmd_hmm.extend(["--params_yaml", args_batch.params_yaml])
            
            subprocess.run(cmd_hmm, check=True)
                        
        # Phase 8a: Raw YOLO Evaluation (Baseline)
        raw_result = subprocess.run(["python", r"d:\sebastian\Tesis\VM_FOOTBALL\core\detection\tactical_vision_yolo_eval.py",
                                     "--sequence_dir", seq_dir,
                                     "--detections_json", det_json], capture_output=True, text=True)
        
        # Phase 8b: Refined Viterbi Evaluation (Trajectory Physics)
        refined_result = subprocess.run(["python", r"d:\sebastian\Tesis\VM_FOOTBALL\core\tracking\tactical_vision_eval.py",
                                         "--sequence_dir", seq_dir,
                                         "--trajectory_json", traj_json], capture_output=True, text=True)
        
        # --- Parse Raw Metrics ---
        rf1 = re.search(r"F1-Score Crudo:\s*([0-9.]+)", raw_result.stdout)
        rp = re.search(r"Precision Cruda:\s*([0-9.]+)", raw_result.stdout)
        rr = re.search(r"Recall Crudo:\s*([0-9.]+)", raw_result.stdout)
        rtp = re.search(r"TP:\s*(\d+)", raw_result.stdout)
        rfp = re.search(r"FP:\s*(\d+)", raw_result.stdout)
        rfn = re.search(r"FN:\s*(\d+)", raw_result.stdout)
        
        raw_f1 = float(rf1.group(1)) if rf1 else 0.0
        raw_p = float(rp.group(1)) if rp else 0.0
        raw_r = float(rr.group(1)) if rr else 0.0
        if rtp: global_metrics["Raw_TP"] += int(rtp.group(1))
        if rfp: global_metrics["Raw_FP"] += int(rfp.group(1))
        if rfn: global_metrics["Raw_FN"] += int(rfn.group(1))

        # --- Parse Refined Metrics ---
        f1_m = re.search(r"F1-Score:\s*([0-9.]+)", refined_result.stdout)
        p_m = re.search(r"Precision:\s*([0-9.]+)", refined_result.stdout)
        r_m = re.search(r"Recall:\s*([0-9.]+)", refined_result.stdout)
        rmse_m = re.search(r"RMSE.*?([0-9.]+)\s*px", refined_result.stdout)
        frag_m = re.search(r"Fragmentaciones.*?(\d+)", refined_result.stdout)
        frames_m = re.search(r"Total Frames:\s*(\d+)", refined_result.stdout)
        tp_match = re.search(r"TP_50:\s*(\d+)\s*\|\s*FP_50:\s*(\d+)\s*\|\s*FN:\s*(\d+)", refined_result.stdout)
        
        seq_f1 = float(f1_m.group(1)) if f1_m else 0.0
        seq_p = float(p_m.group(1)) if p_m else 0.0
        seq_r = float(r_m.group(1)) if r_m else 0.0
        
        if tp_match:
            global_metrics["TP_50"] += int(tp_match.group(1))
            global_metrics["FP_50"] += int(tp_match.group(2))
            global_metrics["FN"] += int(tp_match.group(3))
            
        if rmse_m: 
            val_rmse = float(rmse_m.group(1))
            global_metrics["RMSE_sum"] += val_rmse
            all_rmses.append(val_rmse)
        if frag_m: global_metrics["Frag_sum"] += int(frag_m.group(1))
        if frames_m: global_metrics["Total_Frames"] += int(frames_m.group(1))

        # --- Phase 9 Metric Extraction ---

        gt_json = os.path.join(seq_dir, "ground_truth.json")
        gt_data = {}
        if os.path.exists(gt_json):
            with open(gt_json, 'r') as f: gt_data = json.load(f)
        elif has_gt:
            for lbl in glob.glob(os.path.join(lbl_dir, "*.txt")):
                fid_str = str(int(os.path.splitext(os.path.basename(lbl))[0]))
                with open(lbl, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts[0] == "5":
                            cx, cy, w, h = float(parts[1])*1920, float(parts[2])*1080, float(parts[3])*1920, float(parts[4])*1080
                            gt_data[fid_str] = {"bbox": [cx - w/2, cy - h/2, w, h], "is_occluded": False}

        if os.path.exists(det_json) and os.path.exists(traj_json):
            with open(det_json, 'r') as f: d_json = json.load(f)
            with open(traj_json, 'r') as f: t_json = json.load(f)
            vit_map = {str(t['frame_id']): t for t in t_json}
            
            for d in d_json:
                fid_str = str(d['frame_id'])
                cands = d['ball_candidates']
                gti = gt_data.get(fid_str)
                
                if gti:
                    global_metrics["total_gt_boxes"] += 1
                    gt_box = gti['bbox']
                    gt_cx, gt_cy = gt_box[0] + gt_box[2]/2, gt_box[1] + gt_box[3]/2
                    is_occ = gti.get('is_occluded', False)
                    if is_occ: global_metrics["occluded_frames_total"] += 1
                    
                    best_iou, best_dist = 0, float('inf')
                    for c in cands:
                        c_x, c_y = float(c.get('x_center', 0)), float(c.get('y_center', 0))
                        c_w, c_h = float(c.get('w', c.get('width', 20))), float(c.get('h', c.get('height', 20)))
                        c_conf = float(c.get('conf', 0))
                        p_box = [c_x - c_w/2, c_y - c_h/2, c_w, c_h]
                        iou, d_err = bb_iou(gt_box, p_box), dist_pt(c_x, c_y, gt_cx, gt_cy)
                        if iou > best_iou: best_iou = iou
                        if d_err < best_dist: best_dist = d_err
                        global_metrics["yolo_all_detections"].append({"score": c_conf, "iou": iou, "fid": fid_str, "seq": seq_name})
                        
                    global_metrics["iou_yolo_global"].append(best_iou)
                    global_metrics["cle_yolo_global"].append(best_dist if len(cands) > 0 else 200.0)
                    
                    v_node = vit_map.get(fid_str)
                    if v_node:
                        v_x, v_y = float(v_node['x']), float(v_node['y'])
                        v_w, v_h = float(v_node.get('w', 20) or 20), float(v_node.get('h', 20) or 20)
                        err = dist_pt(v_x, v_y, gt_cx, gt_cy)
                        if is_occ and err <= 20.0: global_metrics["viterbi_orr_recovered"] += 1
                        global_metrics["cle_viterbi_global"].append(err)
                        global_metrics["iou_viterbi_global"].append(bb_iou(gt_box, [v_x - v_w/2, v_y - v_h/2, v_w, v_h]))
                    else:
                        global_metrics["cle_viterbi_global"].append(200.0)
                        global_metrics["iou_viterbi_global"].append(0.0)
                else:
                    for c in cands:
                        global_metrics["yolo_all_detections"].append({"score": float(c.get('conf', 0)), "iou": 0.0, "fid": fid_str, "seq": seq_name})

        global_metrics["seq_count"] += 1
        evaluated_sequences.append(seq_name)
        sequence_f1s.append(seq_f1)
        sequence_raw_f1s.append(raw_f1)
        sequence_precisions.append(seq_p)
        sequence_recalls.append(seq_r)
        sequence_raw_precisions.append(raw_p)
        sequence_raw_recalls.append(raw_r)
        
        with open(log_file, "a") as out_txt:
            out_txt.write(f"Sequence: {seq_name}\n")
            out_txt.write("--- RAW YOLO ---\n" + raw_result.stdout)
            out_txt.write("--- REFINED TRACKER ---\n" + refined_result.stdout)
            out_txt.write("\n" + "#"*60 + "\n")
            
    # Compute global stats
    count = global_metrics["seq_count"] if global_metrics["seq_count"] > 0 else 1
    total_f = global_metrics["Total_Frames"] if global_metrics["Total_Frames"] > 0 else 1
    
    gtp_50, gfp_50, gfn = global_metrics["TP_50"], global_metrics["FP_50"], global_metrics["FN"]
    eps = 1e-6
    g_f1 = 2 * (gtp_50/(gtp_50+gfp_50+eps)) * (gtp_50/(gtp_50+gfn+eps)) / ((gtp_50/(gtp_50+gfp_50+eps)) + (gtp_50/(gtp_50+gfn+eps)) + eps)
    avg_rmse = global_metrics["RMSE_sum"] / count
    g_fppi = gfp_50 / total_f
    raw_fppi = global_metrics["Raw_FP"] / total_f

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # --- 1. Contraste F1 (Barplot Agrupado) ---
    plt.figure(figsize=(15, 7))
    x = np.arange(len(evaluated_sequences))
    plt.bar(x - 0.2, sequence_raw_f1s, 0.4, label='YOLO Crudo', color='#ff9999')
    plt.bar(x + 0.2, sequence_f1s, 0.4, label='Viterbi Refined', color='#66b3ff')
    plt.axhline(y=g_f1, color='red', linestyle='--', label=f'Promedio Global Viterbi ({g_f1:.2f})')
    plt.xticks(x, evaluated_sequences, rotation=45, ha='right', fontsize=8)
    plt.title('Gráfica 1: Contraste de F1-Score (YOLO vs Viterbi)')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "plot1_f1_contrast.png"), dpi=300, bbox_inches='tight')

    # --- 2. Reducción de Ruido FPPI (Global) ---
    plt.figure(figsize=(6, 8))
    bars = plt.bar(['YOLOv26 Crudo', 'Viterbi HMM'], [raw_fppi, g_fppi], color=['#ff9999', '#66b3ff'], width=0.6)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, h, f'{h:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.title('Gráfica 2: Reducción de Ruido (FPPI)')
    plt.ylabel('False Positives Per Frame')
    plt.savefig(os.path.join(results_dir, "plot2_fppi_reduction.png"), dpi=300)

    # --- 3. Precision vs Recall Migration (Scatter) ---
    plt.figure(figsize=(10, 8))
    plt.scatter(sequence_raw_recalls, sequence_raw_precisions, color='red', marker='x', alpha=0.6, label='YOLO Crudo')
    plt.scatter(sequence_recalls, sequence_precisions, color='blue', marker='o', alpha=0.8, label='Viterbi Refined')
    for i in range(len(evaluated_sequences)):
        plt.annotate("", xy=(sequence_recalls[i], sequence_precisions[i]), 
                     xytext=(sequence_raw_recalls[i], sequence_raw_precisions[i]),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3))
    plt.xlim(0, 1), plt.ylim(0, 1)
    plt.xlabel('Recall'), plt.ylabel('Precision')
    plt.title('Gráfica 3: Trade-off Precision vs Recall (Migración de Métricas)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "plot3_pr_migration.png"), dpi=300)

    # --- 4. RMSE Distribution (Boxplot) ---
    plt.figure(figsize=(6, 8))
    if any(r > 1000 for r in all_rmses): plt.yscale('log')
    sns.boxplot(y=all_rmses, color='#66b3ff', flierprops=dict(marker='D', markerfacecolor='red'))
    plt.title('Gráfica 4: Distribución del Error Espacial RMSE')
    plt.ylabel('RMSE Error (Pixels)')
    plt.savefig(os.path.join(results_dir, "plot4_rmse_distribution.png"), dpi=300)

    # --- 5. Temporal Masterpiece (SNMOT-116) ---
    master_seq = "SNMOT-116"
    master_dir = os.path.join(base_test_dir, master_seq)
    if os.path.exists(master_dir):
        det_json = os.path.join(results_dir, f"{master_seq}_detections.json")
        traj_json = os.path.join(results_dir, f"{master_seq}_trajectory.json")
        try:
            with open(det_json, 'r') as f: dets = json.load(f)
            with open(traj_json, 'r') as f: trajs = json.load(f)
            
            # Extract GT (First ball)
            gt_x = {}
            for lbl_path in sorted(glob.glob(os.path.join(master_dir, "labels", "*.txt"))):
                f_id = int(os.path.splitext(os.path.basename(lbl_path))[0])
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts[0] == "5": gt_x[f_id] = float(parts[1])
            
            # Extract YOLO noise
            yolo_x, yolo_f = [], []
            for d in dets:
                for c in d['ball_candidates']:
                    yolo_f.append(d['frame_id'])
                    yolo_x.append(c['x_center']/1920) 
            
            # Extract Viterbi
            vit_f = [t['frame_id'] for t in trajs]
            vit_x = [t['x']/1920 for t in trajs]

            plt.figure(figsize=(15, 6))
            plt.plot(list(gt_x.keys()), list(gt_x.values()), color='black', lw=3, label='Ground Truth')
            plt.scatter(yolo_f, yolo_x, color='red', s=5, alpha=0.2, label='YOLO Detections (Noise)')
            plt.plot(vit_f, vit_x, color='#66b3ff', lw=2, label='Viterbi Trajectory')
            
            # Occlusion Shading: find continuous missing GT or gaps
            for d in dets:
                if len(d['ball_candidates']) == 0:
                    plt.axvspan(d['frame_id'], d['frame_id']+1, color='gray', alpha=0.1)

            plt.title(f'Gráfica 5: Análisis Temporal de Trayectoria ({master_seq})')
            plt.xlabel('Frame ID'), plt.ylabel('Normalized X Coordinate')
            plt.legend(), plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(results_dir, "plot5_temporal_masterpiece.png"), dpi=300)
            plt.close()
        except Exception as e: print(f"Masterpiece error: {e}")

    # ==========================================================
    # Phase 9: Advanced Thesis Analytics (Charts 6-8 & ORR)
    # ==========================================================
    
    # --- ORR Metric ---
    orr_viterbi = (global_metrics["viterbi_orr_recovered"] / max(global_metrics["occluded_frames_total"], 1)) * 100
        
    orr_msg = f"[INFO] Occlusion Recovery Rate (ORR - Tolerancia 20px): YOLOv26=0.0% | Viterbi={orr_viterbi:.1f}%"
    print("\n" + "="*80)
    print(orr_msg)
    print("="*80 + "\n")
    with open(log_file, "a") as out_txt:
        out_txt.write(f"\n{orr_msg}\n")
        
    # --- PR Curve Calculation (YOLO) ---
    yolo_dets = sorted(global_metrics["yolo_all_detections"], key=lambda x: x["score"], reverse=True)
    matched_gts = set()
    tps, fps = [], []
    for d in yolo_dets:
        key = f"{d['seq']}_{d['fid']}"
        if d['iou'] > 0.5 and key not in matched_gts:
            tps.append(1)
            fps.append(0)
            matched_gts.add(key)
        else:
            tps.append(0)
            fps.append(1)
            
    acc_tps = np.cumsum(tps)
    acc_fps = np.cumsum(fps)
    tot_gts = global_metrics["total_gt_boxes"]
    if tot_gts > 0 and len(acc_tps) > 0:
        recalls = acc_tps / tot_gts
        precisions = acc_tps / (acc_tps + acc_fps + 1e-6)
    else:
        recalls, precisions = np.array([0]), np.array([0])
        
    map_yolo = np.trapezoid(precisions, recalls) if len(recalls) > 1 else 0.0

    # --- Gráfica 6: Curva de Precisión-Recall (mAP) ---
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='red', lw=2, label=f'YOLO PR Curve (mAP: {map_yolo:.3f})')
    plt.fill_between(recalls, precisions, alpha=0.1, color='red')
    # Use g_recall, g_precision from earlier (lines ~245) or recalculate
    v_prec = gtp_50 / (gtp_50 + gfp_50 + 1e-6)
    v_rec = gtp_50 / (gtp_50 + gfn + 1e-6)
    plt.scatter([v_rec], [v_prec], color='blue', marker='*', s=300, label='Punto Óptimo Viterbi', zorder=5)
    plt.annotate("Punto Óptimo Viterbi", (v_rec, v_prec), xytext=(v_rec-0.2, v_prec-0.1),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    plt.xlim(0, 1.05), plt.ylim(0, 1.05)
    plt.xlabel('Recall'), plt.ylabel('Precision')
    plt.title('Gráfica 6: Curva de Precisión-Recall (mAP)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "plot6_pr_curve.png"), dpi=300)
    plt.close()

    # --- Gráfica 7: Center Location Error (CLE) ---
    cle_yo = np.array(global_metrics["cle_yolo_global"])
    cle_vi = np.array(global_metrics["cle_viterbi_global"])
    th_cle = np.linspace(0, 50, 50)
    yo_cle_pct = [(np.sum(cle_yo <= th) / max(len(cle_yo), 1)) * 100 for th in th_cle]
    vi_cle_pct = [(np.sum(cle_vi <= th) / max(len(cle_vi), 1)) * 100 for th in th_cle]

    plt.figure(figsize=(8, 6))
    plt.plot(th_cle, yo_cle_pct, 'r--', lw=2, label='YOLO (Mejor caja por frame)')
    plt.plot(th_cle, vi_cle_pct, 'b-', lw=3, label='Viterbi Tracker')
    plt.xlabel('Error de Localización / Distancia Euclidiana (px)')
    plt.ylabel('Frames Exitosos (%)')
    plt.title('Gráfica 7: Center Location Error (CLE)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "plot7_cle_curve.png"), dpi=300)
    plt.close()

    # --- Gráfica 8: Success Plot (AUC del IoU) ---
    iou_yo = np.array(global_metrics["iou_yolo_global"])
    iou_vi = np.array(global_metrics["iou_viterbi_global"])
    th_iou = np.linspace(0, 1.0, 100)
    yo_iou_pct = [(np.sum(iou_yo >= th) / max(len(iou_yo), 1)) * 100 for th in th_iou]
    vi_iou_pct = [(np.sum(iou_vi >= th) / max(len(iou_vi), 1)) * 100 for th in th_iou]

    auc_yolo = np.trapezoid([p/100 for p in yo_iou_pct], th_iou)
    auc_vit = np.trapezoid([p/100 for p in vi_iou_pct], th_iou)

    plt.figure(figsize=(8, 6))
    plt.plot(th_iou, yo_iou_pct, color='red', lw=2, label=f'YOLO (Success Rate: {auc_yolo:.3f})')
    plt.plot(th_iou, vi_iou_pct, color='blue', lw=3, label=f'Viterbi (Success Rate: {auc_vit:.3f})')
    plt.xlabel('Umbral de IoU')
    plt.ylabel('Frames Exitosos (%)')
    plt.title('Gráfica 8: Success Plot (IoU)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(results_dir, "plot8_success_plot.png"), dpi=300)
    plt.close()

    plt.close('all')
    print(f"\nAll 8 thesis graphs saved to: {results_dir}")

if __name__ == "__main__":
    run_batch()
