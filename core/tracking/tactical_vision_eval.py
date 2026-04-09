import os
import glob
import json
import argparse
import cv2

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Tracking Metrics against custom YOLO labels")
    parser.add_argument("--sequence_dir", type=str, required=True, help="Path to sequence directory")
    parser.add_argument("--trajectory_json", type=str, required=True, help="Path to predicted trajectory JSON")
    parser.add_argument("--output_plot", type=str, default="", help="Optional: Path to save the tracking evaluation graph (.png)")
    return parser.parse_args()

def bb_iou(box1, box2):
    # box1, box2: (cx, cy, w, h)
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = b1_area + b2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def evaluate_metrics():
    args = get_args()
    
    lbl_folder = os.path.join(args.sequence_dir, "labels")
    img_dir = os.path.join(args.sequence_dir, "img1")
    
    if not os.path.exists(lbl_folder):
        print(f"Ground Truth labels folder not found at {lbl_folder}. Skipping evaluation.")
        return

    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        print("No images found to determine video resolution.")
        return
        
    sample_img = cv2.imread(images[0])
    img_h, img_w = sample_img.shape[:2]
    
    # Load Predictions (Absolute coordinates)
    with open(args.trajectory_json, 'r') as f:
        preds = json.load(f)
        
    pred_frames = {}
    for p in preds:
        frame = int(p['frame_id'])
        x, y, w, h = p['x'], p['y'], p['w'], p['h']
        if x != -1 and y != -1 and w != -1 and h != -1:
            # Convert to normalized coordinates
            nx = x / img_w
            ny = y / img_h
            nw = w / img_w
            nh = h / img_h
            pred_frames[frame] = (nx, ny, nw, nh)

    TP = 0
    FP = 0
    FN = 0
    
    TP_50 = 0
    FP_50 = 0
    
    # Physical Tracking Metrics
    squared_errors = []
    abs_errors = []
    fragmentations = 0
    prev_node_dummy = False
    
    plot_frames = []
    plot_gt_x = []
    plot_pred_x = []
    plot_dist = []
    
    import numpy as np # Ensure numpy is available for nan
    
    print(f"Evaluando {len(images)} frames con lógica Custom...")
    
    for img_path in images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        frame = int(img_id)
        
        # Ground truth
        gt_ball = None
        lbl_file = os.path.join(lbl_folder, f"{img_id}.txt")
        if os.path.exists(lbl_file):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "5": # class 5 = ball
                        gt_ball = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        break
        
        pred_ball = pred_frames.get(frame)
        
        if gt_ball is not None and pred_ball is not None:
            # Normalized distance for classification
            dist = ((gt_ball[0] - pred_ball[0])**2 + (gt_ball[1] - pred_ball[1])**2)**0.5
            
            # Pixel distance for RMSE/MAE
            gt_px = (gt_ball[0] * img_w, gt_ball[1] * img_h)
            pred_px = (pred_ball[0] * img_w, pred_ball[1] * img_h)
            sq_err = (gt_px[0] - pred_px[0])**2 + (gt_px[1] - pred_px[1])**2
            squared_errors.append(sq_err)
            abs_errors.append(sq_err**0.5)
            
            # Fragmentation check: If we recover from a dummy state
            if prev_node_dummy:
                fragmentations += 1
            prev_node_dummy = False

            # IoU
            iou = bb_iou(gt_ball, pred_ball)
            if iou >= 0.5:
                TP_50 += 1
            else:
                FP_50 += 1
            
            # For plotting
            plot_frames.append(frame)
            plot_gt_x.append(gt_ball[0])
            plot_pred_x.append(pred_ball[0])
            plot_dist.append(dist)
            
            if dist <= 0.02: 
                TP += 1
            else:
                FP += 1
                FN += 1
        elif gt_ball is not None and pred_ball is None:
            FN += 1
            prev_node_dummy = True
            plot_frames.append(frame)
            plot_gt_x.append(gt_ball[0])
            plot_pred_x.append(np.nan)
            plot_dist.append(np.nan)
        elif gt_ball is None and pred_ball is not None:
            FP += 1
            FP_50 += 1
            prev_node_dummy = False # Not exactly a dummy, but we found a box where there shouldn't be one

    eps = 1e-6
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    map_50 = TP_50 / (TP_50 + FP_50 + eps) 
    
    rmse = (sum(squared_errors) / len(squared_errors))**0.5 if squared_errors else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    fppi = FP / len(images)

    print("\n===============================================")
    print(f" MÉTRICAS EVALUACIÓN: {os.path.basename(args.sequence_dir)}")
    print("===============================================")
    print(f"Total Frames: {len(images)}")
    print(f"F1-Score:  {f1:.4f} | mAP@50: {map_50:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
    print("-" * 47)
    print(f"RMSE (Error Píxeles): {rmse:.2f} px")
    print(f"MAE  (Error Píxeles): {mae:.2f} px")
    print(f"FPPI (Falsos (+) por imagen): {fppi:.4f}")
    print(f"Fragmentaciones de Trayectoria: {fragmentations}")
    print(f"TP_50: {TP_50} | FP_50: {FP_50} | FN: {FN}")
    print("===============================================\n")
    
    if args.output_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Ax1: Trajectory Tracking (X Coordinate)
            ax1.plot(plot_frames, plot_gt_x, 'g-', label='Ground Truth (X)', linewidth=2)
            
            # We filter out NaNs for the prediction line
            pf_valid = [f for f, x in zip(plot_frames, plot_pred_x) if not np.isnan(x)]
            px_valid = [x for x in plot_pred_x if not np.isnan(x)]
            ax1.plot(pf_valid, px_valid, 'r--', label='Viterbi Prediction (X)', linewidth=2)
            
            ax1.set_title('Ball Trajectory Tracking (Normalized X Component)')
            ax1.set_ylabel('X Coordinate (0 to 1)')
            ax1.legend()
            ax1.grid(True)
            
            # Ax2: Distance Error
            ax2.plot(plot_frames, plot_dist, 'b-', label='Euclidean Error', linewidth=1.5)
            ax2.axhline(y=0.02, color='red', linestyle=':', label='TP Tolerance Threshold (0.02)', linewidth=2)
            ax2.set_title('Normalized Euclidean Distance Error over Time')
            ax2.set_xlabel('Frame ID')
            ax2.set_ylabel('Dist Error')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(args.output_plot, dpi=300)
            plt.close()
            print(f"Plot saved to {args.output_plot}")
        except Exception as e:
            print(f"Graph generation failed: {e}. Is matplotlib installed?")

if __name__ == "__main__":
    evaluate_metrics()
