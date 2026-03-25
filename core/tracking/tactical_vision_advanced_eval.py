import json
import argparse
import numpy as np
import cv2
import os

def get_args():
    parser = argparse.ArgumentParser(description="Advanced TacticalVision Metrics (Thesis Defense)")
    parser.add_argument("--gt_json", type=str, required=True, help="Path to CVAT / Custom Ground Truth JSON")
    parser.add_argument("--pred_json", type=str, required=True, help="Path to Viterbi Trajectory JSON")
    parser.add_argument("--img_w", type=int, default=1920, help="Image Width")
    parser.add_argument("--img_h", type=int, default=1080, help="Image Height")
    parser.add_argument("--tolerance_pixels", type=float, default=38.0, help="Distance in pixels to be considered TP")
    parser.add_argument("--masks_dir", type=str, default="", help="Optional: Dir with predicted SAM2 binary mask PNGs")
    parser.add_argument("--gt_masks_dir", type=str, default="", help="Optional: Dir with Ground Truth binary mask PNGs")
    return parser.parse_args()

def evaluate_advanced_metrics():
    args = get_args()
    
    # 1. Load Viterbi predictions
    with open(args.pred_json, 'r') as f:
        preds = json.load(f)
        
    pred_frames = {}
    for p in preds:
        frame_id = int(p['frame_id'])
        x, y = p['x'], p['y'] # Absolute center pixels
        if x != -1 and y != -1:
            pred_frames[frame_id] = (x, y, p.get('is_dummy', False))
            
    # 2. Load Ground Truth Object
    # Format expected: {"frame_id": {"x_center": float, "y_center": float, "occluded": bool}}
    with open(args.gt_json, 'r') as f:
        gt_data = json.load(f)
        
    # --- Bounding Box / Trajectory Metrics ---
    TP = 0
    FP = 0
    FN = 0
    
    squared_errors_all = []
    squared_errors_occ = []
    
    for frame_str, gt_info in gt_data.items():
        frame_id = int(frame_str)
        gt_x = gt_info['x_center'] * args.img_w if gt_info['x_center'] <= 1.0 else gt_info['x_center']
        gt_y = gt_info['y_center'] * args.img_h if gt_info['y_center'] <= 1.0 else gt_info['y_center']
        is_occluded = gt_info.get('occluded', False)
        
        if frame_id in pred_frames:
            pred_x, pred_y, pred_dummy = pred_frames[frame_id]
            
            # Distance
            dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            
            if dist <= args.tolerance_pixels:
                TP += 1
                squared_errors_all.append(dist**2)
                if is_occluded:
                    squared_errors_occ.append(dist**2)
            else:
                FN += 1
                FP += 1
        else:
            FN += 1

    # Check for False Positives where Viterbi predicted but GT has no ball
    gt_frame_ids = [int(k) for k in gt_data.keys()]
    for p_frame in pred_frames.keys():
        if p_frame not in gt_frame_ids:
            # If the user annotated all 300 frames perfectly and says "no ball here", this is an FP.
            FP += 1

    eps = 1e-6
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    rmse_all = np.sqrt(np.mean(squared_errors_all)) if squared_errors_all else 0.0
    ata_occ = np.sqrt(np.mean(squared_errors_occ)) if squared_errors_occ else 0.0

    print("="*50)
    print("MÉTRICAS TACTICALVISION (DEFENSA DE TESIS)")
    print("="*50)
    print("--- 1. TRAYECTORIA Y TRACKING (VITERBI) ---")
    print(f"Total Frames Evaluados: {len(gt_data)}")
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nRMSE Trayectoria Global: {rmse_all:.2f} pixeles")
    print(f"ATA (RMSE en Oclusión):  {ata_occ:.2f} pixeles")
    
    # --- Segmentation Metrics ---
    if args.masks_dir and args.gt_masks_dir:
        iou_list = []
        dice_list = []
        
        pred_masks = sorted(os.listdir(args.masks_dir))
        for mask_name in pred_masks:
            pred_path = os.path.join(args.masks_dir, mask_name)
            gt_path = os.path.join(args.gt_masks_dir, mask_name)
            
            if os.path.exists(gt_path):
                pred_m = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                gt_m = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                
                # Binarize
                pred_bin = (pred_m > 127).astype(np.uint8)
                gt_bin = (gt_m > 127).astype(np.uint8)
                
                intersection = np.logical_and(pred_bin, gt_bin).sum()
                union = np.logical_or(pred_bin, gt_bin).sum()
                
                if union > 0:
                    iou = intersection / union
                    iou_list.append(iou)
                    
                dice_denom = pred_bin.sum() + gt_bin.sum()
                if dice_denom > 0:
                    dice = (2.0 * intersection) / dice_denom
                    dice_list.append(dice)
                    
        mean_iou = np.mean(iou_list) if iou_list else 0.0
        mean_dice = np.mean(dice_list) if dice_list else 0.0
        
        print("\n--- 2. SEGMENTACIÓN SEMÁNTICA (SAM 2) ---")
        print(f"Frames con máscara evaluados: {len(iou_list)}")
        print(f"IoU (Índice Jaccard): {mean_iou:.4f}")
        print(f"Dice Coefficient:     {mean_dice:.4f}")
    print("="*50)
    
    if args.output_plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot of Errors
            ax.plot(range(len(squared_errors_all)), np.sqrt(squared_errors_all), 'bo-', label='Error Euclidean', alpha=0.6)
            
            ax.axhline(y=args.tolerance_pixels, color='r', linestyle='--', label=f'TP Threshold ({args.tolerance_pixels} px)')
            ax.set_title('Error de Tracking por Frame (RMSE)')
            ax.set_xlabel('Evaluated Frame Index')
            ax.set_ylabel('Euclidean Distance Error (px)')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(args.output_plot, dpi=300)
            plt.close()
            print(f"Plot saved to {args.output_plot}")
        except Exception as e:
            print(f"Graph generation failed: {e}. Is matplotlib installed?")

if __name__ == "__main__":
    evaluate_advanced_metrics()
