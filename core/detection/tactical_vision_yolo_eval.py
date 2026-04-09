import os
import glob
import json
import argparse
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Raw YOLO Detections (Phase 1) baseline")
    parser.add_argument("--sequence_dir", type=str, required=True, help="Path to sequence directory")
    parser.add_argument("--detections_json", type=str, required=True, help="Path to raw detections JSON")
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

def evaluate_raw_yolo():
    args = get_args()
    
    lbl_folder = os.path.join(args.sequence_dir, "labels")
    img_dir = os.path.join(args.sequence_dir, "img1")
    
    if not os.path.exists(lbl_folder):
        print(f"GT not found. Skipping.")
        return

    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    img_sample = cv2.imread(images[0])
    img_h, img_w = img_sample.shape[:2]
    
    with open(args.detections_json, 'r') as f:
        detections = json.load(f)
        
    det_by_frame = {d['frame_id']: d['ball_candidates'] for d in detections}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    all_detections_for_map = [] # To calculate global mAP if needed, but per-frame is enough for contrast
    
    for img_path in images:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        frame_idx = int(img_id)
        
        # Ground Truth
        gt_ball = None
        lbl_file = os.path.join(lbl_folder, f"{img_id}.txt")
        if os.path.exists(lbl_file):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "5":
                        gt_ball = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        break
        
        candidates = det_by_frame.get(frame_idx, [])
        
        if gt_ball:
            # Match best candidate
            best_iou = 0
            best_idx = -1
            for i, c in enumerate(candidates):
                c_box = (c['x_center']/img_w, c['y_center']/img_h, c['w']/img_w, c['h']/img_h)
                iou = bb_iou(gt_ball, c_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= 0.5:
                total_tp += 1
                total_fp += (len(candidates) - 1) # All others are FP
            else:
                total_fn += 1
                total_fp += len(candidates)
        else:
            total_fp += len(candidates)

    eps = 1e-6
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fppi = total_fp / len(images)

    print("\n===============================================")
    print(f" MÉTRICAS YOLO CRUDO (BASELINE): {os.path.basename(args.sequence_dir)}")
    print("===============================================")
    print(f"Total Frames: {len(images)}")
    print(f"F1-Score Crudo: {f1:.4f}")
    print(f"Precision Cruda: {precision:.4f} | Recall Crudo: {recall:.4f}")
    print(f"FPPI (Ruido Crudo): {fppi:.4f}")
    print(f"TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
    print("===============================================\n")

if __name__ == "__main__":
    evaluate_raw_yolo()
