import os
import json
import glob
import cv2
import argparse
from ultralytics import YOLO, RTDETR

def get_args():
    parser = argparse.ArgumentParser(description="Phase 1: Feature Extraction (YOLOv26 + RT-DETR)")
    parser.add_argument("--sequence_dir", type=str, required=True,
                        help="Path to sequence directory containing 'img1' folder (e.g. SNMOT-197)")
    parser.add_argument("--yolo_weights", type=str, required=True,
                        help="Path to YOLO ball detection weights")
    parser.add_argument("--rtdetr_weights", type=str, default="models/rtdetr-l.pt",
                        help="Path to RT-DETR weights for player detection")
    parser.add_argument("--output_json", type=str, default="secuencia_197_detections.json",
                        help="Path to output JSON file")
    parser.add_argument("--ball_conf", type=float, default=0.05,
                        help="Confidence threshold for dense ball detection")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()

def extract_features():
    args = get_args()

    print(f"Loading YOLO for ball detection from {args.yolo_weights}...")
    ball_model = YOLO(args.yolo_weights)
    
    print(f"Loading RT-DETR for player detection from {args.rtdetr_weights}...")
    player_model = RTDETR(args.rtdetr_weights)

    img_dir = os.path.join(args.sequence_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if not images:
        print(f"Error: No images found in {img_dir}")
        return

    print(f"Found {len(images)} frames. Starting Phase 1 extraction...")
    
    # Store results frame by frame
    all_detections = []

    for img_path in images:
        frame_name = os.path.basename(img_path)
        frame_id = int(os.path.splitext(frame_name)[0])
        
        # 1. Dense Ball Detection (YOLO)
        # We save all detections with score > ball_conf (default 0.05)
        # Assuming class 0 is ball, or model is only trained for ball (classes=[0])
        ball_results = ball_model.predict(img_path, conf=args.ball_conf, imgsz=1280, verbose=False, device=args.device)
        
        ball_candidates = []
        if len(ball_results) > 0 and len(ball_results[0].boxes) > 0:
            for box in ball_results[0].boxes:
                # box.xywh returns center_x, center_y, width, height
                cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                score = float(box.conf[0].cpu().numpy())
                ball_candidates.append({
                    "x_center": cx,
                    "y_center": cy,
                    "w": w,
                    "h": h,
                    "score": score
                })
        
        # 2. Player Spatial Tracking (RT-DETR + ByteTrack)
        # Class 0 in COCO is typically 'person'
        player_results = player_model.track(img_path, classes=[0], persist=True, tracker="bytetrack.yaml", verbose=False, device=args.device)
        
        players = []
        if len(player_results) > 0 and len(player_results[0].boxes) > 0:
            for box in player_results[0].boxes:
                if box.id is not None:
                    track_id = int(box.id[0].item())
                    xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().tolist()
                    players.append({
                        "track_id": track_id,
                        "x_min": xmin,
                        "y_min": ymin,
                        "x_max": xmax,
                        "y_max": ymax
                    })
        
        # Add to JSON structure
        frame_data = {
            "frame_id": frame_id,
            "ball_candidates": ball_candidates,
            "players": players
        }
        all_detections.append(frame_data)
        
        if frame_id % 50 == 0:
            print(f"Processed frame {frame_id}/{len(images)}...")

    # Export to JSON
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_json, 'w') as f:
        json.dump(all_detections, f, indent=4)
        
    print(f"Phase 1 complete! Extracted data saved to {args.output_json}")

if __name__ == "__main__":
    extract_features()
