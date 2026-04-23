import os
import json
import glob
import cv2
import argparse
from ultralytics import YOLO
from PIL import Image

try:
    from rfdetr.detr import RFDETRBase
    import supervision as sv
except ImportError:
    print("Warning: Missing rfdetr or supervision packages. Run 'pip install rfdetr supervision'.")

def get_args():
    parser = argparse.ArgumentParser(description="Phase 1: Feature Extraction (YOLOv26 + RF-DETR)")
    parser.add_argument("--sequence_dir", type=str, required=True,
                        help="Path to sequence directory containing 'img1' folder (e.g. SNMOT-197)")
    parser.add_argument("--yolo_weights", type=str, required=True,
                        help="Path to YOLO ball detection weights")
    parser.add_argument("--rfdetr_weights", type=str, 
                        default="models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth",
                        help="Path to RF-DETR weights for player detection")
    parser.add_argument("--output_json", type=str, default="secuencia_197_detections.json",
                        help="Path to output JSON file")
    parser.add_argument("--ball_conf", type=float, default=0.05,
                        help="Confidence threshold for dense ball detection")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda/cpu)")
    # Keep backward compat
    parser.add_argument("--rtdetr_weights", type=str, default=None, help="Deprecated, use --rfdetr_weights")
    return parser.parse_args()

def extract_features():
    args = get_args()

    # Determine which weights to use to allow seamless transitions
    rfdetr_path = args.rfdetr_weights
    if args.rtdetr_weights and not args.rfdetr_weights:
        rfdetr_path = args.rtdetr_weights

    print(f"Loading YOLO for ball detection from {args.yolo_weights}...")
    ball_model = YOLO(args.yolo_weights)
    
    print(f"Loading RF-DETR for player tracking from {rfdetr_path}...")
    player_model = RFDETRBase(pretrain_weights=rfdetr_path, resolution=448)
    tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=120)

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
        ball_results = ball_model.predict(img_path, conf=args.ball_conf, imgsz=1280, verbose=False, device=args.device)
        
        ball_candidates = []
        if len(ball_results) > 0 and len(ball_results[0].boxes) > 0:
            for box in ball_results[0].boxes:
                cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                score = float(box.conf[0].cpu().numpy())
                ball_candidates.append({
                    "x_center": cx,
                    "y_center": cy,
                    "w": w,
                    "h": h,
                    "score": score
                })
        
        # 2. Player Spatial Tracking (RF-DETR + ByteTrack)
        pil_img = Image.open(img_path).convert("RGB")
        players = []
        try:
            player_dets = player_model.predict(pil_img, threshold=0.15)
            if isinstance(player_dets, list) and len(player_dets) > 0:
                player_dets = player_dets[0]
            
            # Sólo rastrear si hay detecciones
            if hasattr(player_dets, 'xyxy') and len(player_dets.xyxy) > 0:
                # Limpiar datos extra (logits) que causan IndexError en ByteTrack
                clean_dets = sv.Detections(
                    xyxy=player_dets.xyxy,
                    confidence=player_dets.confidence,
                    class_id=player_dets.class_id
                )
                tracked_dets = tracker.update_with_detections(clean_dets)
                
                for j in range(len(tracked_dets.xyxy)):
                    xmin, ymin, xmax, ymax = tracked_dets.xyxy[j]
                    track_id = int(tracked_dets.tracker_id[j]) if tracked_dets.tracker_id is not None else -1
                    players.append({
                        "track_id": track_id,
                        "x_min": float(xmin),
                        "y_min": float(ymin),
                        "x_max": float(xmax),
                        "y_max": float(ymax),
                        "class_id": int(tracked_dets.class_id[j]) if tracked_dets.class_id is not None else 0
                    })
        except Exception as e:
            print(f"Tracking error on {frame_name}: {e}")
        
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
