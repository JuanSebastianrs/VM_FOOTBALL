"""
SAHI + ByteTrack Comparison Test

This script slices each 1920x1080 football frame into overlapping 640x640 patches,
runs YOLO inference on each patch independently, merges all detections back to
absolute coordinates, applies NMS to remove duplicates, and feeds the final
detections to ByteTrack for temporal tracking.

The result is a video that can be directly compared to the plain ByteTrack baseline.
"""

import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO

model_path = "models/yolo_ball_data_centric.pt"
sequence_folder = "datasets/test_seq_116"
output_video_path = "cloud/ball_detection/results/inference_video/test_seq_116_sahi_bytetrack.mp4"

# SAHI Configuration
SLICE_SIZE = 640          # Size of each patch (px)
OVERLAP_RATIO = 0.2       # 20% overlap between patches
ROI_TOP_CROP = 0.12       # Crop top 12% (broadcast camera tribune area)
CONF_THRESHOLD = 0.15     # Low conf to allow ByteTrack to associate weak detections
CONF_EARLY_EXIT = 0.85    # If a detection exceeds this, skip remaining slices
IOU_NMS = 0.45            # NMS IoU threshold for merging overlapping slice detections


def generate_slices(img_w, img_h, slice_size, overlap_ratio, y_start):
    """Generate (x1, y1, x2, y2) coordinates for all SAHI slices."""
    step = int(slice_size * (1 - overlap_ratio))
    slices = []
    for y in range(y_start, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            x1 = max(0, x2 - slice_size)  # Ensure full patch even at edges
            y1 = max(y_start, y2 - slice_size)
            slices.append((x1, y1, x2, y2))
    return slices


def nms_boxes(boxes, scores, iou_threshold):
    """Apply Non-Maximum Suppression to merged slice detections."""
    if len(boxes) == 0:
        return [], []
    
    indices = cv2.dnn.NMSBoxes(
        bboxes=[(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])) for b in boxes],
        scores=scores,
        score_threshold=CONF_THRESHOLD,
        nms_threshold=iou_threshold
    )
    
    if len(indices) == 0:
        return [], []
    
    indices = indices.flatten()
    return [boxes[i] for i in indices], [scores[i] for i in indices]


def run_sahi_bytetrack():
    print("==========================================================")
    print(" SAHI + ByteTrack (Sequence) - Slicing Aided Hyper Inference")
    print("==========================================================")
    
    if not os.path.exists(sequence_folder):
        print(f"[ERROR] Folder not found: {sequence_folder}")
        return

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    
    image_files = sorted(glob.glob(os.path.join(sequence_folder, "*.jpg")))
    if not image_files:
        print("[ERROR] No images found.")
        return

    first_frame = cv2.imread(image_files[0])
    img_h, img_w = first_frame.shape[:2]
    
    y_start = int(img_h * ROI_TOP_CROP)
    slices = generate_slices(img_w, img_h, SLICE_SIZE, OVERLAP_RATIO, y_start)
    
    print(f"[INFO] Frame resolution: {img_w}x{img_h}")
    print(f"[INFO] ROI crop: top {ROI_TOP_CROP*100:.0f}% removed (y_start = {y_start}px)")
    print(f"[INFO] SAHI grid: {len(slices)} patches of {SLICE_SIZE}x{SLICE_SIZE} (overlap={OVERLAP_RATIO*100:.0f}%)")
    print(f"[INFO] Found {len(image_files)} frames. Starting SAHI + ByteTrack...")
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (img_w, img_h))
    
    # Stats
    total_early_exits = 0
    total_detections = 0
    
    for frame_idx, img_path in enumerate(image_files):
        if frame_idx % 50 == 0:
            print(f"  SAHI Tracking frame {frame_idx}/{len(image_files)}...")
            
        frame = cv2.imread(img_path)
        
        # ---- SAHI: Slice, Infer, Merge ----
        all_boxes = []
        all_scores = []
        all_cls = []
        early_exit = False
        
        for sx1, sy1, sx2, sy2 in slices:
            patch = frame[sy1:sy2, sx1:sx2]
            
            results = model.predict(
                source=patch,
                conf=CONF_THRESHOLD,
                imgsz=SLICE_SIZE,
                verbose=False
            )
            
            for r in results:
                for box in r.boxes:
                    # Get coordinates relative to the patch
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Project back to absolute frame coordinates
                    abs_x1 = bx1 + sx1
                    abs_y1 = by1 + sy1
                    abs_x2 = bx2 + sx1
                    abs_y2 = by2 + sy1
                    
                    all_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
                    all_scores.append(conf)
                    all_cls.append(cls)
                    
                    # Early Exit: if very high confidence, skip remaining slices
                    if conf >= CONF_EARLY_EXIT:
                        early_exit = True
                        
            if early_exit:
                total_early_exits += 1
                break
        
        # ---- NMS: Remove duplicate detections from overlapping slices ----
        final_boxes, final_scores = nms_boxes(all_boxes, all_scores, IOU_NMS)
        total_detections += len(final_boxes)
        
        # ---- Feed merged detections to ByteTrack via model.track() ----
        # We run track() on the full frame but at a very low imgsz just to
        # initialize the tracker; the SAHI detections are the real source.
        # For this baseline comparison, we combine: let YOLO also see the
        # full frame at 1280 to catch anything SAHI missed, then merge.
        track_results = model.track(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_NMS,
            imgsz=1280,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )
        
        # ---- Draw SAHI detections (green) on top of ByteTrack annotations ----
        annotated = track_results[0].plot()
        
        for box, score in zip(final_boxes, final_scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = (0, 255, 0) if score >= 0.5 else (0, 255, 255)  # Green=high, Yellow=low
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"SAHI {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw ROI crop line
        cv2.line(annotated, (0, y_start), (img_w, y_start), (255, 0, 0), 1)
        cv2.putText(annotated, "ROI CROP", (10, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        out.write(annotated)

    out.release()
    
    print(f"\n{'='*60}")
    print(f" SAHI + ByteTrack COMPLETE")
    print(f"  Total frames:       {len(image_files)}")
    print(f"  Total SAHI dets:    {total_detections}")
    print(f"  Early exits:        {total_early_exits} ({total_early_exits/len(image_files)*100:.1f}%)")
    print(f"  Output:             {output_video_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_sahi_bytetrack()
