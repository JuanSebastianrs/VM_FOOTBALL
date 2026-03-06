"""
Pseudo-Mask Visualization: Creates videos overlaying SAM2 masks on original frames.

For each sequence, generates a video showing:
  - Original frame
  - SAM2 mask overlaid in semi-transparent green
  - Bounding box from ByteTrack prompt
  - Confidence + IoU score annotations

Usage:
  python visualize_masks.py --data_dir inferences_video/sam2_masks --output_dir inferences_video
"""

import os
import cv2
import json
import argparse
import numpy as np


def create_mask_overlay_video(seq_dir, output_path, fps=25):
    """Create a video showing mask overlays on original frames."""
    
    images_dir = os.path.join(seq_dir, "images")
    masks_dir = os.path.join(seq_dir, "masks")
    prompts_path = os.path.join(seq_dir, "prompts.json")
    seq_name = os.path.basename(seq_dir)
    
    if not os.path.exists(prompts_path):
        print(f"  [SKIP] {seq_name}: no prompts.json")
        return
    
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    # Get all image files sorted
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    if not all_images:
        print(f"  [SKIP] {seq_name}: no images found")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(images_dir, all_images[0]))
    h, w = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    mask_count = 0
    no_mask_count = 0
    
    for frame_name in all_images:
        frame = cv2.imread(os.path.join(images_dir, frame_name))
        
        mask_name = frame_name.replace('.jpg', '.png')
        mask_path = os.path.join(masks_dir, mask_name)
        
        if frame_name in prompts and os.path.exists(mask_path):
            # Load mask and create overlay
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                mask_bool = mask > 127
                
                # Green overlay for mask
                overlay = frame.copy()
                overlay[mask_bool] = [0, 255, 0]  # Green
                frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                
                # Find mask contour for clean outline
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                
                # Draw bounding box
                info = prompts[frame_name]
                bbox = info["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                
                # Annotation: confidence + IoU
                conf = info.get("conf", 0)
                iou = info.get("sam2_iou_score", 0)
                area = info.get("mask_area", 0)
                label = f"conf:{conf:.2f} IoU:{iou:.2f} px:{area}"
                
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                mask_count += 1
            else:
                # Mask exists but wrong shape
                cv2.putText(frame, "MASK SIZE MISMATCH", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                no_mask_count += 1
        else:
            # No mask for this frame
            cv2.putText(frame, "NO DETECTION", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            no_mask_count += 1
        
        # Sequence label
        cv2.putText(frame, f"{seq_name} | {frame_name}", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"  [OK] {seq_name}: {mask_count} masked, {no_mask_count} no-mask → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM2 pseudo-masks")
    parser.add_argument("--data_dir", type=str, default="inferences_video/sam2_masks",
                        help="Directory containing downloaded sequence folders")
    parser.add_argument("--output_dir", type=str, default="inferences_video",
                        help="Where to save output videos")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all sequence directories
    sequences = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("SNMOT")
    ])
    
    if not sequences:
        print("[ERROR] No SNMOT-* sequence folders found!")
        return
    
    print(f"Found {len(sequences)} sequences to visualize:")
    for seq in sequences:
        print(f"  • {seq}")
    print()
    
    for seq_name in sequences:
        seq_dir = os.path.join(args.data_dir, seq_name)
        output_path = os.path.join(args.output_dir, f"sam2_masks_{seq_name}.mp4")
        
        print(f"Processing {seq_name}...")
        create_mask_overlay_video(seq_dir, output_path, fps=args.fps)
    
    print(f"\nDone! Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
