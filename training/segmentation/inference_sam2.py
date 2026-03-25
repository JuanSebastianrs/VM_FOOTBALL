"""
SAM2 Inference on Test Sequence (Vertex AI)

Loads the fine-tuned SAM2 model and runs inference on a test sequence,
producing a video with mask overlays.

Usage (via Vertex AI):
  python inference_sam2.py --gcs_bucket vm-football-data \
    --sequence SNMOT-143 --checkpoint_path models/sam2_ball_finetuned/best_model.pt
"""

import os
import cv2
import json
import argparse
import numpy as np
import sys
import glob

import torch
import torch.nn.functional as F
from google.cloud import storage


def download_from_gcs(bucket_name, prefix, local_dir):
    """Download files from GCS."""
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    count = 0
    for b in blobs:
        if b.name.endswith('/'):
            continue
        rel = b.name[len(prefix):].lstrip('/')
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        b.download_to_filename(dest)
        count += 1
    print(f"[GCS] Downloaded {count} files from gs://{bucket_name}/{prefix}")
    return count


def upload_blob(bucket_name, local_path, gcs_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"[GCS] Uploaded -> gs://{bucket_name}/{gcs_path}")


def load_yolo_ball_detections(labels_dir, img_w, img_h):
    """Load ball detections (class 5) from YOLO label files.
    Returns dict: frame_num -> [x1, y1, x2, y2] in pixel coords.
    """
    detections = {}
    for label_file in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        frame_num = int(os.path.basename(label_file).replace('.txt', ''))
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                if cls == 5:  # ball class
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Convert normalized YOLO to pixel coords [x1, y1, x2, y2]
                    x1 = (cx - w/2) * img_w
                    y1 = (cy - h/2) * img_h
                    x2 = (cx + w/2) * img_w
                    y2 = (cy + h/2) * img_h
                    detections[frame_num] = [x1, y1, x2, y2]
                    break  # Take first ball detection per frame
    return detections


def load_gt_ball_detections(gt_path, img_w, img_h):
    """Fallback: Load ball detections from gt.txt (MOT format).
    gt.txt columns: frame, id, x, y, w, h, conf, ...
    We look for the ball track (typically smallest bbox).
    Returns dict: frame_num -> [x1, y1, x2, y2]
    """
    detections = {}
    # Read all detections
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            # Ball is usually the smallest object - check if area < threshold
            area = w * h
            if area < 1000:  # Ball is tiny
                detections[frame] = [x, y, x + w, y + h]
    return detections


def sam2_forward_inference(model, image_tensor, bbox, device):
    """
    Run SAM2 forward pass for inference (same as training but no_grad).
    """
    with torch.no_grad():
        # 1. Image encoding
        backbone_out = model.forward_image(image_tensor)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        
        B = image_tensor.shape[0]
        feat_sizes = []
        for feat in vision_feats[::-1]:
            hw = feat.shape[0]
            s = int(hw ** 0.5)
            feat_sizes.append((s, s))
        
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes)
        ][::-1]
        
        image_embed = feats[-1]
        high_res_feats = feats[:-1]
        
        # 2. Prompt encoding (box -> points format)
        box_coords = bbox.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
        box_labels = box_labels.repeat(bbox.shape[0], 1)
        concat_points = (box_coords, box_labels)
        
        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=concat_points, boxes=None, masks=None,
        )
        
        # 3. Mask decoding
        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        
    return low_res_masks, iou_predictions


def main():
    parser = argparse.ArgumentParser(description="SAM2 Inference on Test Sequence")
    parser.add_argument("--gcs_bucket", type=str, default="vm-football-data")
    parser.add_argument("--sequence", type=str, required=True,
                        help="Test sequence name (e.g., SNMOT-143)")
    parser.add_argument("--gcs_test_prefix", type=str,
                        default="test_sequences")
    parser.add_argument("--checkpoint_gcs_path", type=str,
                        default="models/sam2_ball_finetuned/best_model.pt")
    parser.add_argument("--output_gcs_prefix", type=str,
                        default="models/sam2_ball_finetuned/inference_videos")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Download test sequence
    print(f"\n[1/4] Downloading test sequence {args.sequence}...")
    seq_dir = f"/tmp/test_data/{args.sequence}"
    download_from_gcs(args.gcs_bucket, 
                      f"{args.gcs_test_prefix}/{args.sequence}/",
                      seq_dir)
    
    # Find images
    img_dir = os.path.join(seq_dir, "img1")
    labels_dir = os.path.join(seq_dir, "labels")
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        print("[FATAL] No images found!")
        sys.exit(1)
    
    # Get image dimensions
    sample_img = cv2.imread(images[0])
    img_h, img_w = sample_img.shape[:2]
    print(f"  Found {len(images)} frames at {img_w}x{img_h}")
    
    # 2. Load ball detections
    print("\n[2/4] Loading ball detections...")
    if os.path.exists(labels_dir):
        detections = load_yolo_ball_detections(labels_dir, img_w, img_h)
        print(f"  Loaded {len(detections)} ball detections from YOLO labels")
    elif os.path.exists(gt_path):
        detections = load_gt_ball_detections(gt_path, img_w, img_h)
        print(f"  Loaded {len(detections)} ball detections from gt.txt")
    else:
        print("[FATAL] No detection source found!")
        sys.exit(1)
    
    # 3. Load SAM2 model with fine-tuned weights
    print("\n[3/4] Loading SAM2 model...")
    from sam2.build_sam import build_sam2
    import urllib.request
    
    base_ckpt_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    base_ckpt_path = "/tmp/sam2.1_hiera_small.pt"
    
    if not os.path.exists(base_ckpt_path):
        print(f"  Downloading base checkpoint...")
        urllib.request.urlretrieve(base_ckpt_url, base_ckpt_path)
    
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    model = build_sam2(model_cfg, base_ckpt_path)
    
    # Load fine-tuned weights
    ft_ckpt_path = "/tmp/best_model.pt"
    if not os.path.exists(ft_ckpt_path):
        print(f"  Downloading fine-tuned checkpoint from GCS...")
        download_from_gcs(args.gcs_bucket, args.checkpoint_gcs_path, "/tmp/")
        # The file might be nested, let's find it
        import shutil
        candidate = f"/tmp/{os.path.basename(args.checkpoint_gcs_path)}"
        if os.path.exists(candidate):
            ft_ckpt_path = candidate
    
    ft_state = torch.load(ft_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ft_state["model_state_dict"])
    print(f"  Fine-tuned weights loaded (trained epoch: {ft_state.get('epoch', '?')}, "
          f"val IoU: {ft_state.get('val_iou', '?')})")
    
    model = model.to(device)
    model.eval()
    
    # 4. Run inference and create video
    print(f"\n[4/4] Running inference on {len(images)} frames...")
    output_dir = "/tmp/inference_output"
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, f"{args.sequence}_sam2_finetuned.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, args.fps, (img_w, img_h))
    
    # Mask color (semi-transparent green)
    mask_color = np.array([0, 255, 0], dtype=np.uint8)
    mask_alpha = 0.4
    
    frames_with_mask = 0
    frames_without_detection = 0
    total_iou_pred = 0
    
    for idx, img_path in enumerate(images):
        frame_num = idx + 1
        frame = cv2.imread(img_path)
        
        if frame_num in detections:
            bbox = detections[frame_num]
            
            # Prepare image for SAM2 (resize to 1024x1024)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1024, 1024))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Scale bbox to 1024x1024
            scale_x = 1024.0 / img_w
            scale_y = 1024.0 / img_h
            bbox_scaled = [
                bbox[0] * scale_x, bbox[1] * scale_y,
                bbox[2] * scale_x, bbox[3] * scale_y
            ]
            bbox_tensor = torch.tensor([bbox_scaled], dtype=torch.float32, device=device)
            
            # SAM2 inference
            low_res_masks, iou_pred = sam2_forward_inference(model, img_tensor, bbox_tensor, device)
            
            # Upscale mask to original resolution
            mask_256 = torch.sigmoid(low_res_masks)  # (1, 1, 256, 256)
            mask_full = F.interpolate(mask_256, size=(img_h, img_w), mode='bilinear', align_corners=False)
            mask_binary = (mask_full.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
            # Overlay mask on frame
            overlay = frame.copy()
            overlay[mask_binary == 1] = (
                frame[mask_binary == 1] * (1 - mask_alpha) + 
                mask_color * mask_alpha
            ).astype(np.uint8)
            
            # Draw bbox
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Add IoU prediction text
            iou_val = iou_pred[0, 0].item()
            total_iou_pred += iou_val
            cv2.putText(overlay, f"IoU: {iou_val:.3f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add "SAM2 Fine-tuned" label
            cv2.putText(overlay, "SAM2 Fine-tuned", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out_video.write(overlay)
            frames_with_mask += 1
        else:
            # No ball detection for this frame
            cv2.putText(frame, "No ball detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out_video.write(frame)
            frames_without_detection += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(images)} frames "
                  f"({frames_with_mask} with mask)")
    
    out_video.release()
    
    avg_iou = total_iou_pred / max(frames_with_mask, 1)
    print(f"\n  Video saved: {video_path}")
    print(f"  Frames with mask: {frames_with_mask}")
    print(f"  Frames without detection: {frames_without_detection}")
    print(f"  Average predicted IoU: {avg_iou:.4f}")
    
    # Upload video to GCS
    gcs_video_path = f"{args.output_gcs_prefix}/{args.sequence}_sam2_finetuned.mp4"
    upload_blob(args.gcs_bucket, video_path, gcs_video_path)
    
    print(f"\n{'='*60}")
    print(f" INFERENCE COMPLETE")
    print(f" Video: gs://{args.gcs_bucket}/{gcs_video_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
