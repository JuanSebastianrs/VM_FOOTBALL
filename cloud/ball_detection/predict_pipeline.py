"""
End-to-End Tracking Pipeline: YOLO11 -> ByteTrack -> SAM2
Runs inference on a sequence of frames.
"""
import os
import sys
import glob
import cv2
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser(description="E2E Pipeline: YOLO -> ByteTrack -> SAM2")
    parser.add_argument("--sequence_dir", type=str, required=True,
                        help="Path to sequence directory containing 'img1' folder")
    parser.add_argument("--yolo_weights", type=str, required=True,
                        help="Path to YOLO11n-p2 best.pt weights")
    parser.add_argument("--sam2_weights", type=str, required=True,
                        help="Path to fine-tuned SAM2 best_model.pt")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output MP4 video")
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()


def sam2_forward_inference(model, image_tensor, bbox, device):
    """Run SAM2 forward pass using the native internal API."""
    with torch.no_grad():
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
            feat.permute(1, 2, 0).view(B, -1, *fs)
            for feat, fs in zip(vision_feats[::-1], feat_sizes)
        ][::-1]

        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        box_coords = bbox.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device).repeat(bbox.shape[0], 1)

        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=(box_coords, box_labels), boxes=None, masks=None,
        )

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
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ─── Load YOLO ──────────────────────────────────────────────
    print(f"\n[1/4] Loading YOLO model from {args.yolo_weights} ...")
    if not os.path.exists(args.yolo_weights):
        print(f"[FATAL] YOLO weights not found at {args.yolo_weights}")
        sys.exit(1)
    yolo_model = YOLO(args.yolo_weights)
    print("  [OK] YOLO loaded.")

    # ─── Load SAM2 ──────────────────────────────────────────────
    print(f"\n[2/4] Loading SAM2 model from {args.sam2_weights} ...")
    from sam2.build_sam import build_sam2
    import urllib.request
    
    BASE_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    BASE_CKPT_PATH = os.path.join(os.path.dirname(args.sam2_weights), "sam2.1_hiera_small.pt")
    if not os.path.exists(BASE_CKPT_PATH):
        print(f"  Downloading base checkpoint (~150MB)...")
        urllib.request.urlretrieve(BASE_CKPT_URL, BASE_CKPT_PATH)
        
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, BASE_CKPT_PATH)

    ft_state = torch.load(args.sam2_weights, map_location=device, weights_only=False)
    sam2_model.load_state_dict(ft_state["model_state_dict"])
    sam2_model = sam2_model.to(device)
    sam2_model.eval()
    print(f"  [OK] SAM2 loaded (epoch={ft_state.get('epoch', '?')}, val_iou={ft_state.get('val_iou', '?')}).")

    # ─── Load Images ─────────────────────────────────────────────
    img_dir = os.path.join(args.sequence_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        print(f"[FATAL] No images found in {img_dir}")
        sys.exit(1)
    print(f"\n[3/4] Found {len(images)} frames in {img_dir}")
    
    sample = cv2.imread(images[0])
    img_h, img_w = sample.shape[:2]
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output_path, fourcc, args.fps, (img_w, img_h))

    # ─── Inference ──────────────────────────────────────────────
    print(f"\n[4/4] Running E2E Inference...")
    mask_color = np.array([0, 255, 0], dtype=np.uint8)
    mask_alpha = 0.45
    
    frames_processed = 0
    frames_with_ball = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        
        # 1. Run YOLO + ByteTrack
        # Ultralytics track() directly integrates ByteTrack to assign consistent IDs
        # We explicitly filter for class 5 if the model has multiple classes
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[5])
        
        ball_bbox = None
        ball_id = None
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # Find the ball detection (if multiple, take highest confidence or first)
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == 5:
                    ball_bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    ball_id = int(box.id[0].item()) if box.id is not None else -1
                    break
        
        # 2. Run SAM2 Mask generation
        if ball_bbox is not None:
            frames_with_ball += 1
            
            # Prepare image & bbox for SAM2
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1024, 1024))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            scale_x, scale_y = 1024.0 / img_w, 1024.0 / img_h
            bbox_scaled = [ball_bbox[0]*scale_x, ball_bbox[1]*scale_y, ball_bbox[2]*scale_x, ball_bbox[3]*scale_y]
            bbox_tensor = torch.tensor([bbox_scaled], dtype=torch.float32, device=device)

            low_res_masks, iou_pred = sam2_forward_inference(sam2_model, img_tensor, bbox_tensor, device)

            # Upscale and apply mask
            mask_256 = torch.sigmoid(low_res_masks)
            mask_full = F.interpolate(mask_256, size=(img_h, img_w), mode='bilinear', align_corners=False)
            mask_binary = (mask_full.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            overlay = frame.copy()
            overlay[mask_binary == 1] = (
                frame[mask_binary == 1] * (1 - mask_alpha) + mask_color * mask_alpha
            ).astype(np.uint8)

            # Draw bbox and text
            x1, y1, x2, y2 = [int(v) for v in ball_bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            iou_val = iou_pred[0, 0].item()
            txt = f"ID:{ball_id} | IoU:{iou_val:.2f}" if ball_id != -1 else f"IoU:{iou_val:.2f}"
            cv2.putText(overlay, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(overlay, "YOLO11->ByteTrack->SAM2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out_video.write(overlay)
        else:
            # No ball detected
            cv2.putText(frame, "No ball detected by YOLO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out_video.write(frame)
            
        frames_processed += 1
        if frames_processed % 50 == 0:
            print(f"  Processed {frames_processed}/{len(images)} frames ({frames_with_ball} with ball)")

    out_video.release()
    print(f"\n{'='*60}")
    print(f" DONE!")
    print(f" Video saved to: {args.output_path}")
    print(f" Frames with ball: {frames_with_ball}/{len(images)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
