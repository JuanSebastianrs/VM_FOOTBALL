"""
Local SAM2 Inference on test_seq_116 Test Sequence

Loads the fine-tuned SAM2 model and runs inference on test_seq_116,
producing a video with mask overlays. Runs locally on 2080 Ti.
"""
import os, sys, glob, cv2, numpy as np, torch, torch.nn.functional as F, urllib.request

# ── Paths ────────────────────────────────────────────────────
SEQ_DIR   = r"D:\sebastian\Tesis\VM_FOOTBALL\datasets\test_seq_116"
IMG_DIR   = os.path.join(SEQ_DIR, "img1")
LABELS_DIR = os.path.join(SEQ_DIR, "labels")
CKPT_PATH = r"D:\sebastian\Tesis\VM_FOOTBALL\training\segmentation\results\checkpoints\best_model.pt"
OUTPUT_DIR = r"D:\sebastian\Tesis\VM_FOOTBALL\cloud\ball_detection\results\inference_video"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "test_seq_116_sam2_finetuned.mp4")

BASE_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
BASE_CKPT_PATH = os.path.join(os.path.dirname(CKPT_PATH), "sam2.1_hiera_small.pt")
FPS = 25
BALL_CLASS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_yolo_ball_detections(labels_dir, img_w, img_h):
    """Load ball detections (class 5) from YOLO label files."""
    detections = {}
    for label_file in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        frame_num = int(os.path.basename(label_file).replace('.txt', ''))
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                if cls == BALL_CLASS:
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (cx - w / 2) * img_w
                    y1 = (cy - h / 2) * img_h
                    x2 = (cx + w / 2) * img_w
                    y2 = (cy + h / 2) * img_h
                    detections[frame_num] = [x1, y1, x2, y2]
                    break
    return detections


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load images
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"\n[1/4] Found {len(images)} frames in {IMG_DIR}")
    sample = cv2.imread(images[0])
    img_h, img_w = sample.shape[:2]
    print(f"  Resolution: {img_w}x{img_h}")

    # 2. Load ball detections
    print(f"\n[2/4] Loading ball detections from {LABELS_DIR}")
    detections = load_yolo_ball_detections(LABELS_DIR, img_w, img_h)
    print(f"  Found {len(detections)} frames with ball (class {BALL_CLASS})")

    # 3. Load SAM2 model
    print(f"\n[3/4] Loading SAM2 model...")
    from sam2.build_sam import build_sam2

    if not os.path.exists(BASE_CKPT_PATH):
        print(f"  Downloading base checkpoint (~150MB)...")
        urllib.request.urlretrieve(BASE_CKPT_URL, BASE_CKPT_PATH)
        print(f"  Saved to {BASE_CKPT_PATH}")

    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    model = build_sam2(model_cfg, BASE_CKPT_PATH)

    # Load fine-tuned weights
    print(f"  Loading fine-tuned weights from {CKPT_PATH}")
    ft_state = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ft_state["model_state_dict"])
    epoch = ft_state.get("epoch", "?")
    val_iou = ft_state.get("val_iou", "?")
    print(f"  Loaded: epoch={epoch}, val_iou={val_iou}")

    model = model.to(device)
    model.eval()
    print(f"  Model on {device}, ready!")

    # 4. Run inference
    print(f"\n[4/4] Running inference...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (img_w, img_h))

    mask_color = np.array([0, 255, 0], dtype=np.uint8)
    mask_alpha = 0.45
    frames_with_mask = 0
    total_iou_pred = 0.0

    for idx, img_path in enumerate(images):
        frame_num = idx + 1
        frame = cv2.imread(img_path)

        if frame_num in detections:
            bbox = detections[frame_num]

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (1024, 1024))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            scale_x, scale_y = 1024.0 / img_w, 1024.0 / img_h
            bbox_scaled = [bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y]
            bbox_tensor = torch.tensor([bbox_scaled], dtype=torch.float32, device=device)

            low_res_masks, iou_pred = sam2_forward_inference(model, img_tensor, bbox_tensor, device)

            mask_256 = torch.sigmoid(low_res_masks)
            mask_full = F.interpolate(mask_256, size=(img_h, img_w), mode='bilinear', align_corners=False)
            mask_binary = (mask_full.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            # Overlay
            overlay = frame.copy()
            overlay[mask_binary == 1] = (
                frame[mask_binary == 1] * (1 - mask_alpha) +
                mask_color * mask_alpha
            ).astype(np.uint8)

            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

            iou_val = iou_pred[0, 0].item()
            total_iou_pred += iou_val
            cv2.putText(overlay, f"IoU: {iou_val:.3f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(overlay, "SAM2 Fine-tuned (Ball)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out_video.write(overlay)
            frames_with_mask += 1
        else:
            cv2.putText(frame, "No ball detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out_video.write(frame)

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(images)} frames ({frames_with_mask} with mask)")

    out_video.release()

    avg_iou = total_iou_pred / max(frames_with_mask, 1)
    print(f"\n{'='*60}")
    print(f" DONE!")
    print(f" Video: {OUTPUT_VIDEO}")
    print(f" Frames with mask: {frames_with_mask}/{len(images)}")
    print(f" Average predicted IoU: {avg_iou:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
