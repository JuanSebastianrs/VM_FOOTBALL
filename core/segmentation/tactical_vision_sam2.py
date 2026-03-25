import os
import cv2
import glob
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(description="Phase 6: SAM 2 Segmentation and Rendering")
    parser.add_argument("--sequence_dir", type=str, required=True, help="Path to sequence directory 'img1'")
    parser.add_argument("--detections_json", type=str, default="secuencia_197_detections.json")
    parser.add_argument("--trajectory_json", type=str, default="secuencia_197_trajectory.json")
    parser.add_argument("--sam2_weights", type=str, required=True, help="Path to SAM2 weights")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output videos")
    parser.add_argument("--output_masks_dir", type=str, default="", help="Optional directory to save SAM2 binary masks")
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()

def sam2_box_inference(model, image_tensor, box_coords, device):
    """Run SAM2 forward pass using box prompts."""
    with torch.no_grad():
        backbone_out = model.forward_image(image_tensor)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

        B = image_tensor.shape[0]
        feat_sizes = [(int(hw ** 0.5), int(hw ** 0.5)) for hw in [f.shape[0] for f in vision_feats[::-1]]]
        feats = [f.permute(1, 2, 0).view(B, -1, *fs) for f, fs in zip(vision_feats[::-1], feat_sizes)][::-1]

        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        # bbox shape required: [B, N, 4]
        # Our box_coords is [x1, y1, x2, y2]
        box_tensor = torch.tensor(box_coords, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, 4]

        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=None, boxes=box_tensor, masks=None,
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

def render_videos():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup SAM 2
    print(f"Loading SAM2 model from {args.sam2_weights} ...")
    from sam2.build_sam import build_sam2
    import urllib.request
    
    BASE_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    BASE_CKPT_PATH = "sam2.1_hiera_small.pt"
    if not os.path.exists(BASE_CKPT_PATH):
        print(f"Downloading base checkpoint...")
        urllib.request.urlretrieve(BASE_CKPT_URL, BASE_CKPT_PATH)
        
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, BASE_CKPT_PATH)

    ft_state = torch.load(args.sam2_weights, map_location=device, weights_only=False)
    if "model_state_dict" in ft_state:
        sam2_model.load_state_dict(ft_state["model_state_dict"])
    else:
        sam2_model.load_state_dict(ft_state)
        
    sam2_model = sam2_model.to(device)
    sam2_model.eval()

    # 2. Load JSON data
    with open(args.detections_json, 'r') as f:
        detections = json.load(f)
    
    with open(args.trajectory_json, 'r') as f:
        trajectory = json.load(f)
        
    # Map by frame ID
    det_map = {d["frame_id"]: d for d in detections}
    traj_map = {d["frame_id"]: d for d in trajectory}

    img_dir = os.path.join(args.sequence_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if not images:
        print(f"Error: No images found in {img_dir}")
        return

    sample = cv2.imread(images[0])
    img_h, img_w = sample.shape[:2]
    
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, f"{os.path.basename(args.sequence_dir)}_yolo_raw.mp4")
    refined_path = os.path.join(args.output_dir, f"{os.path.basename(args.sequence_dir)}_tacticalvision_refined.mp4")
    
    if args.output_masks_dir:
        os.makedirs(args.output_masks_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_raw = cv2.VideoWriter(raw_path, fourcc, args.fps, (img_w, img_h))
    out_refined = cv2.VideoWriter(refined_path, fourcc, args.fps, (img_w, img_h))

    print(f"Starting Video Renderization...")
    mask_color = np.array([0, 255, 0], dtype=np.uint8)
    mask_alpha = 0.5
    
    for idx, img_path in enumerate(images):
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
        frame = cv2.imread(img_path)
        
        # --- RAW VIDEO RENDER ---
        raw_frame = frame.copy()
        if frame_id in det_map:
            # Draw players for RAW video as well
            for p in det_map[frame_id].get("players", []):
                px1, py1, px2, py2 = int(p["x_min"]), int(p["y_min"]), int(p["x_max"]), int(p["y_max"])
                tid = p["track_id"]
                cv2.rectangle(raw_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(raw_frame, f"P:{tid}", (px1, max(0, py1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            # Draw balls
            for ball in det_map[frame_id].get("ball_candidates", []):
                x_c, y_c, w, h = ball["x_center"], ball["y_center"], ball["w"], ball["h"]
                sc = ball["score"]
                x1, y1 = int(x_c - w/2), int(y_c - h/2)
                x2, y2 = int(x_c + w/2), int(y_c + h/2)
                cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(raw_frame, f"B:{sc:.2f}", (x1, max(0, y1-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(raw_frame, "RAW Detections (YOLOv26 + RT-DETR)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out_raw.write(raw_frame)
        
        # --- REFINED VIDEO RENDER (SAM 2) ---
        refined_frame = frame.copy()
        if frame_id in det_map:
            # Draw players for Refined video
            for p in det_map[frame_id].get("players", []):
                px1, py1, px2, py2 = int(p["x_min"]), int(p["y_min"]), int(p["x_max"]), int(p["y_max"])
                tid = p["track_id"]
                cv2.rectangle(refined_frame, (px1, py1), (px2, py2), (255, 128, 0), 2)
                cv2.putText(refined_frame, f"P:{tid}", (px1, max(0, py1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)

        if frame_id in traj_map:
            traj_node = traj_map[frame_id]
            x, y, w, h = traj_node["x"], traj_node["y"], traj_node["w"], traj_node["h"]
            
            if x != -1 and y != -1 and w > 0 and h > 0: # Valid position and size
                # Prepare bounding box for SAM2
                bx1, by1 = x - w/2, y - h/2
                bx2, by2 = x + w/2, y + h/2
                
                # Draw the bounding box of the optimized ball
                cv2.rectangle(refined_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 255), 2)
                
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (1024, 1024))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                scale_x, scale_y = 1024.0 / img_w, 1024.0 / img_h
                box_scaled = [bx1 * scale_x, by1 * scale_y, bx2 * scale_x, by2 * scale_y]
                
                low_res_masks, iou_pred = sam2_box_inference(sam2_model, img_tensor, box_scaled, device)

                mask_256 = torch.sigmoid(low_res_masks)
                mask_full = F.interpolate(mask_256, size=(img_h, img_w), mode='bilinear', align_corners=False)
                mask_binary = (mask_full.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                if args.output_masks_dir:
                    mask_filename = os.path.basename(img_path).replace(".jpg", ".png")
                    mask_save_path = os.path.join(args.output_masks_dir, mask_filename)
                    cv2.imwrite(mask_save_path, mask_binary * 255)

                refined_frame[mask_binary == 1] = (
                    refined_frame[mask_binary == 1] * (1 - mask_alpha) + mask_color * mask_alpha
                ).astype(np.uint8)
                
                status_txt = "TacticalVision: Tracked" if not traj_node['is_dummy'] else f"TacticalVision: Occluded (Player {traj_node['attached_player_id']})"
                color = (0, 255, 0) if not traj_node['is_dummy'] else (0, 165, 255)
            else:
                status_txt = "TacticalVision: Lost"
                color = (0, 0, 255)
            
            cv2.putText(refined_frame, status_txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(refined_frame, "TacticalVision: No Data", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            
        cv2.putText(refined_frame, "TacticalVison Refined (RT-DETR + Optimized Ball + SAM2)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_refined.write(refined_frame)
        
        if idx % 50 == 0:
            print(f"Rendered frame {idx}/{len(images)}...")

    out_raw.release()
    out_refined.release()
    print(f"Done! Saved {raw_path} and {refined_path}")

if __name__ == "__main__":
    render_videos()
