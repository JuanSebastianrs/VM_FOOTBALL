"""
SAM2 Fine-Tuning on Pseudo-Masks (Vertex AI)

Trains SAM2 sam_mask_decoder + sam_prompt_encoder on football ball pseudo-masks.
Image encoder is frozen to fit on T4 (16GB VRAM).

Uses SAM2's ACTUAL internal API (forward_image, sam_prompt_encoder, sam_mask_decoder)
instead of the SAM1/HF convenience wrappers.

Input:
  gs://bucket/data_generation/sam2_training/SNMOT-XXX/
  ├── images/*.jpg
  ├── masks/*.png (binary, ball=255)
  └── prompts.json ({frame: {bbox, conf, ...}})

Output:
  gs://bucket/models/sam2_ball_finetuned/
  ├── best_model.pt
  ├── final_model.pt
  └── training_log.json
"""

import os
import cv2
import json
import time
import random
import argparse
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage


# ============================================================
# Dataset
# ============================================================
class BallSegDataset(Dataset):
    """Dataset for SAM2 fine-tuning. Each sample = (image, mask, box_prompt)."""

    def __init__(self, samples, img_size=1024):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load image
        image = cv2.imread(s["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Load mask
        mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        # Resize image to SAM2 input size
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # Resize mask to SAM2 low-res output (256x256)
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        # Scale bbox to img_size
        bbox = np.array(s["bbox"], dtype=np.float32)
        scale_x = float(self.img_size) / w
        scale_y = float(self.img_size) / h
        bbox_scaled = bbox.copy()
        bbox_scaled[0] *= scale_x
        bbox_scaled[2] *= scale_x
        bbox_scaled[1] *= scale_y
        bbox_scaled[3] *= scale_y
        bbox_tensor = torch.from_numpy(bbox_scaled).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "original_size": torch.tensor([h, w]),
        }


def load_all_samples(data_dir, sequences, split="train"):
    """Load all (image, mask, bbox) samples from downloaded sequences."""
    samples = []
    for seq_name in sequences:
        seq_dir = os.path.join(data_dir, seq_name)
        prompts_path = os.path.join(seq_dir, "prompts.json")

        if not os.path.exists(prompts_path):
            print(f"  [WARN] No prompts.json for {seq_name}, skipping")
            continue

        with open(prompts_path, 'r') as f:
            prompts = json.load(f)

        for frame_name, info in prompts.items():
            img_path = os.path.join(seq_dir, "images", frame_name)
            mask_name = frame_name.replace('.jpg', '.png')
            mask_path = os.path.join(seq_dir, "masks", mask_name)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "bbox": info["bbox"],
                    "sequence": seq_name
                })

    print(f"  [{split}] Loaded {len(samples)} samples from {len(sequences)} sequences")
    return samples


# ============================================================
# Loss Functions
# ============================================================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def combined_loss(pred_masks, gt_masks):
    return focal_loss(pred_masks, gt_masks) + dice_loss(pred_masks, gt_masks)


# ============================================================
# SAM2-Native Forward Pass
# ============================================================
def sam2_forward(model, image, bbox, device):
    """
    Run SAM2's native forward pass for training.
    
    This matches the internal API of SAM2Base exactly:
      1. forward_image() -> _prepare_backbone_features() for image encoding
      2. sam_prompt_encoder() with box-as-points format
      3. sam_mask_decoder() with high_res_features
    
    Args:
        model: SAM2Base model
        image: (1, 3, 1024, 1024) tensor
        bbox: (1, 4) tensor [x1, y1, x2, y2] in 1024x1024 coords
        device: torch device
    
    Returns:
        low_res_masks: (1, 1, 256, 256) logits
        iou_predictions: (1, 1) IoU scores
    """
    # 1. Image encoding (frozen, no_grad)
    with torch.no_grad():
        backbone_out = model.forward_image(image)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        
        # Add no_mem_embed if configured
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        
        # Build feature maps at different resolutions
        B = image.shape[0]
        # The backbone features are in (HW, B, C) format, need to reshape
        # SAM2 uses _bb_feat_sizes which are set during init
        # We compute them from the vision_feats shapes
        feat_sizes = []
        for feat in vision_feats[::-1]:
            # feat shape: (HW, B, C)
            hw = feat.shape[0]
            s = int(hw ** 0.5)
            feat_sizes.append((s, s))
        
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes)
        ][::-1]
        
        # image_embed = lowest res, high_res_feats = higher res
        image_embed = feats[-1]
        high_res_feats = feats[:-1]
    
    # 2. Prompt encoding (box -> points format)
    # SAM2 converts boxes to point format: reshape (B, 4) -> (B, 2, 2)
    # with labels [2, 3] (2=top-left corner, 3=bottom-right corner)
    box_coords = bbox.reshape(-1, 2, 2)  # (B, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
    box_labels = box_labels.repeat(bbox.shape[0], 1)  # (B, 2)
    
    concat_points = (box_coords, box_labels)
    
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
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


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    
    # Keep image encoder in eval mode (frozen)
    model.image_encoder.eval()

    total_loss = 0
    total_iou = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)

        optimizer.zero_grad()

        batch_loss = 0
        batch_iou = 0

        for i in range(images.shape[0]):
            img = images[i:i+1]
            box = bboxes[i:i+1]
            gt_mask = masks[i:i+1]
            
            low_res_masks, iou_pred = sam2_forward(model, img, box, device)

            loss = combined_loss(low_res_masks, gt_mask)
            batch_loss += loss

            with torch.no_grad():
                pred_binary = (torch.sigmoid(low_res_masks) > 0.5).float()
                intersection = (pred_binary * gt_mask).sum()
                union = pred_binary.sum() + gt_mask.sum() - intersection
                iou = (intersection / (union + 1e-8)).item()
                batch_iou += iou

        batch_loss = batch_loss / images.shape[0]
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        total_iou += batch_iou / images.shape[0]
        n_batches += 1

        if batch_idx % 20 == 0:
            avg_loss = total_loss / n_batches
            avg_iou = total_iou / n_batches
            print(f"  Epoch {epoch+1}/{total_epochs} | "
                  f"Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")

    return total_loss / n_batches, total_iou / n_batches


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)

        batch_iou = 0
        batch_loss = 0

        for i in range(images.shape[0]):
            img = images[i:i+1]
            box = bboxes[i:i+1]
            gt_mask = masks[i:i+1]
            
            low_res_masks, iou_pred = sam2_forward(model, img, box, device)

            loss = combined_loss(low_res_masks, gt_mask)
            batch_loss += loss.item()

            pred_binary = (torch.sigmoid(low_res_masks) > 0.5).float()
            intersection = (pred_binary * gt_mask).sum()
            union = pred_binary.sum() + gt_mask.sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
            batch_iou += iou

        total_loss += batch_loss / images.shape[0]
        total_iou += batch_iou / images.shape[0]
        n_batches += 1

    return total_loss / n_batches, total_iou / n_batches


# ============================================================
# GCS Helpers
# ============================================================
def download_training_data(bucket_name, prefix, local_dir):
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


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SAM2 Fine-Tuning")
    parser.add_argument("--gcs_bucket", type=str, required=True)
    parser.add_argument("--gcs_training_prefix", type=str,
                        default="data_generation/sam2_training")
    parser.add_argument("--gcs_output_prefix", type=str,
                        default="models/sam2_ball_finetuned")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="facebook/sam2.1-hiera-small")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print(" SAM2 FINE-TUNING: Ball Segmentation")
    print(f" Checkpoint: {args.sam2_checkpoint}")
    print(f" Epochs: {args.epochs}")
    print(f" Batch size: {args.batch_size}")
    print(f" LR: {args.lr}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # 1. Download training data
    print("\n[1/5] Downloading training data from GCS...")
    data_dir = "/tmp/sam2_training/data"
    download_training_data(args.gcs_bucket, args.gcs_training_prefix, data_dir)

    # 2. Discover sequences and split train/val
    print("\n[2/5] Preparing train/val split...")
    all_sequences = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("SNMOT")
    ])
    print(f"  Found {len(all_sequences)} sequences")

    random.seed(42)
    random.shuffle(all_sequences)
    n_val = max(1, int(len(all_sequences) * args.val_split))
    val_sequences = all_sequences[:n_val]
    train_sequences = all_sequences[n_val:]

    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Val: {len(val_sequences)} sequences")

    train_samples = load_all_samples(data_dir, train_sequences, "train")
    val_samples = load_all_samples(data_dir, val_sequences, "val")

    if not train_samples:
        print("[FATAL] No training samples found!")
        sys.exit(1)

    # 3. Load SAM2 model
    print("\n[3/5] Loading SAM2 model...")
    
    # Always use build_sam2 from the GitHub repo (most reliable)
    try:
        from sam2.build_sam import build_sam2
        import urllib.request
        
        ckpt_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        ckpt_path = "/tmp/sam2.1_hiera_small.pt"
        
        if not os.path.exists(ckpt_path):
            print(f"  Downloading checkpoint to {ckpt_path}...")
            urllib.request.urlretrieve(ckpt_url, ckpt_path)
            print(f"  Downloaded ({os.path.getsize(ckpt_path) / 1e6:.1f} MB)")
        
        # Config path relative to the cloned sam2 repo
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        model = build_sam2(model_cfg, ckpt_path)
        print("[OK] SAM2 loaded via build_sam2")
        
    except Exception as e:
        print(f"[FATAL] Cannot load SAM2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify model has the expected attributes
    print("  Verifying model attributes...")
    for attr in ['image_encoder', 'sam_prompt_encoder', 'sam_mask_decoder', 'forward_image']:
        has = hasattr(model, attr)
        print(f"    model.{attr}: {'OK' if has else 'MISSING!'}")
        if not has:
            print(f"[FATAL] Model missing required attribute: {attr}")
            sys.exit(1)
    
    model = model.to(device)

    # Freeze image encoder
    print("  Freezing image encoder...")
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    trainable_params = (
        list(model.sam_mask_decoder.parameters()) +
        list(model.sam_prompt_encoder.parameters())
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable params: {trainable_count/1e6:.1f}M ({trainable_count/total_params*100:.1f}%)")

    # 4. Create DataLoaders
    train_dataset = BallSegDataset(train_samples)
    val_dataset = BallSegDataset(val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # 5. Train
    print(f"\n[4/5] Training {args.epochs} epochs...")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    output_dir = "/tmp/sam2_training/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    best_val_iou = 0
    training_log = []

    for epoch in range(args.epochs):
        t_epoch = time.time()

        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs
        )

        val_loss, val_iou = validate(model, val_loader, device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t_epoch

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_iou": round(train_iou, 4),
            "val_loss": round(val_loss, 4),
            "val_iou": round(val_iou, 4),
            "lr": lr,
            "time_sec": round(epoch_time, 1)
        }
        training_log.append(log_entry)

        print(f"\n  Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} | "
              f"LR: {lr:.2e} | Time: {epoch_time:.0f}s")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_iou": val_iou,
                "train_iou": train_iou,
            }, best_path)
            print(f"  * New best model saved (IoU: {val_iou:.4f})")

            # Upload best to GCS immediately
            upload_blob(args.gcs_bucket, best_path,
                       f"{args.gcs_output_prefix}/best_model.pt")

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_iou": val_iou,
        "training_log": training_log,
    }, final_path)

    upload_blob(args.gcs_bucket, final_path,
               f"{args.gcs_output_prefix}/final_model.pt")

    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump({
            "config": {
                "checkpoint": args.sam2_checkpoint,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "val_split": args.val_split,
                "train_sequences": len(train_sequences),
                "val_sequences": len(val_sequences),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
            },
            "best_val_iou": round(best_val_iou, 4),
            "log": training_log
        }, f, indent=2)
    upload_blob(args.gcs_bucket, log_path,
               f"{args.gcs_output_prefix}/training_log.json")

    print(f"\n{'='*60}")
    print(f" TRAINING COMPLETE")
    print(f" Best Val IoU: {best_val_iou:.4f}")
    print(f" Model: gs://{args.gcs_bucket}/{args.gcs_output_prefix}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
