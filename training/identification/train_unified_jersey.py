"""
Unified training script for jersey number recognition.
Supports multi-source training:
  - "soccernet": SoccerNet Jersey 2023 official dataset
  - "tracking": Tracking-derived dataset (jersey_tracking_v1)
  - "mixed": Both datasets concatenated

Usage (Phase A - Pretrain on SoccerNet):
    python training/identification/train_unified_jersey.py \
        --dataset_mode soccernet \
        --soccernet_dir datasets/soccernet/jersey-2023 \
        --output_dir runs/jersey_unified_soccernet \
        --epochs 30 --batch_size 8 --lr 1e-4

Usage (Phase B - Fine-tune on Tracking):
    python training/identification/train_unified_jersey.py \
        --dataset_mode tracking \
        --tracking_dir datasets/jersey_tracking_v1 \
        --output_dir runs/jersey_unified_finetune \
        --pretrained runs/jersey_unified_soccernet/best.pt \
        --epochs 15 --batch_size 8 --lr 3e-5

Usage (Phase Mixed - Both from scratch):
    python training/identification/train_unified_jersey.py \
        --dataset_mode mixed \
        --soccernet_dir datasets/soccernet/jersey-2023 \
        --tracking_dir datasets/jersey_tracking_v1 \
        --output_dir runs/jersey_unified_mixed \
        --epochs 30 --batch_size 8 --lr 1e-4
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.identity.jersey_model import (
    DigitCompositionalMIL,
    TRANSFORM_TRAIN,
    TRANSFORM_INFERENCE as TRANSFORM_VAL,
    save_full_checkpoint,
    load_full_checkpoint,
)


# ──────────────────────────────────────────────────────────────────────
# Helper: convert jersey number -> head targets
# ──────────────────────────────────────────────────────────────────────

def jersey_to_heads(number: int) -> Tuple[int, int, int]:
    """Convert jersey number (1-99) to (length, tens, ones) targets."""
    if number <= 9:
        return 0, 0, number
    else:
        return 1, number // 10, number % 10


# ──────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ──────────────────────────────────────────────────────────────────────

class SoccerNetJerseyWrapper(Dataset):
    """Wrapper around SoccerNetJerseyDataset that returns (x, len_t, tens_t, ones_t)."""

    def __init__(self, root_dir: str, split: str = "train", K: int = 16,
                 transform=None, exclude_not_visible: bool = True):
        # Import here to avoid circular issues
        from training.identification.soccernet_jersey_loader import SoccerNetJerseyDataset
        self.base = SoccerNetJerseyDataset(
            root_dir=root_dir,
            split=split,
            K=K,
            transform=transform,
            exclude_not_visible=exclude_not_visible,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        crops_tensor, target, player_id = self.base[idx]
        # target: 0-98 (numbers 1-99), -1 should have been filtered
        if target < 0:
            raise RuntimeError(f"Got target={target} for player_id={player_id}. Use exclude_not_visible=True.")
        number = target + 1
        length, tens, ones = jersey_to_heads(number)
        return crops_tensor, torch.tensor(length), torch.tensor(tens), torch.tensor(ones)


class TrackingJerseyDataset(Dataset):
    """Tracking-derived dataset (jersey_tracking_v1). Mirrors train_jersey_digit_mil.py logic."""

    def __init__(self, dataset_dir: str, split: str = "train", K: int = 16,
                 transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.K = K
        self.transform = transform

        # Load records
        tracklets_path = self.dataset_dir / "tracklets.json"
        splits_path = self.dataset_dir / "splits.json"

        with open(tracklets_path) as f:
            all_records = json.load(f)
        with open(splits_path) as f:
            splits_data = json.load(f)

        # Resolve split per record
        train_seqs = {s for s, v in splits_data.get("train", {}).items() if v}
        val_seqs = {s for s, v in splits_data.get("val", {}).items() if v}
        test_seqs = {s for s, v in splits_data.get("test", {}).items() if v}

        import hashlib
        def _is_val(seq):
            h = int(hashlib.md5(seq.encode()).hexdigest(), 16)
            return (h % 100) < 15  # 15% val ratio

        self.records = []
        for rec in all_records:
            seq = rec["sequence"]
            if seq in test_seqs:
                rec["split"] = "test"
            elif seq in val_seqs and _is_val(seq):
                rec["split"] = "val"
            elif seq in train_seqs:
                rec["split"] = "train"
            else:
                rec["split"] = "unknown"

            if rec["split"] == split and rec.get("jersey_number", -1) >= 0:
                self.records.append(rec)

        print(f"Tracking {split}: {len(self.records)} tracklets loaded")

    def _load_crop(self, record, frame_idx):
        crop_rel = record["crop_paths"][frame_idx]
        crop_path = self.dataset_dir / crop_rel
        if not crop_path.exists():
            raise FileNotFoundError(f"Crop missing: {crop_path}")
        from PIL import Image
        img = Image.open(crop_path).convert("RGB")
        return np.array(img)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        jersey = int(record["jersey_number"])
        if jersey < 0:
            jersey = 0
        length, tens, ones = jersey_to_heads(jersey)

        selected = []
        n_frames = min(len(record["frame_ids"]), self.K)
        for i in range(n_frames):
            img = self._load_crop(record, i)
            if self.transform:
                img = self.transform(img)
            selected.append(img)
        if len(selected) == 0:
            raise RuntimeError(f"Tracklet {record['sequence']}/{record['track_id']} has 0 valid crops.")
        while len(selected) < self.K:
            selected.append(selected[random.randrange(len(selected))])
        x = torch.stack(selected[:self.K])
        return x, torch.tensor(length), torch.tensor(tens), torch.tensor(ones)


# ──────────────────────────────────────────────────────────────────────
# Training / validation logic (reused from train_jersey_digit_mil.py)
# ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, len_t, tens_t, ones_t in loader:
        x = x.to(device)
        len_t = len_t.to(device)
        tens_t = tens_t.to(device)
        ones_t = ones_t.to(device)
        optimizer.zero_grad()
        out_len, out_tens, out_ones = model(x)
        loss = criterion(out_len, len_t) + criterion(out_tens, tens_t) + criterion(out_ones, ones_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct_len, correct_tens, correct_ones, correct_num, total = 0, 0, 0, 0, 0
    for x, len_t, tens_t, ones_t in loader:
        x = x.to(device)
        len_t = len_t.to(device)
        tens_t = tens_t.to(device)
        ones_t = ones_t.to(device)
        out_len, out_tens, out_ones = model(x)
        pred_len = out_len.argmax(1)
        pred_tens = out_tens.argmax(1)
        pred_ones = out_ones.argmax(1)
        correct_len += (pred_len == len_t).sum().item()
        correct_tens += (pred_tens == tens_t).sum().item()
        correct_ones += (pred_ones == ones_t).sum().item()
        pred_jersey = pred_ones + pred_tens * 10
        pred_jersey = torch.where(pred_len == 0, pred_ones, pred_jersey)
        gt_jersey = ones_t + tens_t * 10
        gt_jersey = torch.where(len_t == 0, ones_t, gt_jersey)
        correct_num += (pred_jersey == gt_jersey).sum().item()
        total += len_t.size(0)
    return {
        "length_acc": correct_len / max(total, 1),
        "tens_acc": correct_tens / max(total, 1),
        "ones_acc": correct_ones / max(total, 1),
        "jersey_acc": correct_num / max(total, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified jersey MIL training")
    parser.add_argument("--dataset_mode", type=str, required=True,
                        choices=["soccernet", "tracking", "mixed"],
                        help="Training data source")
    parser.add_argument("--soccernet_dir", type=str, default="datasets/soccernet/jersey-2023",
                        help="Path to SoccerNet Jersey 2023 root")
    parser.add_argument("--tracking_dir", type=str, default="datasets/jersey_tracking_v1",
                        help="Path to tracking-derived dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint (for fine-tuning)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    log_path = output_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    log.info(f"Mode: {args.dataset_mode}")
    log.info(f"K={args.K}, epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}")

    # ─── Build datasets ───
    train_datasets = []
    val_datasets = []

    if args.dataset_mode in ("soccernet", "mixed"):
        train_datasets.append(SoccerNetJerseyWrapper(
            args.soccernet_dir, split="train", K=args.K,
            transform=TRANSFORM_TRAIN, exclude_not_visible=True,
        ))
        val_datasets.append(SoccerNetJerseyWrapper(
            args.soccernet_dir, split="test", K=args.K,
            transform=TRANSFORM_VAL, exclude_not_visible=True,
        ))

    if args.dataset_mode in ("tracking", "mixed"):
        train_datasets.append(TrackingJerseyDataset(
            args.tracking_dir, split="train", K=args.K,
            transform=TRANSFORM_TRAIN,
        ))
        val_datasets.append(TrackingJerseyDataset(
            args.tracking_dir, split="val", K=args.K,
            transform=TRANSFORM_VAL,
        ))

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    log.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ─── Model ───
    model = DigitCompositionalMIL(backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0.0

    # Load pretrained (Phase B style)
    if args.pretrained and Path(args.pretrained).exists():
        log.info(f"Loading pretrained checkpoint: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            log.info("  Loaded model_state_dict from pretrained checkpoint")
        else:
            model.load_state_dict(ckpt, strict=True)
            log.info("  Loaded raw state_dict from pretrained checkpoint")

    # Resume (full training state)
    if args.resume and Path(args.resume).exists():
        log.info(f"Resuming from {args.resume}")
        resume_info = load_full_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        start_epoch = resume_info["epoch"] + 1
        best_acc = resume_info["best_acc"]
        if best_acc < 0:
            log.info("  Legacy checkpoint: running validation pass to discover best_acc...")
            val_metrics = validate(model, val_loader, device)
            best_acc = val_metrics["jersey_acc"]
            log.info(f"  Discovered best_acc={best_acc:.4f}")
        else:
            log.info(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.4f}")

    # ─── Training loop ───
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        log.info(
            f"Epoch {epoch+1}/{args.epochs} | loss: {train_loss:.4f} | "
            f"jersey_acc: {val_metrics['jersey_acc']:.4f} | "
            f"len: {val_metrics['length_acc']:.4f} | "
            f"tens: {val_metrics['tens_acc']:.4f} | "
            f"ones: {val_metrics['ones_acc']:.4f}"
        )

        if val_metrics["jersey_acc"] > best_acc:
            best_acc = val_metrics["jersey_acc"]
            save_full_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                output_dir / "best.pt",
            )
            log.info(f"  -> New best: {best_acc:.4f}")

        if (epoch + 1) % 5 == 0:
            save_full_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                output_dir / f"epoch_{epoch+1}.pt",
            )

    log.info(f"\nBest val jersey accuracy: {best_acc:.4f}")
    save_full_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, best_acc,
        output_dir / "final.pt",
    )
    log.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
