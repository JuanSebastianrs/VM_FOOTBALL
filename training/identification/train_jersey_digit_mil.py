"""
Digit-compositional MIL model for jersey number recognition.
Each tracklet = bag of K frames; shared backbone + per-frame embedding +
attention pooling + three heads: length (1/2), tens digit (0-9), ones digit (0-9).

Usage:
    python training/identification/train_jersey_digit_mil.py \
        --dataset_dir datasets/jersey_tracking_v1 \
        --output_dir runs/jersey_digit_mil \
        --img_size 128 \
        --K 16 \
        --epochs 30 \
        --batch_size 8 \
        --lr 1e-4 \
        --backbone efficientnet_b0 \
        --device cuda:0
"""

import argparse
import json
import math
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.identity.jersey_model import (
    DigitCompositionalMIL,
    TRANSFORM_TRAIN,
    TRANSFORM_INFERENCE as TRANSFORM_VAL,
    save_full_checkpoint,
    load_full_checkpoint,
)


class JerseyTrackletDataset(Dataset):
    def __init__(self, records, root_dir, split="train", K=16, transform=None):
        # records is the list from tracklets.json
        self.records = [r for r in records if r.get("split") == split]
        self.root_dir = Path(root_dir)
        self.K = K
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def _load_crop(self, record, frame_idx):
        # crop_paths are relative to output_dir, stored like "crops/SNMOT-148/1_000281.jpg"
        crop_rel = record["crop_paths"][frame_idx]
        crop_path = self.root_dir / crop_rel
        if not crop_path.exists():
            raise FileNotFoundError(f"Crop missing for tracklet {record['sequence']}/{record['track_id']}: {crop_path}")
        img = Image.open(crop_path).convert("RGB")
        return np.array(img)

    def __getitem__(self, idx):
        record = self.records[idx]
        jersey = int(record["jersey_number"])
        if jersey < 0:
            jersey = 0
        # length target: 0 = 1 digit, 1 = 2 digits (for CrossEntropyLoss over 2 classes)
        if jersey <= 9:
            length = 0
            tens = 0
            ones = jersey
        else:
            length = 1
            tens = jersey // 10
            ones = jersey % 10

        selected = []
        n_frames = min(len(record["frame_ids"]), self.K)
        for i in range(n_frames):
            img = self._load_crop(record, i)
            if self.transform:
                img = self.transform(img)
            selected.append(img)
        if len(selected) == 0:
            raise RuntimeError(f"Tracklet {record['sequence']}/{record['track_id']} has 0 valid crops.")
        # Pad with random existing frames from this tracklet (with-replacement augmentation)
        while len(selected) < self.K:
            selected.append(selected[random.randrange(len(selected))])
        x = torch.stack(selected[:self.K])
        return x, torch.tensor(length), torch.tensor(tens), torch.tensor(ones)


# Model is imported from core.identity.jersey_model — no local definition


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


def build_splits(tracklets_path, splits_json_path, val_ratio=0.15):
    with open(tracklets_path) as f:
        records = json.load(f)
    with open(splits_json_path) as f:
        splits_data = json.load(f)
    train_seqs = {s for s, v in splits_data.get("train", {}).items() if v}
    val_seqs = {s for s, v in splits_data.get("val", {}).items() if v}
    test_seqs = {s for s, v in splits_data.get("test", {}).items() if v}

    # Stratified val split: from sequences in both train and val, assign some to val
    # Use hash of sequence name for deterministic split
    import hashlib
    def _is_val(seq):
        h = int(hashlib.md5(seq.encode()).hexdigest(), 16)
        return (h % 100) < int(val_ratio * 100)

    for rec in records:
        seq = rec["sequence"]
        if seq in test_seqs:
            rec["split"] = "test"
        elif seq in val_seqs and _is_val(seq):
            rec["split"] = "val"
        elif seq in train_seqs:
            rec["split"] = "train"
        else:
            rec["split"] = "unknown"
    return [r for r in records if r["split"] != "unknown"]


def main():
    parser = argparse.ArgumentParser(description="Train digit-compositional MIL jersey model")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    import logging
    log_path = args.log_file or str(output_dir / "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    tracklets_path = Path(args.dataset_dir) / "tracklets.json"
    splits_json_path = Path(args.dataset_dir) / "splits.json"
    records = build_splits(tracklets_path, splits_json_path)

    log.info(f"Dataset: {len(records)} tracklets")
    log.info(f"  train: {sum(1 for r in records if r['split']=='train')}")
    log.info(f"  val: {sum(1 for r in records if r['split']=='val')}")
    log.info(f"  test: {sum(1 for r in records if r['split']=='test')}")

    train_ds = JerseyTrackletDataset(records, args.dataset_dir, split="train", K=args.K, transform=TRANSFORM_TRAIN)
    val_ds = JerseyTrackletDataset(records, args.dataset_dir, split="val", K=args.K, transform=TRANSFORM_VAL)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DigitCompositionalMIL(backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0.0
    if args.resume and Path(args.resume).exists():
        log.info(f"Resuming from {args.resume}")
        resume_info = load_full_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        start_epoch = resume_info["epoch"] + 1
        best_acc = resume_info["best_acc"]
        # Legacy checkpoint: best_acc is unknown (-1.0), discover via validation
        if best_acc < 0:
            log.info("  Legacy checkpoint detected (best_acc unknown). Running validation pass...")
            val_metrics = validate(model, val_loader, device)
            best_acc = val_metrics["jersey_acc"]
            log.info(f"  Discovered best_acc={best_acc:.4f} from validation")
        else:
            log.info(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        log.info(f"Epoch {epoch+1}/{args.epochs} | loss: {train_loss:.4f} | "
                 f"jersey_acc: {val_metrics['jersey_acc']:.4f} | "
                 f"len: {val_metrics['length_acc']:.4f} | "
                 f"tens: {val_metrics['tens_acc']:.4f} | "
                 f"ones: {val_metrics['ones_acc']:.4f}")

        if val_metrics["jersey_acc"] > best_acc:
            best_acc = val_metrics["jersey_acc"]
            # Save full checkpoint for best model
            save_full_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                output_dir / "best.pt",
            )
            log.info(f"  -> New best: {best_acc:.4f}")

        # Save periodic checkpoint every 5 epochs (full state)
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