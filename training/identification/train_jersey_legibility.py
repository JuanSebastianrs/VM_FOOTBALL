"""
Train Jersey Legibility Classifier.
Trains a binary classifier (EfficientNet-B0) to predict if a crop shows a visible jersey number.

Usage:
    python training/identification/train_jersey_legibility.py \
        --dataset_dir datasets/jersey_legibility_v1 \
        --output_dir runs/jersey_legibility_v1 \
        --epochs 10 \
        --batch_size 16 \
        --lr 3e-4
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, confusion_matrix, auc as sklearn_auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.identity.jersey_model import TRANSFORM_TRAIN, TRANSFORM_INFERENCE, LegibilityClassifier


class LegibilityDataset(Dataset):
    def __init__(self, filenames, root_dir, df_metadata, transform=None):
        self.filenames = filenames
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.meta = df_metadata.set_index("filename")["label"].to_dict()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = self.root_dir / filename
        
        # Rigorous check: raise FileNotFoundError if image doesn't exist
        if not img_path.exists():
            raise FileNotFoundError(f"Legibility training image not found: {img_path}")
            
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.meta.get(filename, 0)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy())
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Metrics at threshold = 0.5
    binary_preds = (all_preds >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, binary_preds, pos_label=1, average="binary", zero_division=0
    )
    
    try:
        auc_roc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_roc = 0.5
        
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(all_targets, all_preds)
        auc_pr = sklearn_auc(rec_curve, prec_curve)
    except ValueError:
        auc_pr = 0.0

    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return {
        "val_loss": total_loss / len(loader),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train Legibility Classifier")
    parser.add_argument("--dataset_dir", type=str, default="datasets/jersey_legibility_v1")
    parser.add_argument("--output_dir", type=str, default="runs/jersey_legibility_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Strict Seed Initialization
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load splits and metadata
    dataset_path = Path(args.dataset_dir)
    with open(dataset_path / "splits.json") as f:
        splits = json.load(f)
        
    df_metadata = pd.read_csv(dataset_path / "metadata.csv")
    
    train_filenames = splits["train"]
    val_filenames = splits["val"]
    
    print(f"Dataset summary:")
    print(f"  Train: {len(train_filenames)} samples")
    print(f"  Val: {len(val_filenames)} samples")
    
    # 2. Balanced Loss (pos_weight) computation
    train_metadata = df_metadata[df_metadata["filename"].isin(train_filenames)]
    train_labels = train_metadata["label"].values
    n_positive = int(sum(train_labels == 1))
    n_negative = int(sum(train_labels == 0))
    pos_weight_val = n_negative / max(n_positive, 1)
    print(f"Train labels balance: positives={n_positive}, negatives={n_negative}")
    print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val:.4f}")
    
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # 3. Write rich dataset_stats.json
    stats = {
        "dataset_dir": str(dataset_path),
        "total_samples": len(df_metadata),
        "label_0": {
            "total": int(sum(df_metadata["label"] == 0)),
            "train": int(sum(train_metadata["label"] == 0)),
            "val": int(sum(df_metadata[df_metadata["filename"].isin(val_filenames)]["label"] == 0)),
        },
        "label_1": {
            "total": int(sum(df_metadata["label"] == 1)),
            "train": int(sum(train_metadata["label"] == 1)),
            "val": int(sum(df_metadata[df_metadata["filename"].isin(val_filenames)]["label"] == 1)),
        },
        "sources": {}
    }
    for src in df_metadata["source"].unique():
        src_df = df_metadata[df_metadata["source"] == src]
        stats["sources"][src] = {
            "total": len(src_df),
            "label_0": int(sum(src_df["label"] == 0)),
            "label_1": int(sum(src_df["label"] == 1)),
        }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset stats saved to {output_dir / 'dataset_stats.json'}")

    train_ds = LegibilityDataset(train_filenames, args.dataset_dir, df_metadata, transform=TRANSFORM_TRAIN)
    val_ds = LegibilityDataset(val_filenames, args.dataset_dir, df_metadata, transform=TRANSFORM_INFERENCE)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Pretrained=True enables Transfer Learning from ImageNet
    model = LegibilityClassifier(pretrained=True).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_f1 = 0.0
    best_metrics = None
    
    print("\nStarting training...", flush=True)
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device, criterion)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Precision: {val_metrics['precision']:.1%} | "
              f"Recall: {val_metrics['recall']:.1%} | "
              f"F1: {val_metrics['f1']:.1%} | "
              f"AUC-ROC: {val_metrics['auc_roc']:.4f}", flush=True)
              
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "config": vars(args)
            }, output_dir / "best.pt")
            print(f"  -> Saved best model checkpoint with F1: {best_f1:.1%}", flush=True)
            
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs - 1,
        "config": vars(args)
    }, output_dir / "final.pt")
    
    # 4. Save detailed validation metrics for best epoch
    if best_metrics:
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"Best metrics saved to {output_dir / 'metrics.json'}")
    
    print("\nTraining completed.", flush=True)
    print(f"Best Val F1-score: {best_f1:.1%}", flush=True)


if __name__ == "__main__":
    main()
