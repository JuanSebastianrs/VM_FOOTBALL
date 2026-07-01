# core/scanning/models/train_orientation_model.py
"""
Bucle de entrenamiento del modelo temporal de orientacion.

`Trainer` encapsula datasets, optimizador, loss y checkpointing para que el
script CLI (scripts/scanning/train_orientation_model.py) sea una capa delgada.

Guarda:
  outputs/scanning/checkpoints/best_orientation_<model>.pt
  outputs/scanning/train_logs.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .evaluate_orientation_model import evaluate_model
from .orientation_dataset import FEATURE_DIM, OrientationSequenceDataset
from .orientation_tcn import OrientationLoss, build_model


class Trainer:
    def __init__(self, config: dict):
        tr = config.get("training", config)
        data = config.get("data", {})
        self.cfg = tr
        self.device = tr.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_bins = int(tr.get("num_orientation_bins", 8))
        self.seq_len = int(tr.get("sequence_length", 16))
        self.epochs = int(tr.get("epochs", 50))
        self.ckpt_dir = Path(tr.get("checkpoint_dir", "outputs/scanning/checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(data.get("output_dir", "outputs/scanning"))

        labels = data.get("annotation_csv", "data/annotations/orientation_labels.csv")
        feats = config.get("features_parquet")  # opcional

        self.train_ds = OrientationSequenceDataset(
            labels, feats, sequence_length=self.seq_len, num_bins=self.num_bins,
            split="train", val_split=tr.get("val_split", 0.2),
            augment=tr.get("augment"), seed=tr.get("seed", 42))
        self.val_ds = OrientationSequenceDataset(
            labels, feats, sequence_length=self.seq_len, num_bins=self.num_bins,
            split="val", val_split=tr.get("val_split", 0.2),
            seed=tr.get("seed", 42), norm_stats=self.train_ds.norm_stats())

        bs = int(tr.get("batch_size", 32))
        nw = int(tr.get("num_workers", 0))
        if tr.get("balance_bins", False) and len(self.train_ds) > 0:
            w = self.train_ds.sample_weights()
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
            self.train_loader = DataLoader(self.train_ds, batch_size=bs,
                                           sampler=sampler, num_workers=nw)
        else:
            self.train_loader = DataLoader(self.train_ds, batch_size=bs,
                                           shuffle=True, num_workers=nw)
        self.val_loader = DataLoader(self.val_ds, batch_size=bs, shuffle=False,
                                     num_workers=nw)

        self.model = build_model(
            tr.get("model", "tcn"), FEATURE_DIM, num_bins=self.num_bins,
            hidden_dim=tr.get("hidden_dim", 64), num_blocks=tr.get("num_tcn_blocks", 4),
            dropout=tr.get("dropout", 0.2)).to(self.device)
        self.criterion = OrientationLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tr.get("learning_rate", 3e-4),
            weight_decay=tr.get("weight_decay", 1e-4))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.epochs))

    def _train_epoch(self) -> float:
        self.model.train()
        losses = []
        for batch in self.train_loader:
            x = batch["features"].to(self.device)
            theta = batch["theta_gt"].to(self.device)
            bin_gt = batch["orientation_bin"].to(self.device)
            conf = batch["confidence"].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            ld = self.criterion(out, theta, bin_gt, conf)
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            losses.append(float(ld["total"].item()))
        return float(np.mean(losses)) if losses else float("nan")

    def fit(self) -> dict:
        if len(self.train_ds) == 0:
            raise RuntimeError("Dataset de entrenamiento vacio. Revisa el labels CSV.")
        log_path = self.output_dir / "train_logs.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        best_mae = float("inf")
        best_path = self.ckpt_dir / f"best_orientation_{self.cfg.get('model','tcn')}.pt"
        rows = []
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            self.scheduler.step()
            metrics = (evaluate_model(self.model, self.val_loader, self.device,
                                      self.num_bins)
                       if len(self.val_ds) > 0 else {"angular_mae_deg": float("nan")})
            mae = metrics.get("angular_mae_deg", float("nan"))
            row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            rows.append(row)
            print(f"[ep {epoch:03d}] loss {train_loss:.4f} | val MAE {mae:.1f} deg "
                  f"| acc@22.5 {metrics.get('accuracy_at_22_5_deg', float('nan')):.3f} "
                  f"| bin_acc {metrics.get('orientation_bin_accuracy', float('nan')):.3f}")
            if not np.isnan(mae) and mae < best_mae:
                best_mae = mae
                self._save(best_path, epoch, metrics)
        # log
        if rows:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(
                    {k for r in rows for k in r}))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Mejor MAE val: {best_mae:.2f} deg -> {best_path}")
        return {"best_mae_deg": best_mae, "checkpoint": str(best_path)}

    def _save(self, path: Path, epoch: int, metrics: dict) -> None:
        mean, std = self.train_ds.norm_stats()
        torch.save({
            "model_state": self.model.state_dict(),
            "model_name": self.cfg.get("model", "tcn"),
            "feature_dim": FEATURE_DIM,
            "num_bins": self.num_bins,
            "sequence_length": self.seq_len,
            "hidden_dim": self.cfg.get("hidden_dim", 64),
            "num_blocks": self.cfg.get("num_tcn_blocks", 4),
            "norm_mean": mean, "norm_std": std,
            "epoch": epoch, "metrics": metrics,
        }, path)
