# core/scanning/models/evaluate_orientation_model.py
"""
Metricas de evaluacion del modelo de orientacion.

  - angular_mae_deg
  - accuracy@22.5 / accuracy@45  (fraccion con error angular < umbral)
  - orientation_bin_accuracy
  - confidence_mae (calibracion basica)
  - per-bin accuracy
  - valid_fraction (muestras procesadas)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from .orientation_tcn import circular_delta


@torch.no_grad()
def evaluate_model(model, loader, device: str = "cuda:0",
                   num_bins: int = 8) -> Dict[str, float]:
    model.eval()
    d_err: List[float] = []
    bin_correct = 0
    bin_total = 0
    conf_err: List[float] = []
    per_bin_correct = np.zeros(num_bins)
    per_bin_total = np.zeros(num_bins)

    for batch in loader:
        x = batch["features"].to(device)
        theta_gt = batch["theta_gt"].to(device)
        bin_gt = batch["orientation_bin"].to(device)
        conf_gt = batch["confidence"].to(device)
        out = model(x)
        d = torch.abs(circular_delta(out["theta"], theta_gt))
        d_err.extend(torch.rad2deg(d).cpu().numpy().tolist())
        pred_bin = out["bin_logits"].argmax(dim=-1)
        bin_correct += int((pred_bin == bin_gt).sum().item())
        bin_total += int(bin_gt.numel())
        conf_err.extend(torch.abs(out["confidence"] - conf_gt).cpu().numpy().tolist())
        for b, c in zip(bin_gt.cpu().numpy(), (pred_bin == bin_gt).cpu().numpy()):
            per_bin_total[b] += 1
            per_bin_correct[b] += int(c)

    d_err = np.asarray(d_err) if d_err else np.array([np.nan])
    metrics = {
        "angular_mae_deg": float(np.nanmean(d_err)),
        "accuracy_at_22_5_deg": float(np.mean(d_err < 22.5)),
        "accuracy_at_45_deg": float(np.mean(d_err < 45.0)),
        "orientation_bin_accuracy": float(bin_correct / max(1, bin_total)),
        "confidence_mae": float(np.mean(conf_err)) if conf_err else 0.0,
        "num_samples": int(bin_total),
    }
    for b in range(num_bins):
        if per_bin_total[b] > 0:
            metrics[f"bin_{b}_accuracy"] = float(per_bin_correct[b] / per_bin_total[b])
    return metrics
