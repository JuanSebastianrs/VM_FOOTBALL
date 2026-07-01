# core/scanning/models/orientation_tcn.py
"""
Modelo temporal entrenable de orientacion visual.

Predice, a partir de una secuencia de features por frame (keypoints + contexto),
la orientacion visual aproximada del frame final de la ventana, con tres cabezas:

  A. Regresion angular : (sin, cos)  -> theta
  B. Clasificacion     : orientation_bin_8 (o 16)
  C. Confianza         : confidence_pred en [0, 1]

Arquitectura por defecto: TCN pequena (convoluciones 1D dilatadas con conexiones
residuales). Alternativa: BiLSTM. Ambas comparten las cabezas de salida.

Loss (ver `OrientationLoss`):
  circular_loss = 1 - cos(theta_pred - theta_gt)
  bin_loss      = CE(bin_pred, bin_gt)
  conf_loss     = BCE(conf_pred, conf_gt)
  total         = circular + 0.5*bin + 0.2*conf
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilidades circulares (torch)
# ---------------------------------------------------------------------------

def angle_to_sincos(theta: torch.Tensor) -> torch.Tensor:
    """(...,) -> (..., 2) con [sin, cos]."""
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)


def sincos_to_angle(sincos: torch.Tensor) -> torch.Tensor:
    """(..., 2) [sin, cos] -> angulo en [-pi, pi]."""
    return torch.atan2(sincos[..., 0], sincos[..., 1])


def circular_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Diferencia angular con signo en [-pi, pi]."""
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def angular_mae_deg(pred_theta: torch.Tensor, gt_theta: torch.Tensor) -> float:
    """MAE angular en grados."""
    d = torch.abs(circular_delta(pred_theta, gt_theta))
    return float(torch.rad2deg(d).mean().item())


# ---------------------------------------------------------------------------
# TCN
# ---------------------------------------------------------------------------

class _TemporalBlock(nn.Module):
    """Bloque TCN: dos conv1d dilatadas causales + residual."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int,
                 dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation, padding=pad)
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). El padding causal recorta el exceso por la derecha.
        out = self.conv1(x)[:, :, :-self.pad] if self.pad else self.conv1(x)
        out = self.drop(F.relu(self.norm1(out)))
        out = self.conv2(out)[:, :, :-self.pad] if self.pad else self.conv2(out)
        out = self.drop(F.relu(self.norm2(out)))
        res = x if self.down is None else self.down(x)
        return F.relu(out + res)


class _OutputHeads(nn.Module):
    def __init__(self, dim: int, num_bins: int):
        super().__init__()
        self.sincos = nn.Linear(dim, 2)
        self.bins = nn.Linear(dim, num_bins)
        self.conf = nn.Linear(dim, 1)

    def forward(self, feat: torch.Tensor) -> dict:
        sincos = F.normalize(self.sincos(feat), dim=-1)   # vector unitario
        return {
            "sincos": sincos,
            "theta": sincos_to_angle(sincos),
            "bin_logits": self.bins(feat),
            "confidence": torch.sigmoid(self.conf(feat)).squeeze(-1),
        }


class OrientationTCN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64,
                 num_blocks: int = 4, kernel: int = 3, dropout: float = 0.2,
                 num_bins: int = 8):
        super().__init__()
        layers = []
        c_in = feature_dim
        for i in range(num_blocks):
            layers.append(_TemporalBlock(c_in, hidden_dim, kernel,
                                         dilation=2 ** i, dropout=dropout))
            c_in = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.heads = _OutputHeads(hidden_dim, num_bins)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, F) -> (B, F, T)
        h = self.tcn(x.transpose(1, 2))
        feat = h[:, :, -1]                # ultimo paso temporal
        return self.heads(feat)


class OrientationBiLSTM(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, num_bins: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.heads = _OutputHeads(hidden_dim * 2, num_bins)

    def forward(self, x: torch.Tensor) -> dict:
        out, _ = self.lstm(x)
        return self.heads(out[:, -1, :])


def build_model(name: str, feature_dim: int, num_bins: int = 8,
                hidden_dim: int = 64, num_blocks: int = 4,
                dropout: float = 0.2) -> nn.Module:
    if name == "tcn":
        return OrientationTCN(feature_dim, hidden_dim, num_blocks,
                              dropout=dropout, num_bins=num_bins)
    if name == "bilstm":
        return OrientationBiLSTM(feature_dim, hidden_dim, dropout=dropout,
                                 num_bins=num_bins)
    raise ValueError(f"modelo desconocido: {name!r}")


# ---------------------------------------------------------------------------
# Loss multitarea
# ---------------------------------------------------------------------------

class OrientationLoss(nn.Module):
    def __init__(self, w_bin: float = 0.5, w_conf: float = 0.2):
        super().__init__()
        self.w_bin = w_bin
        self.w_conf = w_conf

    def forward(self, out: dict, theta_gt: torch.Tensor,
                bin_gt: torch.Tensor, conf_gt: torch.Tensor) -> dict:
        # circular: 1 - cos(delta)
        circ = (1.0 - torch.cos(out["theta"] - theta_gt)).mean()
        bin_loss = F.cross_entropy(out["bin_logits"], bin_gt)
        conf_loss = F.binary_cross_entropy(out["confidence"], conf_gt)
        total = circ + self.w_bin * bin_loss + self.w_conf * conf_loss
        return {"total": total, "circular": circ, "bin": bin_loss, "conf": conf_loss}
