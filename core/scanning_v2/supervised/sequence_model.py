# core/scanning_v2/supervised/sequence_model.py
"""
Arquitectura SECUENCIAL para el clasificador de scanning V2.

`GRUSequenceClassifier` es un estimador sklearn-compatible (fit/predict/
predict_proba, clonable) que consume las columnas de secuencia generadas por el
FeatureExtractor (`seq_cos_XX`, `seq_sin_XX`, `seq_valid_XX`: la serie de yaw
re-muestreada a T pasos sobre la ventana previa a la recepcion) y, opcionalmente,
las features tabulares agregadas.

Diseno:
  - GRU bidireccional sobre (cos, sin, valid) por paso temporal -> estado final.
  - Rama tabular (mediana-imputada + estandarizada) -> MLP pequena.
  - Fusion -> logit unico. BCEWithLogitsLoss con pos_weight (clases desbalanceadas).
  - Early stopping interno con split estratificado; restaura el mejor estado.

El yaw se representa como (cos, sin) porque es una variable CIRCULAR: evita la
discontinuidad en +-180 grados. `valid` marca pasos sin estimacion de pose
(el GRU aprende a ignorarlos; no se interpola informacion inexistente).

Sigue siendo orientacion APROXIMADA de cabeza/cuerpo, NO gaze real.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

_SEQ_RE = re.compile(r"^seq_(cos|sin|valid)_(\d+)$")


class _GRUNet(nn.Module):
    """Red interna: GRU sobre (cos,sin,valid) + rama tabular opcional.
    Definida a nivel de modulo para que el modelo sea picklable (joblib)."""

    def __init__(self, hidden_size: int, num_layers: int, bidirectional: bool,
                 tabular_hidden: int, dropout: float, tab_dim: int):
        super().__init__()
        self.gru = nn.GRU(3, hidden_size, num_layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.tab = (nn.Sequential(nn.Linear(tab_dim, tabular_hidden), nn.ReLU(),
                                  nn.Dropout(dropout)) if tab_dim else None)
        hs = hidden_size * (2 if bidirectional else 1)
        fused = hs + (tabular_hidden if tab_dim else 0)
        self.head = nn.Sequential(nn.Linear(fused, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, seq, tab):
        out, _ = self.gru(seq)
        h = out[:, -1, :]                     # estado final (bi: concat fwd/bwd)
        if self.tab is not None:
            h = torch.cat([h, self.tab(tab)], dim=1)
        return self.head(h).squeeze(-1)


def sequence_columns(columns) -> dict:
    """Agrupa columnas de secuencia por tipo -> lista ordenada por paso."""
    found = {"cos": {}, "sin": {}, "valid": {}}
    for c in columns:
        m = _SEQ_RE.match(str(c))
        if m:
            found[m.group(1)][int(m.group(2))] = c
    steps = sorted(found["cos"])
    if not steps or sorted(found["sin"]) != steps or sorted(found["valid"]) != steps:
        return {}
    return {k: [found[k][s] for s in steps] for k in ("cos", "sin", "valid")}


class GRUSequenceClassifier(BaseEstimator, ClassifierMixin):
    """GRU temporal + rama tabular. sklearn-compatible (torch por debajo)."""

    def __init__(self, hidden_size: int = 32, num_layers: int = 1,
                 bidirectional: bool = True, tabular_hidden: int = 32,
                 dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 32, max_epochs: int = 300, patience: int = 30,
                 val_fraction: float = 0.2, use_tabular: bool = True,
                 class_weight: Optional[str] = "balanced",
                 random_state: int = 42, device: str = "auto"):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.tabular_hidden = tabular_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.use_tabular = use_tabular
        self.class_weight = class_weight
        self.random_state = random_state
        self.device = device

    # ------------------------------------------------------------------
    def _resolve_device(self):
        import torch
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _split_inputs(self, X):
        """DataFrame/array -> (seq (N,T,3) float32, tab (N,D) float32)."""
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self._fit_columns_)
        cols = sequence_columns(X.columns)
        if not cols:
            raise ValueError(
                "GRUSequenceClassifier requiere columnas seq_cos_XX/seq_sin_XX/"
                "seq_valid_XX (activa dataset.include_sequence_features).")
        cos = X[cols["cos"]].to_numpy(dtype=np.float32)
        sin = X[cols["sin"]].to_numpy(dtype=np.float32)
        val = X[cols["valid"]].to_numpy(dtype=np.float32)
        val = np.nan_to_num(val, nan=0.0)
        cos = np.nan_to_num(cos, nan=0.0) * val
        sin = np.nan_to_num(sin, nan=0.0) * val
        seq = np.stack([cos, sin, val], axis=-1)          # (N, T, 3)

        seq_names = set(cols["cos"]) | set(cols["sin"]) | set(cols["valid"])
        tab_cols = [c for c in X.columns
                    if c not in seq_names and np.issubdtype(
                        np.asarray(X[c]).dtype, np.number)]
        tab = X[tab_cols].to_numpy(dtype=np.float32) if (self.use_tabular and tab_cols) \
            else np.zeros((len(X), 0), dtype=np.float32)
        return seq, tab, tab_cols

    def _prep_tabular(self, tab, fit: bool):
        if tab.shape[1] == 0:
            return tab
        if fit:
            self.tab_median_ = np.nanmedian(tab, axis=0)
            self.tab_median_ = np.nan_to_num(self.tab_median_, nan=0.0)
        idx = np.where(np.isnan(tab))
        tab = tab.copy()
        tab[idx] = np.take(self.tab_median_, idx[1])
        if fit:
            self.tab_mean_ = tab.mean(axis=0)
            self.tab_std_ = tab.std(axis=0) + 1e-8
        return (tab - self.tab_mean_) / self.tab_std_

    def _build_net(self, tab_dim: int):
        return _GRUNet(self.hidden_size, self.num_layers, self.bidirectional,
                       self.tabular_hidden, self.dropout, tab_dim)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        import pandas as pd
        import torch

        y = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce")).astype(int)
        self.classes_ = np.unique(y)
        if isinstance(X, pd.DataFrame):
            self._fit_columns_ = list(X.columns)
        seq, tab, self.tab_columns_ = self._split_inputs(X)
        tab = self._prep_tabular(tab, fit=True)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = self._resolve_device()

        # split interno para early stopping (estratificado si es posible)
        n = len(y)
        idx = np.arange(n)
        val_idx = np.array([], dtype=int)
        if 0 < self.val_fraction < 1 and n >= 10 and len(np.unique(y)) == 2:
            from sklearn.model_selection import train_test_split
            try:
                idx, val_idx = train_test_split(
                    idx, test_size=self.val_fraction,
                    random_state=self.random_state, stratify=y)
            except ValueError:
                pass

        net = self._build_net(tab.shape[1]).to(device)
        pos = max(1, int((y[idx] == 1).sum()))
        neg = max(1, int((y[idx] == 0).sum()))
        pos_weight = torch.tensor(
            [neg / pos if self.class_weight == "balanced" else 1.0],
            dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        def _tensors(rows):
            return (torch.from_numpy(seq[rows]).to(device),
                    torch.from_numpy(tab[rows]).to(device),
                    torch.from_numpy(y[rows].astype(np.float32)).to(device))

        best_state, best_val, since_best = None, np.inf, 0
        for _epoch in range(self.max_epochs):
            net.train()
            perm = np.random.permutation(idx)
            for i in range(0, len(perm), self.batch_size):
                bs, bt, by = _tensors(perm[i:i + self.batch_size])
                opt.zero_grad()
                loss = loss_fn(net(bs, bt), by)
                loss.backward()
                opt.step()
            # validacion (o loss de train si no hay split)
            net.eval()
            with torch.no_grad():
                rows = val_idx if len(val_idx) else idx
                vs, vt, vy = _tensors(rows)
                vloss = float(loss_fn(net(vs, vt), vy))
            if vloss < best_val - 1e-5:
                best_val, since_best = vloss, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
            else:
                since_best += 1
                if since_best >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net.cpu()
        self.n_features_in_ = seq.shape[1] * 3 + tab.shape[1]
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        import torch
        seq, tab, _ = self._split_inputs(X)
        tab = self._prep_tabular(tab, fit=False)
        with torch.no_grad():
            logit = self.net_(torch.from_numpy(seq), torch.from_numpy(tab))
            p1 = torch.sigmoid(logit).numpy()
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
