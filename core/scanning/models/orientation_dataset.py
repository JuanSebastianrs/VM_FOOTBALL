# core/scanning/models/orientation_dataset.py
"""
Dataset PyTorch de secuencias de orientacion.

Une dos fuentes:
  - labels CSV    (data/annotations/orientation_labels.csv): una fila por muestra
    etiquetada con orientation_angle_deg / orientation_bin_8 / confidence /
    visibility / scan_label en un (video_id, frame_id, track_id).
  - features store (parquet): features POR FRAME (keypoints + contexto) para cada
    (video_id, frame_id, track_id). Lo produce el script de construccion del
    dataset (build_orientation_annotation_dataset.py) corriendo el extractor de
    pose sobre los crops.

Para cada muestra etiquetada se arma una secuencia de `sequence_length` frames
que TERMINA en el frame etiquetado (para ese video+track), con padding al frente
si faltan frames. El esquema de features (`FEATURE_COLUMNS`) es compartido por el
builder y el dataset para que siempre coincidan.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Landmarks canonicos (mismo orden que keypoint_extractor.CANONICAL_LANDMARKS)
LANDMARKS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
]

_KP_COLUMNS: List[str] = []
for _n in LANDMARKS:
    _KP_COLUMNS += [f"kp_{_n}_x", f"kp_{_n}_y", f"kp_{_n}_v"]

_CONTEXT_COLUMNS = [
    "bbox_w", "bbox_h", "vx", "vy",
    "run_sin", "run_cos", "ballrel_sin", "ballrel_cos",
    "x_field", "y_field", "dist_ball", "time_to_reception",
]

FEATURE_COLUMNS = _KP_COLUMNS + _CONTEXT_COLUMNS
FEATURE_DIM = len(FEATURE_COLUMNS)            # 9*3 + 12 = 39
KEY_COLUMNS = ["video_id", "frame_id", "track_id"]


# ---------------------------------------------------------------------------
# Construccion de features (compartida con el builder de anotaciones)
# ---------------------------------------------------------------------------

def build_feature_dict(
    pose: dict,
    bbox: np.ndarray,
    velocity: Optional[Tuple[float, float]],
    run_angle: Optional[float],
    ball_relative_angle: Optional[float],
    field_xy: Optional[Tuple[float, float]],
    dist_ball_m: Optional[float],
    time_to_reception: float = 0.0,
    frame_size: Tuple[int, int] = (1920, 1080),
) -> Dict[str, float]:
    """Construye una fila de features (FEATURE_COLUMNS) ya normalizada.

    Keypoints en coords normalizadas del crop [0,1]; bbox y velocidad
    normalizadas por el tamano de frame; campos de cancha por las dimensiones
    de la cancha; angulos como (sin, cos).
    """
    feat: Dict[str, float] = {c: 0.0 for c in FEATURE_COLUMNS}
    lm = pose.get("landmarks", {}) if pose else {}
    for n in LANDMARKS:
        if n in lm:
            x, y, _z, v = lm[n]
            feat[f"kp_{n}_x"] = float(x)
            feat[f"kp_{n}_y"] = float(y)
            feat[f"kp_{n}_v"] = float(v)
    W, H = frame_size
    feat["bbox_w"] = float((bbox[2] - bbox[0]) / W)
    feat["bbox_h"] = float((bbox[3] - bbox[1]) / H)
    if velocity is not None:
        feat["vx"] = float(velocity[0] / W)
        feat["vy"] = float(velocity[1] / H)
    if run_angle is not None:
        feat["run_sin"], feat["run_cos"] = math.sin(run_angle), math.cos(run_angle)
    if ball_relative_angle is not None:
        feat["ballrel_sin"] = math.sin(ball_relative_angle)
        feat["ballrel_cos"] = math.cos(ball_relative_angle)
    if field_xy is not None:
        feat["x_field"] = float(field_xy[0] / 52.5)
        feat["y_field"] = float(field_xy[1] / 34.0)
    if dist_ball_m is not None:
        feat["dist_ball"] = float(min(dist_ball_m, 60.0) / 60.0)
    feat["time_to_reception"] = float(time_to_reception)
    return feat


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrientationSequenceDataset(Dataset):
    def __init__(
        self,
        labels_csv: str | Path,
        features_parquet: Optional[str | Path] = None,
        sequence_length: int = 16,
        num_bins: int = 8,
        split: str = "train",
        val_split: float = 0.2,
        visibility_filter: Optional[List[str]] = None,
        augment: Optional[dict] = None,
        seed: int = 42,
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.sequence_length = int(sequence_length)
        self.num_bins = int(num_bins)
        self.split = split
        self.augment = augment if (augment and split == "train") else None

        labels = pd.read_csv(labels_csv)
        if visibility_filter:
            labels = labels[labels["visibility"].isin(visibility_filter)]
        labels = labels.dropna(subset=["orientation_angle_deg"]).reset_index(drop=True)

        # features store
        if features_parquet is not None and Path(features_parquet).exists():
            self.features = pd.read_parquet(features_parquet)
        else:
            # fallback: usar solo el contexto presente en el CSV (sin keypoints)
            self.features = labels.copy()
            for c in FEATURE_COLUMNS:
                if c not in self.features.columns:
                    self.features[c] = 0.0
        # indexar features por (video_id, track_id) -> df ordenado por frame
        self._fidx: Dict[Tuple[str, int], pd.DataFrame] = {}
        for (vid, tid), g in self.features.groupby(["video_id", "track_id"]):
            self._fidx[(str(vid), int(tid))] = g.sort_values("frame_id").reset_index(drop=True)

        # split determinista por video_id (evita fuga temporal entre splits)
        rng = np.random.default_rng(seed)
        vids = sorted(labels["video_id"].astype(str).unique())
        if len(vids) > 1:
            rng.shuffle(vids)
            n_val = max(1, int(round(len(vids) * val_split)))
            val_vids = set(vids[:n_val])
            mask = labels["video_id"].astype(str).isin(val_vids)
            labels = labels[mask] if split == "val" else labels[~mask]
        else:
            idx = np.arange(len(labels)); rng.shuffle(idx)
            n_val = int(round(len(labels) * val_split))
            sel = idx[:n_val] if split == "val" else idx[n_val:]
            labels = labels.iloc[sel]
        self.labels = labels.reset_index(drop=True)

        # estadisticas de normalizacion (calculadas en train, reusadas en val)
        if norm_stats is not None:
            self.feat_mean, self.feat_std = norm_stats
        else:
            X = self.features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            self.feat_mean = X.mean(axis=0)
            # piso del std: columnas casi-constantes (p.ej. landmarks invisibles
            # siempre en 0) no deben amplificar el jitter de augmentacion.
            self.feat_std = np.maximum(X.std(axis=0), 0.05).astype(np.float32)

    def norm_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.feat_mean, self.feat_std

    def __len__(self) -> int:
        return len(self.labels)

    def _sequence(self, video_id: str, track_id: int, frame_id: int) -> np.ndarray:
        g = self._fidx.get((str(video_id), int(track_id)))
        T, Fd = self.sequence_length, FEATURE_DIM
        if g is None or len(g) == 0:
            return np.zeros((T, Fd), dtype=np.float32)
        upto = g[g["frame_id"] <= frame_id]
        if len(upto) == 0:
            upto = g.iloc[:1]
        window = upto.iloc[-T:]
        X = window[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if len(X) < T:                     # padding al frente con la fila mas vieja
            pad = np.repeat(X[:1], T - len(X), axis=0)
            X = np.concatenate([pad, X], axis=0)
        return X

    def _apply_augment(self, X: np.ndarray) -> np.ndarray:
        a = self.augment
        # jitter de keypoints (sobre coords normalizadas)
        jit = a.get("keypoint_jitter_px", 0.0) / 384.0
        if jit > 0:
            kp_cols = [i for i, c in enumerate(FEATURE_COLUMNS) if c.endswith(("_x", "_y"))]
            X[:, kp_cols] += np.random.normal(0, jit, size=(X.shape[0], len(kp_cols))).astype(np.float32)
        # dropout de landmarks (cero a x,y,v de un landmark)
        p = a.get("landmark_dropout_p", 0.0)
        if p > 0:
            for li, n in enumerate(LANDMARKS):
                if np.random.rand() < p:
                    base = li * 3
                    X[:, base:base + 3] = 0.0
        # ruido en bbox
        bn = a.get("bbox_noise_frac", 0.0)
        if bn > 0:
            for c in ("bbox_w", "bbox_h"):
                j = FEATURE_COLUMNS.index(c)
                X[:, j] *= (1.0 + np.random.normal(0, bn))
        return X

    def __getitem__(self, idx: int) -> dict:
        row = self.labels.iloc[idx]
        X = self._sequence(row["video_id"], int(row["track_id"]), int(row["frame_id"]))
        if self.augment is not None:
            X = self._apply_augment(X.copy())
        X = np.clip((X - self.feat_mean) / self.feat_std, -8.0, 8.0)

        theta = math.radians(float(row["orientation_angle_deg"]))
        if "orientation_bin_8" in row and not pd.isna(row["orientation_bin_8"]):
            bin_idx = int(row["orientation_bin_8"]) % self.num_bins
        else:
            from ..circular import angle_to_bin
            bin_idx = angle_to_bin(theta, self.num_bins)
        conf = float(row["confidence"]) if "confidence" in row and not pd.isna(row.get("confidence")) else 1.0
        if conf > 1.0:                     # por si viene como 'high/medium/low' mapeado fuera
            conf = 1.0

        return {
            "features": torch.from_numpy(X).float(),
            "theta_gt": torch.tensor(theta, dtype=torch.float32),
            "orientation_bin": torch.tensor(bin_idx, dtype=torch.long),
            "confidence": torch.tensor(conf, dtype=torch.float32),
            "metadata": {
                "video_id": str(row["video_id"]),
                "track_id": int(row["track_id"]),
                "frame_id": int(row["frame_id"]),
                "visibility": str(row.get("visibility", "")),
            },
        }

    def sample_weights(self) -> np.ndarray:
        """Pesos para WeightedRandomSampler que balancea por orientation_bin."""
        bins = []
        for _, row in self.labels.iterrows():
            if "orientation_bin_8" in row and not pd.isna(row["orientation_bin_8"]):
                bins.append(int(row["orientation_bin_8"]) % self.num_bins)
            else:
                from ..circular import angle_to_bin
                bins.append(angle_to_bin(math.radians(float(row["orientation_angle_deg"])),
                                         self.num_bins))
        bins = np.asarray(bins)
        counts = np.bincount(bins, minlength=self.num_bins).astype(np.float64)
        counts[counts == 0] = 1.0
        w = 1.0 / counts[bins]
        return w / w.sum()
