# core/scanning_v2/supervised/predictor.py
"""
Prediccion con el clasificador supervisado entrenado.

Si NO existe el modelo, falla con un error claro (no silenciosamente): la decision
de caer a la heuristica V2 la toma el llamador, no se enmascara aqui.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema import PREDICTION_COLUMNS, FEATURE_VERSION


class ScanningPredictor:
    def __init__(self, model, feature_columns, model_version=None,
                 feature_version=FEATURE_VERSION,
                 allow_feature_version_mismatch=False):
        self.model = model
        self.feature_columns = list(feature_columns)
        self.model_version = model_version
        self.feature_version = feature_version
        self.allow_feature_version_mismatch = bool(allow_feature_version_mismatch)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_path: str,
             allow_feature_version_mismatch: bool = False) -> "ScanningPredictor":
        import joblib
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(
                f"No existe el modelo entrenado en '{model_path}'. Entrena primero "
                "con train_scanning_model.py, o usa la heuristica V2 como fallback.")
        model = joblib.load(mp)
        meta_path = mp.parent / "model_metadata.json"
        feat_cols, mver, fver = None, None, FEATURE_VERSION
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feat_cols = meta.get("feature_columns")
            mver = meta.get("repo_version") or meta.get("trained_at")
            fver = meta.get("feature_version", FEATURE_VERSION)
        if not feat_cols:
            raise ValueError(
                f"No se pudo leer feature_columns de {meta_path}; modelo no usable.")
        return cls(model, feat_cols, mver, fver,
                   allow_feature_version_mismatch=allow_feature_version_mismatch)

    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        self._check_compatibility(features)
        X = features[self.feature_columns]    # solo columnas esperadas (extras OK)
        proba = self._proba(X)
        label = (proba >= 0.5).astype(int)
        conf = np.maximum(proba, 1.0 - proba)
        out = pd.DataFrame({
            "event_id": features["event_id"].values,
            "video_id": features["video_id"].values if "video_id" in features else None,
            "receiver_track_id": (features["receiver_track_id"].values
                                  if "receiver_track_id" in features else None),
            "scan_label_model": label,
            "scan_probability_model": np.round(proba, 4),
            "model_confidence": np.round(conf, 4),
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "prediction_source": "model",
        })
        return out[PREDICTION_COLUMNS]

    def _check_compatibility(self, features: pd.DataFrame):
        # 1) feature_version del dataset vs del modelo
        if "feature_version" in features.columns:
            vers = set(str(v) for v in features["feature_version"].dropna().unique())
            mismatch = vers and vers != {str(self.feature_version)}
            if mismatch and not self.allow_feature_version_mismatch:
                raise ValueError(
                    f"feature_version del dataset {sorted(vers)} != del modelo "
                    f"'{self.feature_version}'. Re-construye features con la version "
                    "correcta o pasa allow_feature_version_mismatch=True.")
        # 2) todas las columnas esperadas deben existir (extras se permiten)
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"Faltan {len(missing)} feature_columns esperadas por el modelo: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
                "No se reindexa silenciosamente.")

    def _proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        # fallback: decision_function -> sigmoide
        d = self.model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))

    @staticmethod
    def write(df: pd.DataFrame, out_dir: str) -> str:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "scanning_model_predictions.parquet"
        df.to_parquet(p, index=False)
        return str(p)
