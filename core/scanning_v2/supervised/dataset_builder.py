# core/scanning_v2/supervised/dataset_builder.py
"""
Construye el dataset supervisado (features + labels) desde los outputs V2 de uno
o varios videos, mas la anotacion humana.

Un sample = un `event_id`. Las etiquetas humanas vienen de
`data/annotations/scanning_windows_gt.csv`; si `scan_label_gt` esta vacio, el
sample queda UNLABELED (no se usa para entrenar y se reporta). NO se inventa GT.

Salidas:
  features.parquet, labels.parquet, dataset_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .feature_extractor import FeatureExtractor
from .schema import FEATURE_VERSION, LABEL_COLUMNS, LABEL_COLUMN


def _read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _norm_label(v):
    """scan_label_gt normalizado a 0/1 o None (vacio/no valido)."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        if s in ("0", "0.0"):
            return 0
        if s in ("1", "1.0"):
            return 1
    except (TypeError, ValueError):
        pass
    return None


class DatasetBuilder:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        ds = dict(c.get("dataset", {}))
        self.feature_version = str(ds.get("feature_version", FEATURE_VERSION))
        self.include_vision_map = bool(ds.get("include_vision_map", True))
        self.include_heuristic_pred = bool(ds.get("include_heuristic_pred_as_feature", False))
        self.drop_low_visibility = bool(ds.get("drop_low_visibility", False))
        self.min_valid_pose_ratio = float(ds.get("min_valid_pose_ratio_for_training", 0.0))
        self.warnings: list = []
        self._excl_vis = 0
        self._excl_pose = 0
        self._dup_keys: list = []
        self._stale_keys: list = []
        self._missing_keys: list = []
        self.fe = FeatureExtractor({
            "feature_version": self.feature_version,
            "include_vision_map": self.include_vision_map,
            "fps": c.get("video", {}).get("fps", 25.0),
            "include_sequence_features": ds.get("include_sequence_features", False),
            "sequence_length": ds.get("sequence_length", 32),
            "sequence_seconds": ds.get("sequence_seconds", 3.0),
        })

    # ------------------------------------------------------------------
    def build(self, video_ids: List[str], outputs_root: str,
              annotations_path: Optional[str]) -> dict:
        root = Path(outputs_root)
        ann = self._load_annotations(annotations_path)
        # reset de contadores/keys por construccion
        self.warnings, self._excl_vis, self._excl_pose = [], 0, 0
        self._dup_keys, self._stale_keys, self._missing_keys = [], [], []

        feats: List[pd.DataFrame] = []
        labels: List[pd.DataFrame] = []
        per_video = {}
        for vid in video_ids:
            vdir = root / vid
            events = _read(vdir / "pass_reception_events.parquet")
            if events is None or not len(events):
                per_video[vid] = {"events": 0, "note": "sin pass_reception_events"}
                continue
            head_pose = _read(vdir / "head_pose.parquet")
            scanning = _read(vdir / "scanning_events.parquet")
            vision = _read(vdir / "vision_map_metrics.csv")
            if head_pose is None:
                head_pose = pd.DataFrame(columns=["event_id", "frame_id"])
            fdf = self.fe.extract(events, head_pose, scanning, vision)
            feats.append(fdf)
            ldf = self._labels_for(fdf, ann, vid)
            labels.append(ldf)
            # claves (video_id, event_id) desincronizadas para el manifest
            cur = set(fdf["event_id"].astype(str))
            ann_ids = (set(ann[ann.get("video_id").astype(str) == str(vid)]["event_id"].astype(str))
                       if ann is not None and "video_id" in ann.columns
                       else (set(ann["event_id"].astype(str)) if ann is not None else set()))
            self._stale_keys += [f"{vid}/{e}" for e in sorted(ann_ids - cur)]
            self._missing_keys += [f"{vid}/{e}" for e in sorted(cur - ann_ids)]
            per_video[vid] = {
                "events": int(len(fdf)),
                "labeled": int(ldf[LABEL_COLUMN].notna().sum()),
                "trainable": int(ldf["training_eligible"].sum()),
            }

        features = (pd.concat(feats, ignore_index=True) if feats
                    else pd.DataFrame())
        label_df = (pd.concat(labels, ignore_index=True) if labels
                    else pd.DataFrame(columns=LABEL_COLUMNS))

        manifest = self._manifest(features, label_df, video_ids, per_video,
                                  annotations_path)
        return {"features": features, "labels": label_df, "manifest": manifest}

    # ------------------------------------------------------------------
    def _load_annotations(self, path) -> Optional[pd.DataFrame]:
        if not path or not Path(path).exists():
            return None
        df = pd.read_csv(path)
        if "event_id" not in df.columns:
            return None
        return df

    def _labels_for(self, fdf: pd.DataFrame, ann: Optional[pd.DataFrame],
                    video_id: str) -> pd.DataFrame:
        out = pd.DataFrame({"event_id": fdf["event_id"].astype(str),
                            "video_id": video_id})
        for c in LABEL_COLUMNS:
            if c in ("event_id", "video_id"):
                continue
            out[c] = None
        if ann is not None and "event_id" in ann.columns:
            sub = ann.copy()
            sub["event_id"] = sub["event_id"].astype(str)
            # match por (video_id, event_id) — no solo event_id
            if "video_id" in sub.columns:
                sub = sub[sub["video_id"].astype(str) == str(video_id)]
            sub = sub[sub["event_id"].isin(out["event_id"])]
            dups = sub["event_id"][sub["event_id"].duplicated(keep=False)].unique().tolist()
            for e in sorted(dups):
                self._dup_keys.append(f"{video_id}/{e}")
                labs = set(_norm_label(v) for v in sub[sub["event_id"] == e][LABEL_COLUMN])
                labs.discard(None)
                if len(labs) > 1:
                    # duplicado CONFLICTIVO -> error claro (no resolver en silencio)
                    raise ValueError(
                        f"Anotacion duplicada CONFLICTIVA para ({video_id}, {e}): "
                        f"scan_label_gt={sorted(labs)}. Corrige el GT antes de construir.")
            if len(dups):
                self.warnings.append(
                    f"{video_id}: duplicados no-conflictivos {sorted(dups)} "
                    "(labels identicos; se conserva el ultimo)")
            sub = sub.drop_duplicates("event_id", keep="last").set_index("event_id")
            for c in LABEL_COLUMNS:
                if c in ("event_id", "video_id") or c not in sub.columns:
                    continue
                out[c] = out["event_id"].map(sub[c])
        # scan_label_gt -> numerico (vacio/no-numerico = unlabeled/NaN)
        out[LABEL_COLUMN] = pd.to_numeric(out[LABEL_COLUMN], errors="coerce")

        # ---- elegibilidad para entrenamiento (gates de config) ----
        labeled = out[LABEL_COLUMN].isin([0, 1])
        eligible = labeled.copy()
        if self.drop_low_visibility:
            is_low = out["visibility"].astype(str).str.lower() == "low"
            self._excl_vis += int((labeled & is_low).sum())
            eligible &= ~is_low
        if self.min_valid_pose_ratio > 0 and "valid_pose_ratio" in fdf.columns:
            vpr = fdf.set_index(fdf["event_id"].astype(str))["valid_pose_ratio"]
            out_vpr = pd.to_numeric(out["event_id"].map(vpr), errors="coerce").fillna(0.0)
            low_pose = out_vpr < self.min_valid_pose_ratio
            self._excl_pose += int((eligible & low_pose).sum())
            eligible &= ~low_pose
        out["training_eligible"] = eligible.fillna(False).astype(bool)
        return out[LABEL_COLUMNS + ["training_eligible"]]

    def _manifest(self, features, labels, video_ids, per_video, ann_path) -> dict:
        import datetime as _dt
        n = int(len(features))
        if len(labels):
            lab = pd.to_numeric(labels[LABEL_COLUMN], errors="coerce")
            elig = labels["training_eligible"] if "training_eligible" in labels else pd.Series([], dtype=bool)
            n_lab = int(lab.notna().sum())
            n_pos = int((lab == 1).sum())
            n_neg = int((lab == 0).sum())
            n_train = int(elig.sum()) if len(elig) else 0
            n_train_pos = int(((lab == 1) & elig).sum()) if len(elig) else 0
            n_train_neg = int(((lab == 0) & elig).sum()) if len(elig) else 0
        else:
            n_lab = n_pos = n_neg = n_train = n_train_pos = n_train_neg = 0
        return {
            "feature_version": self.feature_version,
            "key_columns": ["video_id", "event_id"],
            "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "video_ids": list(video_ids),
            "annotations_path": str(ann_path) if ann_path else None,
            "n_samples": n,
            "n_labeled": n_lab,
            "n_unlabeled": int(n - n_lab),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_trainable": n_train,
            "n_trainable_positive": n_train_pos,
            "n_trainable_negative": n_train_neg,
            "n_excluded_low_visibility": int(self._excl_vis),
            "n_excluded_low_pose_ratio": int(self._excl_pose),
            "duplicated_annotation_keys": list(self._dup_keys),
            "stale_annotation_keys": list(self._stale_keys),
            "missing_annotation_keys": list(self._missing_keys),
            "gates": {
                "drop_low_visibility": self.drop_low_visibility,
                "min_valid_pose_ratio_for_training": self.min_valid_pose_ratio,
                "include_heuristic_pred_as_feature": self.include_heuristic_pred,
            },
            "warnings": list(self.warnings),
            "per_video": per_video,
            "n_feature_columns": int(features.shape[1]) if n else 0,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def write(result: dict, out_dir: str) -> dict:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        fp = d / "features.parquet"
        lp = d / "labels.parquet"
        result["features"].to_parquet(fp, index=False)
        result["labels"].to_parquet(lp, index=False)
        (d / "dataset_manifest.json").write_text(
            json.dumps(result["manifest"], indent=2), encoding="utf-8")
        return {"features": str(fp), "labels": str(lp),
                "manifest": str(d / "dataset_manifest.json")}
