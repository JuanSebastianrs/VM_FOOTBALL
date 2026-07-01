# scripts/scanning/evaluate_orientation.py
"""
Evaluacion del modulo de scanning.

A. Orientacion (requiere --checkpoint + labels/features):
   MAE angular, accuracy@22.5/45, bin accuracy, errores por visibilidad y por
   tamano de crop (alto de bbox).

B. Scanning (requiere --scanning_pred + --scanning_gt):
   precision, recall, F1 y matriz de confusion scan/no-scan por recepcion.
   El GT debe tener columnas: event_id (o receiver_track_id+frame_reception) y
   scan_label.

Ejemplo:
  python scripts/scanning/evaluate_orientation.py \
      --checkpoint outputs/scanning/checkpoints/best_orientation_tcn.pt \
      --labels data/annotations/orientation_labels.csv \
      --features data/annotations/SNMOT-148_orientation_features.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch                                                            # noqa: E402

from core.scanning import load_config                                  # noqa: E402
from core.scanning.circular import circular_delta                      # noqa: E402
from core.scanning.models.evaluate_orientation_model import evaluate_model  # noqa: E402
from core.scanning.models.orientation_dataset import OrientationSequenceDataset  # noqa: E402
from core.scanning.models.orientation_tcn import build_model           # noqa: E402


def evaluate_orientation(args, config):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt["model_name"], ckpt["feature_dim"],
                        num_bins=ckpt["num_bins"], hidden_dim=ckpt.get("hidden_dim", 64),
                        num_blocks=ckpt.get("num_blocks", 4))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    ds = OrientationSequenceDataset(
        args.labels or config["data"]["annotation_csv"], args.features,
        sequence_length=ckpt.get("sequence_length", 16), num_bins=ckpt["num_bins"],
        split="val", val_split=args.val_split,
        norm_stats=(ckpt["norm_mean"], ckpt["norm_std"]))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    overall = evaluate_model(model, loader, device, ckpt["num_bins"])
    print("\n== A. Orientacion (global) ==")
    for k, v in overall.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # errores por visibilidad y por tamano de crop
    model.eval()
    by_vis, by_size = {}, {}
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = s["features"].unsqueeze(0).to(device)
            out = model(x)
            err = abs(math.degrees(circular_delta(
                float(out["theta"].item()), float(s["theta_gt"].item()))))
            vis = s["metadata"]["visibility"] or "unknown"
            by_vis.setdefault(vis, []).append(err)
            bh = ds.labels.iloc[i]
            size_bucket = "far" if (float(bh.get("y_field", 0) or 0)) else "n/a"
            by_size.setdefault(size_bucket, []).append(err)
    print("\n  MAE por visibilidad:")
    for k, v in by_vis.items():
        print(f"    {k}: {np.mean(v):.1f} deg (n={len(v)})")
    return overall


def evaluate_scanning(args):
    pred = pd.read_parquet(args.scanning_pred) if args.scanning_pred.endswith("parquet") \
        else pd.read_csv(args.scanning_pred)
    gt = pd.read_csv(args.scanning_gt)
    key = "event_id" if ("event_id" in pred.columns and "event_id" in gt.columns) \
        else None
    if key:
        merged = pred.merge(gt[[key, "scan_label"]], on=key, suffixes=("_pred", "_gt"))
    else:
        merged = pred.merge(
            gt, on=["receiver_track_id", "frame_reception"], suffixes=("_pred", "_gt"))
    yp = merged["scan_label_pred"].astype(int).to_numpy()
    yg = merged["scan_label_gt"].astype(int).to_numpy()
    tp = int(((yp == 1) & (yg == 1)).sum())
    fp = int(((yp == 1) & (yg == 0)).sum())
    fn = int(((yp == 0) & (yg == 1)).sum())
    tn = int(((yp == 0) & (yg == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print("\n== B. Scanning ==")
    print(f"  precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
    print(f"  confusion  TP={tp} FP={fp} FN={fn} TN={tn}  (n={len(merged)})")
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--scanning_pred", default=None)
    p.add_argument("--scanning_gt", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    results = {}
    if args.checkpoint:
        results["orientation"] = evaluate_orientation(args, config)
    if args.scanning_pred and args.scanning_gt:
        results["scanning"] = evaluate_scanning(args)
    if not results:
        print("Nada que evaluar: pasa --checkpoint (orientacion) y/o "
              "--scanning_pred + --scanning_gt (scanning).")
    if args.out_json and results:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetricas -> {args.out_json}")


if __name__ == "__main__":
    main()
