# scripts/scanning_v2/run_supervised_weak_e2e.py
"""
Orquestador E2E de la fase supervisada con weak supervision multi-video.

Pasos:
  1. Detecta videos con outputs V2 (`pass_reception_events.parquet`).
  2. Construye features SIN etiquetas (pasada 1).
  3. Genera pseudo-etiquetas debiles (reglas explicitas; archivo separado).
  4. Re-construye el dataset CON las etiquetas debiles (pasada 2).
  5. Entrena el zoo de arquitecturas (mismo split por video), elige best.
  6. Evalua el best contra el GT humano/visual (`scanning_windows_gt.csv`)
     si tiene etiquetas: metrica INDEPENDIENTE de las reglas debiles.

Ejemplo:
  python scripts/scanning_v2/run_supervised_weak_e2e.py \
      --config configs/scanning_v2_supervised_weak.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from core.scanning_v2.supervised import (DatasetBuilder, ScanningEvaluator,  # noqa: E402
                                         ScanningPredictor)
from core.scanning_v2.paths import discover_scanning_videos as discover_videos  # noqa: E402


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scanning_v2_supervised_weak.yaml")
    ap.add_argument("--human_gt", default="data/annotations/scanning_windows_gt.csv")
    ap.add_argument("--weak_csv", default="data/annotations/scanning_windows_weak_v1.csv")
    ap.add_argument("--work_dir", default=None,
                    help="default: outputs.root del config "
                         "(outputs/scanning_training)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    outputs_root = config.get("inputs", {}).get("outputs_root", "outputs")
    if args.work_dir is None:
        args.work_dir = config.get("outputs", {}).get(
            "root", "outputs/scanning_training")

    vids = discover_videos(outputs_root)
    print(f"[e2e] videos con eventos V2: {len(vids)} -> {vids}")
    if not vids:
        sys.exit("[e2e] no hay videos con pass_reception_events; corre run_scanning_v2 antes.")

    ds_dir = os.path.join(args.work_dir, "dataset")
    py = sys.executable

    # pasada 1: features sin etiquetas (para generar weak labels)
    run([py, "scripts/scanning_v2/build_scanning_dataset.py", "--config", args.config,
         "--video_ids", *vids, "--outputs_root", outputs_root,
         "--annotations", "NONE_does_not_exist.csv", "--output_dir", ds_dir])

    # pseudo-etiquetas debiles (excluye eventos con GT humano valido)
    run([py, "scripts/scanning_v2/generate_weak_labels.py",
         "--features", os.path.join(ds_dir, "features.parquet"),
         "--human_gt", args.human_gt, "--output", args.weak_csv])

    # pasada 2: dataset con etiquetas debiles
    run([py, "scripts/scanning_v2/build_scanning_dataset.py", "--config", args.config,
         "--video_ids", *vids, "--outputs_root", outputs_root,
         "--annotations", args.weak_csv, "--output_dir", ds_dir])

    # zoo de arquitecturas
    zoo = [py, "scripts/scanning_v2/train_model_zoo.py", "--config", args.config,
           "--features", os.path.join(ds_dir, "features.parquet"),
           "--labels", os.path.join(ds_dir, "labels.parquet"),
           "--output_dir", os.path.join(args.work_dir, "models")]
    if args.overwrite:
        zoo.append("--overwrite")
    run(zoo)

    # evaluacion INDEPENDIENTE contra GT humano/visual (si hay etiquetas)
    gt = pd.read_csv(args.human_gt) if os.path.exists(args.human_gt) else None
    if gt is not None:
        lab = pd.to_numeric(gt.get("scan_label_gt"), errors="coerce")
        gt_ok = gt[lab.isin([0, 1])].copy()
        if len(gt_ok):
            print(f"[e2e] evaluando best contra GT independiente "
                  f"({len(gt_ok)} eventos etiquetados)...")
            features = pd.read_parquet(os.path.join(ds_dir, "features.parquet"))
            predictor = ScanningPredictor.load(
                os.path.join(args.work_dir, "models", "best",
                             "scanning_classifier.pkl"))
            preds = predictor.predict(features)
            heur = features[["event_id", "video_id",
                             "heuristic_scan_label_pred"]].rename(
                columns={"heuristic_scan_label_pred": "scan_label_pred"})
            result = ScanningEvaluator(config).evaluate(preds, gt_ok, heuristic=heur,
                                                        features=features)
            out = ScanningEvaluator.write(result,
                                          os.path.join(args.work_dir, "reports_gt"))
            print(f"[e2e] reporte GT -> {out['report']}")
        else:
            print("[e2e] GT humano sin etiquetas 0/1: se omite evaluacion independiente.")
    print("[e2e] DONE")


if __name__ == "__main__":
    main()
