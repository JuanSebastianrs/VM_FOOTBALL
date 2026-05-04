"""
Generate metric summaries and plots for jersey models.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


MODEL_SOURCES = {
    "full": "gs://vm-football-data/models/jersey_number/jersey/jersey_full/",
    "processed": "gs://vm-football-data/models/jersey_number/jersey/jersey_processed/",
    "preprocessed_full": "gs://vm-football-data/models/jersey_number/jersey/jersey_full_preprocessed_v2/",
    "preprocessed_processed": "gs://vm-football-data/models/jersey_number/jersey/jersey_processed_preprocessed_v2/",
}


@dataclass
class ModelMetrics:
    name: str
    summary: Dict[str, float]
    history: Optional[pd.DataFrame] = None


def run_cmd(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def download_gcs_file(src: str, dst: Path) -> bool:
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        f"& 'C:\\Users\\alejo\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gcloud.ps1' storage cp '{src}' '{dst}'",
    ])
    return dst.exists()


def load_metrics(local_root: Path, model: str, gcs_root: str) -> ModelMetrics:
    summary_path = local_root / model / "summary.json"
    history_path = local_root / model / "history.csv"

    download_gcs_file(f"{gcs_root}summary.json", summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    history = None
    try:
        if download_gcs_file(f"{gcs_root}history.csv", history_path):
            history = pd.read_csv(history_path)
    except RuntimeError:
        history = None

    return ModelMetrics(name=model, summary=summary, history=history)


def plot_bar(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["model"], df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_losses(histories: Dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, hist in histories.items():
        ax.plot(hist["epoch"], hist["train_loss"], label=f"{name} train")
        ax.plot(hist["epoch"], hist["val_loss"], linestyle="--", label=f"{name} val")
    ax.set_title("Train vs Val Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metrics plots for jersey models")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/metrics_cache"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[ModelMetrics] = []
    for name, gcs_root in MODEL_SOURCES.items():
        metrics.append(load_metrics(args.cache_dir, name, gcs_root))

    rows = []
    histories: Dict[str, pd.DataFrame] = {}
    for m in metrics:
        summary = m.summary
        row = {
            "model": m.name,
            "best_val_top1": float(summary.get("best_val_top1", 0.0)),
            "best_val_top3": float(summary.get("best_val_top3", 0.0)) if "best_val_top3" in summary else None,
            "best_epoch": int(summary.get("best_epoch", 0)) if "best_epoch" in summary else None,
            "stopped_epoch": int(summary.get("stopped_epoch", 0)) if "stopped_epoch" in summary else None,
        }

        if m.history is not None:
            history = m.history.copy()
            histories[m.name] = history
            best_epoch = row["best_epoch"] or int(history["epoch"].iloc[history["val_top1"].idxmax()])
            best_row = history[history["epoch"] == best_epoch].iloc[0]
            row["train_top1_at_best"] = float(best_row["train_top1"])
            row["val_top1_at_best"] = float(best_row["val_top1"])
            row["generalization_gap"] = row["train_top1_at_best"] - row["val_top1_at_best"]
            row["min_val_loss"] = float(history["val_loss"].min())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "jersey_models_metrics.csv", index=False)
    (args.out_dir / "jersey_models_metrics.json").write_text(
        df.to_json(orient="records", indent=2), encoding="utf-8"
    )

    plot_bar(df, "best_val_top1", args.out_dir / "best_val_top1.png", "Best Val Top-1")
    if df["best_val_top3"].notnull().any():
        plot_bar(df.fillna(0.0), "best_val_top3", args.out_dir / "best_val_top3.png", "Best Val Top-3")
    if histories:
        plot_losses(histories, args.out_dir / "loss_curves.png")
    if "generalization_gap" in df:
        plot_bar(df.fillna(0.0), "generalization_gap", args.out_dir / "generalization_gap.png", "Generalization Gap")


if __name__ == "__main__":
    main()
