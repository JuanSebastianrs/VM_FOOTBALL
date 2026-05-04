"""
Jersey Number Recognition - Cloud Training Script

Trains two baselines:
  - processed: uses selected frames per tracklet
  - full: uses all frames per tracklet

Designed for Vertex AI Custom Jobs, following the RF-DETR training pattern.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import zipfile


def validate_imports() -> None:
    errors = []
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
    except Exception as e:
        errors.append(f"PyTorch/CUDA: {e}")

    try:
        import yaml
    except ImportError as e:
        errors.append(f"PyYAML: {e}")

    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        assert major < 2, f"NumPy {np.__version__} >= 2.0 will break PyTorch ABI"
    except Exception as e:
        errors.append(f"NumPy: {e}")

    try:
        import cv2
    except ImportError as e:
        errors.append(f"OpenCV: {e}")

    if errors:
        print(f"\n[FATAL] {len(errors)} import validation error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("[OK] All imports validated")


validate_imports()

import yaml
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu() -> None:
    if not torch.cuda.is_available():
        print("[FATAL] No GPU detected!")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)
    print(f"[GPU] {name} ({mem:.1f} GB, compute capability {cc[0]}.{cc[1]})")


class GCSSyncThread(threading.Thread):
    def __init__(self, local_dir: Path, gcs_target: str, interval: int = 300):
        super().__init__()
        self.local_dir = local_dir
        self.gcs_target = gcs_target
        self.interval = interval
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.stop_event.wait(self.interval)
            if not self.stop_event.is_set():
                self.sync()

    def sync(self) -> None:
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", f"{self.local_dir}/*", f"{self.gcs_target}/"],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:
            pass


def _download_tar(tar_path: str, local_tar: Path) -> bool:
    print(f"[DATASET] Downloading tar: {tar_path}")
    result = subprocess.run(["gcloud", "storage", "cp", tar_path, str(local_tar)])
    if result.returncode != 0:
        print("[DATASET] Tar download failed")
        return False
    return local_tar.exists()


def _extract_tar(local_tar: Path, dest_dir: Path) -> bool:
    print("[DATASET] Extracting tar...")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(dest_dir)
    return True


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"[DATASET] Extracting {zip_path.name}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _normalize_split_dir(jersey_root: Path, split: str) -> None:
    split_dir = jersey_root / split
    images_dir = split_dir / "images"
    if images_dir.exists():
        return

    candidates = sorted(split_dir.glob("**/images"))
    if not candidates:
        return

    source_images = candidates[0]
    source_root = source_images.parent

    for item in source_root.iterdir():
        target = split_dir / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))

    try:
        if source_root.exists() and not any(source_root.iterdir()):
            source_root.rmdir()
    except OSError:
        pass


def download_dataset(config_dataset: dict) -> Path:
    local_root = Path(config_dataset["local_root"])
    jersey_root = Path(config_dataset["jersey_root"])
    local_root.mkdir(parents=True, exist_ok=True)

    train_images = jersey_root / "train" / "images"
    test_images = jersey_root / "test" / "images"

    if train_images.exists() and any(train_images.iterdir()):
        print(f"[DATASET] Already present: {train_images}")
        return jersey_root

    tar_path = config_dataset.get("tar_path", "")
    local_tar = local_root / "sn_jersey_2023.tar.gz"

    if tar_path:
        if not _download_tar(tar_path, local_tar):
            print("[FATAL] Could not download dataset tar")
            sys.exit(1)
        _extract_tar(local_tar, local_root)
        local_tar.unlink(missing_ok=True)

    if not jersey_root.exists():
        print(f"[FATAL] jersey root not found: {jersey_root}")
        sys.exit(1)

    for split in ["train", "test", "challenge"]:
        split_dir = jersey_root / split
        split_images = split_dir / "images"
        if split_images.exists() and any(split_images.iterdir()):
            continue
        zip_path = jersey_root / f"{split}.zip"
        if not zip_path.exists():
            print(f"[WARN] Missing zip: {zip_path}")
            continue
        _extract_zip(zip_path, split_dir)
        _normalize_split_dir(jersey_root, split)

    train_images = jersey_root / "train" / "images"
    test_images = jersey_root / "test" / "images"
    if not train_images.exists() or not test_images.exists():
        print("[FATAL] Dataset extraction failed")
        sys.exit(1)

    return jersey_root


def download_processed(config_dataset: dict) -> bool:
    processed_root = Path(config_dataset["processed_root"])
    summary_path = processed_root / "summary.json"
    if summary_path.exists():
        print(f"[PROCESSED] Already present: {summary_path}")
        return True

    processed_tar = config_dataset.get("processed_tar_path", "")
    if not processed_tar:
        print("[PROCESSED] No processed tar configured")
        return False

    local_root = Path(config_dataset["local_root"])
    local_tar = local_root / "processed.tar.gz"

    if not _download_tar(processed_tar, local_tar):
        print("[PROCESSED] Tar download failed")
        return False

    _extract_tar(local_tar, local_root)
    local_tar.unlink(missing_ok=True)

    if summary_path.exists():
        print(f"[PROCESSED] Extracted: {summary_path}")
        return True

    print("[PROCESSED] Summary not found after extract")
    return False


def download_latest_checkpoint(gcs_target: str, local_dir: Path) -> Optional[Path]:
    ckpt = f"{gcs_target}/checkpoint_last.pt"
    result = subprocess.run(["gcloud", "storage", "ls", ckpt], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    local_path = local_dir / "checkpoint_last.pt"
    dl = subprocess.run(["gcloud", "storage", "cp", ckpt, str(local_path)])
    if dl.returncode != 0 or not local_path.exists():
        return None
    return local_path


def train_experiment(exp_name: str, exp_cfg: dict, dataset_cfg: dict, output_dir: Path) -> dict:
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    gcs_output = os.environ.get("GCS_OUTPUT_PATH", "")
    sync_thread = None
    resume_path = None

    if gcs_output:
        gcs_target = f"{gcs_output}/jersey/{exp_name}"
        resume_path = download_latest_checkpoint(gcs_target, exp_dir)
        if resume_path:
            print(f"[RESUME] Downloaded {resume_path.name}")
        sync_thread = GCSSyncThread(exp_dir, gcs_target, interval=300)
        sync_thread.start()
        print(f"[SYNC] Background sync to {gcs_target}")

    args = [
        "python3",
        "train_sn_jersey.py",
        "--mode",
        exp_cfg["mode"],
        "--dataset-root",
        str(dataset_cfg["jersey_root"]),
        "--processed-root",
        str(dataset_cfg["processed_root"]),
        "--run-name",
        exp_name,
        "--output",
        str(output_dir),
        "--epochs",
        str(exp_cfg["epochs"]),
        "--batch-size",
        str(exp_cfg["batch_size"]),
        "--lr",
        str(exp_cfg["lr"]),
        "--weight-decay",
        str(exp_cfg.get("weight_decay", 1e-4)),
        "--img-size",
        str(exp_cfg.get("img_size", 96)),
        "--model",
        exp_cfg.get("model", "resnet18"),
    ]

    if exp_cfg.get("use_test_as_val", False):
        args.append("--use-test-as-val")
    else:
        args.extend(["--val-split", str(exp_cfg.get("val_split", 0.1))])

    if exp_cfg.get("skip_no_number", True):
        args.append("--skip-no-number")

    if resume_path:
        args.extend(["--resume", str(resume_path)])

    print(f"[TRAIN] Running: {' '.join(args)}")
    result = subprocess.run(args)

    if sync_thread:
        sync_thread.stop_event.set()
        sync_thread.join(timeout=10)
        sync_thread.sync()

    if result.returncode != 0:
        return {"status": "error", "error": f"train_sn_jersey exited {result.returncode}"}

    best_path = exp_dir / "best.pt"
    if best_path.exists():
        size_mb = best_path.stat().st_size / 1e6
        print(f"[OK] Best weights: {best_path.name} ({size_mb:.1f} MB)")

    return {"status": "success"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Jersey Number - Cloud Training")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoketest", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    check_gpu()

    if args.dry_run:
        print("[DRY RUN] Config validation only")
        for name, cfg in config.get("jersey", {}).get("experiments", {}).items():
            print(f"  - {name}: {cfg.get('mode')} epochs={cfg.get('epochs')}")
        return

    if args.smoketest:
        print("[SMOKE] Overriding experiment settings for quick validation")
        experiments_cfg = config.get("jersey", {}).get("experiments", {})
        for exp_name, exp_cfg in experiments_cfg.items():
            exp_cfg["epochs"] = min(int(exp_cfg.get("epochs", 2)), 2)
            exp_cfg["batch_size"] = min(int(exp_cfg.get("batch_size", 32)), 32)
            exp_cfg["use_test_as_val"] = True
            exp_cfg["val_split"] = 0.0
            print(
                f"[SMOKE] {exp_name}: mode={exp_cfg.get('mode')} "
                f"epochs={exp_cfg['epochs']} batch_size={exp_cfg['batch_size']} use_test_as_val=True"
            )

    dataset_cfg = config["dataset"]
    output_dir = Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" JERSEY NUMBER - CLOUD TRAINING")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Time: {datetime.now().isoformat()}")

    print("\n" + "=" * 70)
    print(" STEP 1: DOWNLOAD DATASET")
    print("=" * 70)
    dataset_root = download_dataset(dataset_cfg)
    dataset_cfg["jersey_root"] = str(dataset_root)

    print("\n" + "=" * 70)
    print(" STEP 2: PROCESSED DATA (OPTIONAL)")
    print("=" * 70)
    experiments = config.get("jersey", {}).get("experiments", {})
    if args.experiment:
        if args.experiment not in experiments:
            print(f"[ERROR] Experiment '{args.experiment}' not found")
            sys.exit(1)
        experiments = {args.experiment: experiments[args.experiment]}

    needs_processed = any(exp_cfg.get("mode") == "processed" for exp_cfg in experiments.values())
    if needs_processed:
        if not download_processed(dataset_cfg):
            print("[FATAL] Processed dataset required but not available")
            sys.exit(1)
    else:
        print("[PROCESSED] Skipped (no processed experiment selected)")

    print("\n" + "=" * 70)
    print(" STEP 3: TRAIN")
    print("=" * 70)

    results = {}
    for exp_name, exp_cfg in experiments.items():
        results[exp_name] = train_experiment(exp_name, exp_cfg, dataset_cfg, output_dir)

    print("\n" + "=" * 70)
    print(" STEP 4: UPLOAD RESULTS")
    print("=" * 70)

    gcs_output = os.environ.get("GCS_OUTPUT_PATH", config["output"].get("gcs_path", ""))
    if gcs_output:
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{output_dir}/*", f"{gcs_output}/"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print(f"[OK] Results uploaded to {gcs_output}")
        else:
            print(f"[ERROR] Upload failed: {result.stderr}")

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSummary saved to:", summary_path)


if __name__ == "__main__":
    main()
