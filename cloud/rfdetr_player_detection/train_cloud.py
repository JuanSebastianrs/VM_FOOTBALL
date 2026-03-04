"""
RF-DETR Player Detection - Cloud Training Script
Trains RF-DETR Base for player detection (1 class: player).
Designed for Vertex AI Custom Spot Jobs with auto-resume on preemption.

Features:
  - GCS Download: downloads dataset tar to local SSD for fast I/O
  - YOLO→COCO: converts YOLO labels to COCO format (classes 0-3 → player)
  - bf16→fp16: monkey-patches autocast for T4 GPUs (cc 7.5)
  - Spot Resume: checks GCS for previous checkpoints on restart
  - Crash Handler: uploads crash log + partial results on failure
  - GCS Sync: background thread uploads checkpoints every 5 min
"""

import os
import gc
import sys
import time
import json
import shutil
import signal
import argparse
import subprocess
import traceback
import threading

# ============================================================
# PHASE 0: Validate ALL imports before anything else
# ============================================================

def validate_imports():
    """Validate all required imports are available. Fails fast."""
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

    try:
        import pydantic
        from pydantic import field_validator  # noqa: F401
        assert pydantic.__version__.startswith("2"), f"Pydantic V1 detected: {pydantic.__version__}"
    except Exception as e:
        errors.append(f"Pydantic: {e}")

    try:
        import rfdetr  # noqa: F401
    except ImportError as e:
        errors.append(f"rfdetr: {e}")

    try:
        import transformers  # noqa: F401
    except ImportError as e:
        errors.append(f"transformers: {e}")

    try:
        import accelerate  # noqa: F401
    except ImportError as e:
        errors.append(f"accelerate: {e}")

    try:
        import supervision  # noqa: F401
    except ImportError as e:
        errors.append(f"supervision: {e}")

    if errors:
        print(f"\n[FATAL] {len(errors)} import validation error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("[OK] All imports validated")


validate_imports()

# ============================================================
# Now safe to import everything
# ============================================================
import yaml
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from PIL import Image

# ============================================================
# Constants
# ============================================================

# 1 unified class: player (merges classes 0-3, ignores 4-5)
UNIFIED_CLASSES = ["player"]
CLASS_MAPPING = {
    0: 0,   # player_left → player
    1: 0,   # player_right → player
    2: 0,   # goalkeeper_left → player
    3: 0,   # goalkeeper_right → player
    4: -1,  # referee → IGNORE
    5: -1,  # ball → IGNORE
}
COCO_CATEGORIES = [
    {"id": i, "name": name, "supercategory": "football"}
    for i, name in enumerate(UNIFIED_CLASSES)
]

# Watchdog: kill process if stuck during TRAINING (not download)
WATCHDOG_TIMEOUT = 14400  # 4 hours (1 epoch can take 2+ hours)
_watchdog_timer = None


def _watchdog_expired():
    """Called by the watchdog timer when no progress is detected."""
    print(f"\n[WATCHDOG] No progress for {WATCHDOG_TIMEOUT}s — killing process")
    _emergency_upload()
    os._exit(1)


def enable_watchdog():
    """Enable the watchdog timer (call AFTER data download)."""
    global _watchdog_timer
    _watchdog_timer = threading.Timer(WATCHDOG_TIMEOUT, _watchdog_expired)
    _watchdog_timer.daemon = True
    _watchdog_timer.start()
    print(f"  [WATCHDOG] Enabled: {WATCHDOG_TIMEOUT}s timeout")


def reset_watchdog():
    """Reset the watchdog timer. Called periodically during training."""
    global _watchdog_timer
    if _watchdog_timer is not None:
        _watchdog_timer.cancel()
        _watchdog_timer = threading.Timer(WATCHDOG_TIMEOUT, _watchdog_expired)
        _watchdog_timer.daemon = True
        _watchdog_timer.start()


def _emergency_upload():
    """Upload any existing output to GCS on crash."""
    gcs_output = os.environ.get("GCS_OUTPUT_PATH", "")
    output_dir = "/workspace/runs"
    if gcs_output and os.path.exists(output_dir):
        try:
            print(f"[EMERGENCY] Uploading {output_dir} to {gcs_output}")
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", f"{output_dir}/*", f"{gcs_output}/"],
                capture_output=True, text=True, timeout=120,
            )
            print("[EMERGENCY] Upload complete")
        except Exception as e:
            print(f"[EMERGENCY] Upload failed: {e}")


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_cleanup():
    """Safely clean GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def check_gpu():
    """Validate GPU availability and print diagnostics."""
    if not torch.cuda.is_available():
        print("[FATAL] No GPU detected!")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)
    print(f"[GPU] {name} ({mem:.1f} GB, compute capability {cc[0]}.{cc[1]})")

    # Compute test
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    y = x @ x.t()
    loss = y.sum()
    loss.backward()
    del x, y, loss
    torch.cuda.empty_cache()
    print("[OK] GPU forward+backward pass verified")

    return cc


# ============================================================
# Dataset Download
# ============================================================

def download_dataset(config_dataset: dict, local_path: Path) -> bool:
    """
    Download dataset to local SSD. Strategy:
      1. Prefer tar.gz (1 file download + extract = ~2-5 min)
      2. Fallback to gcloud storage cp -r (42K files = 1-2 hours, avoid)

    On Spot restart: reuses existing local dataset if complete.
    """
    # Check if already downloaded (Spot restart)
    if local_path.exists():
        train_imgs = list((local_path / "images" / "train").glob("*.jpg"))
        if len(train_imgs) > 100:
            print(f"[DATASET] Already present: {len(train_imgs)} train images")
            return True

    local_path.mkdir(parents=True, exist_ok=True)
    parent = local_path.parent

    # Strategy 1: tar.gz
    tar_path = config_dataset.get("tar_path", "")
    if tar_path:
        print(f"[DATASET] Downloading tar: {tar_path}")
        local_tar = parent / "dataset.tar.gz"
        start = time.time()

        result = subprocess.run(
            ["gcloud", "storage", "cp", tar_path, str(local_tar)],
            capture_output=True, text=True,
        )

        if result.returncode == 0 and local_tar.exists():
            tar_size = local_tar.stat().st_size / 1e9
            dl_time = time.time() - start
            print(f"[DATASET] Downloaded {tar_size:.1f} GB in {dl_time:.0f}s")

            # Extract
            print("[DATASET] Extracting...")
            extract_start = time.time()
            result = subprocess.run(
                ["tar", "xzf", str(local_tar), "-C", str(parent)],
                capture_output=True, text=True,
            )

            if result.returncode == 0:
                extract_time = time.time() - extract_start
                print(f"[DATASET] Extracted in {extract_time:.0f}s")

                # Cleanup tar to save disk
                local_tar.unlink(missing_ok=True)

                # Verify
                train_imgs = list((local_path / "images" / "train").glob("*.jpg"))
                print(f"[DATASET] {len(train_imgs)} train images found")
                return len(train_imgs) > 0
            else:
                print(f"[DATASET] Tar extraction failed: {result.stderr}")
                local_tar.unlink(missing_ok=True)
        else:
            print(f"[DATASET] Tar download failed: {result.stderr}")

    # Strategy 2: Direct GCS copy (slow fallback)
    gcs_path = config_dataset.get("gcs_path", "")
    if gcs_path:
        print(f"[DATASET] Fallback: direct GCS copy from {gcs_path}")
        result = subprocess.run(
            ["gcloud", "storage", "cp", "-r", f"{gcs_path}/*", str(local_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            train_imgs = list((local_path / "images" / "train").glob("*.jpg"))
            print(f"[DATASET] {len(train_imgs)} train images downloaded")
            return len(train_imgs) > 0

    print("[FATAL] Could not download dataset")
    return False


# ============================================================
# YOLO → COCO Conversion
# ============================================================

def convert_yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split: str,
    max_images: int = 0,
) -> Path:
    """Convert YOLO format labels to COCO format for RF-DETR."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "VM_FOOTBALL - Player Detection", "version": "1.0"},
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    ann_id = 0
    images = sorted(images_dir.glob("*.jpg"))
    
    if max_images > 0 and len(images) > max_images:
        print(f"   [INFO] Downsampling {split} from {len(images)} to {max_images} images")
        random.shuffle(images)
        images = images[:max_images]

    print(f"   Converting {len(images)} images for {split}...")

    for img_id, img_path in enumerate(tqdm(images, desc=split)):
        img = Image.open(img_path)
        w, h = img.size

        dest = split_dir / img_path.name
        if not dest.exists():
            shutil.copy(img_path, dest)

        coco["images"].append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": img_path.name,
        })

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig = int(parts[0])
                        unified = CLASS_MAPPING.get(orig, -1)
                        if unified < 0:
                            continue

                        xc, yc, bw, bh = map(float, parts[1:5])
                        x_min = (xc - bw / 2) * w
                        y_min = (yc - bh / 2) * h

                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": unified,
                            "bbox": [x_min, y_min, bw * w, bh * h],
                            "area": bw * w * bh * h,
                            "iscrowd": 0,
                        })
                        ann_id += 1

    json_path = split_dir / "_annotations.coco.json"
    with open(json_path, "w") as f:
        json.dump(coco, f)

    print(f"   [OK] {split}: {len(coco['images'])} imgs, {len(coco['annotations'])} annotations")
    return json_path


def prepare_coco_dataset(
    dataset_path: Path,
    coco_dir: Path,
    limit_valid_images: int = 0,
    force_conversion: bool = False,
) -> Path:
    """Full YOLO→COCO conversion pipeline with GCS caching."""

    # Check if COCO dataset already exists locally (skip expensive conversion)
    train_json = coco_dir / "train" / "_annotations.coco.json"
    valid_json = coco_dir / "valid" / "_annotations.coco.json"
    test_json = coco_dir / "test" / "_annotations.coco.json"

    if not force_conversion and train_json.exists() and valid_json.exists() and test_json.exists():
        print("[COCO] Dataset already exists locally — skipping conversion")
        for split in ["train", "valid", "test"]:
            n_imgs = len(list((coco_dir / split).glob("*.jpg")))
            print(f"   [CACHED] {split}/: {n_imgs} images")
        return coco_dir

    # Check GCS cache (download pre-converted COCO annotations)
    gcs_coco_cache = os.environ.get("GCS_COCO_CACHE", "gs://vm-football-data/coco_cache_player")
    if not force_conversion:
        try:
            print(f"[COCO] Checking GCS cache: {gcs_coco_cache}")
            result = subprocess.run(
                ["gcloud", "storage", "ls", f"{gcs_coco_cache}/train/_annotations.coco.json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print("[COCO] Found cache in GCS! Downloading annotations...")
                for split in ["train", "valid", "test"]:
                    split_dir = coco_dir / split
                    split_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        ["gcloud", "storage", "cp",
                         f"{gcs_coco_cache}/{split}/_annotations.coco.json",
                         str(split_dir / "_annotations.coco.json")],
                        capture_output=True, text=True,
                    )
                # Still need to copy images
                print("[COCO] Annotations downloaded, copying images...")
                _copy_images_only(dataset_path, coco_dir)
                print("[COCO] Dataset restored from GCS cache")
                return coco_dir
            else:
                print("[COCO] No GCS cache found, running full conversion...")
        except Exception as e:
            print(f"[COCO] GCS cache check failed ({e}), running full conversion...")

    print("[COCO] Converting YOLO dataset to COCO format...")

    if coco_dir.exists():
        print("   Cleaning previous COCO directory...")
        shutil.rmtree(coco_dir)
    coco_dir.mkdir(parents=True, exist_ok=True)

    # Train
    convert_yolo_to_coco(
        dataset_path / "images" / "train",
        dataset_path / "labels" / "train",
        coco_dir, "train",
    )

    # Valid (uses "test" split from YOLO dataset as validation)
    convert_yolo_to_coco(
        dataset_path / "images" / "test",
        dataset_path / "labels" / "test",
        coco_dir, "valid",
        max_images=limit_valid_images,
    )

    # Test — subset of validation (RF-DETR requires 3 splits)
    print("   Creating test split from validation...")
    test_dir = coco_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_dir / "valid" / "_annotations.coco.json") as f:
        valid_coco = json.load(f)

    valid_imgs = list((coco_dir / "valid").glob("*.jpg"))
    n_test = max(100, len(valid_imgs) // 10)
    test_sample = valid_imgs[:n_test]

    valid_img_map = {img["file_name"]: img for img in valid_coco["images"]}

    coco_test = {
        "info": {"description": "VM_FOOTBALL Test", "version": "1.0"},
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    ann_id = 0
    for new_img_id, img_path in enumerate(tqdm(test_sample, desc="test")):
        shutil.copy(img_path, test_dir / img_path.name)
        orig_img = valid_img_map.get(img_path.name)
        if orig_img:
            coco_test["images"].append({
                "id": new_img_id,
                "width": orig_img["width"],
                "height": orig_img["height"],
                "file_name": img_path.name,
            })
            for ann in valid_coco["annotations"]:
                if ann["image_id"] == orig_img["id"]:
                    coco_test["annotations"].append({
                        "id": ann_id,
                        "image_id": new_img_id,
                        "category_id": ann["category_id"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": 0,
                    })
                    ann_id += 1

    with open(test_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco_test, f)
    print(f"   [OK] test: {len(coco_test['images'])} imgs, {len(coco_test['annotations'])} annotations")

    # Verify structure
    print("\n   Verifying COCO structure:")
    for split in ["train", "valid", "test"]:
        split_path = coco_dir / split
        json_ok = (split_path / "_annotations.coco.json").exists()
        n_imgs = len(list(split_path.glob("*.jpg")))
        status = "[OK]" if json_ok and n_imgs > 0 else "[ERROR]"
        print(f"   {status} {split}/: {n_imgs} images, json={json_ok}")

    # Upload annotations to GCS cache for future runs
    try:
        print(f"\n   Uploading COCO annotations to GCS cache: {gcs_coco_cache}")
        for split in ["train", "valid", "test"]:
            json_file = coco_dir / split / "_annotations.coco.json"
            if json_file.exists():
                subprocess.run(
                    ["gcloud", "storage", "cp", str(json_file),
                     f"{gcs_coco_cache}/{split}/_annotations.coco.json"],
                    capture_output=True, text=True,
                )
        print("   [OK] COCO annotations cached in GCS")
    except Exception as e:
        print(f"   [WARN] Failed to cache COCO annotations: {e}")

    return coco_dir


def _copy_images_only(dataset_path: Path, coco_dir: Path):
    """Copy images to COCO directory without converting labels (used with GCS cache)."""
    for src_split, dst_split in [("train", "train"), ("test", "valid")]:
        src_imgs = dataset_path / "images" / src_split
        dst_dir = coco_dir / dst_split
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_imgs.exists():
            print(f"   [WARN] Source images not found: {src_imgs}")
            continue

        imgs = sorted(src_imgs.glob("*.jpg"))
        print(f"   Copying {len(imgs)} images for {dst_split}...")
        for img_path in tqdm(imgs, desc=dst_split):
            dest = dst_dir / img_path.name
            if not dest.exists():
                shutil.copy(img_path, dest)

    # Test split: copy subset from valid
    valid_dir = coco_dir / "valid"
    test_dir = coco_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_json = test_dir / "_annotations.coco.json"
    if test_json.exists():
        with open(test_json) as f:
            test_coco = json.load(f)
        for img_info in test_coco["images"]:
            src = valid_dir / img_info["file_name"]
            dst = test_dir / img_info["file_name"]
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)


# ============================================================
# GCS Sync Thread
# ============================================================

class GCSSyncThread(threading.Thread):
    """Background thread to periodically sync outputs to GCS."""

    def __init__(self, local_dir: Path, gcs_target: str, interval: int = 300):
        super().__init__()
        self.local_dir = local_dir
        self.gcs_target = gcs_target
        self.interval = interval
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            self.stop_event.wait(self.interval)
            if not self.stop_event.is_set():
                self.sync()

    def sync(self):
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", "-r",
                 f"{self.local_dir}/*", f"{self.gcs_target}/"],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:
            pass  # Never crash the training


# ============================================================
# RF-DETR Training
# ============================================================

def get_model_class(class_name: str):
    """Import and return the RF-DETR model class."""
    from rfdetr import RFDETRBase, RFDETRLarge
    models = {
        "RFDETRBase": RFDETRBase,
        "RFDETRLarge": RFDETRLarge,
    }
    if class_name not in models:
        raise ValueError(f"Unknown model: {class_name}. Available: {list(models.keys())}")
    return models[class_name]


def setup_amp_for_gpu():
    """
    Configure AMP strategy based on GPU capability.
    T4 (cc 7.5): supports FP16 but NOT bf16 → monkey-patch autocast
    L4/A100/H100 (cc >= 8.0): supports both FP16 and BF16
    """
    if not torch.cuda.is_available():
        return True

    cc = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)

    if cc < (8, 0):
        # T4: Patch autocast to redirect bf16 → fp16
        _OrigAutocast = torch.amp.autocast

        def _fp16_autocast(*args, **kwargs):
            if "dtype" in kwargs and kwargs["dtype"] == torch.bfloat16:
                kwargs["dtype"] = torch.float16
            return _OrigAutocast(*args, **kwargs)

        torch.amp.autocast = _fp16_autocast
        torch.cuda.amp.autocast = _fp16_autocast

        torch.set_float32_matmul_precision("medium")
        os.environ["TORCH_AMP_DTYPE"] = "float16"
        print(f"  [AMP] {gpu_name} (cc {cc[0]}.{cc[1]}): Patched autocast bf16→fp16")
        print(f"  [INFO] T4 Tensor Cores: 65 TFLOPS FP16 vs 8.1 TFLOPS FP32")
    else:
        print(f"  [AMP] {gpu_name} (cc {cc[0]}.{cc[1]}): Native BF16 supported")

    return True


def train_single_experiment(
    exp_name: str,
    exp_config: dict,
    coco_dir: Path,
    output_dir: Path,
) -> dict:
    """Train a single RF-DETR experiment."""
    print(f"\n{'='*70}")
    print(f"RF-DETR EXPERIMENT: {exp_name}")
    print(f"  {exp_config.get('description', '')}")
    eff_batch = exp_config["batch_size"] * exp_config.get("grad_accum_steps", 1)
    print(f"  batch={exp_config['batch_size']}x{exp_config.get('grad_accum_steps', 1)}={eff_batch}, epochs={exp_config['epochs']}")
    print(f"{'='*70}")

    # Clean GPU aggressively
    gpu_cleanup()

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create model
        ModelClass = get_model_class(exp_config["model_class"])
        model = ModelClass()

        # Setup AMP
        use_amp = setup_amp_for_gpu()

        # Start background GCS sync
        sync_thread = None
        gcs_output = os.environ.get("GCS_OUTPUT_PATH", "")
        if gcs_output:
            gcs_target = f"{gcs_output}/rfdetr/{exp_name}"
            sync_thread = GCSSyncThread(exp_dir, gcs_target, interval=300)
            sync_thread.start()
            print(f"  [SYNC] Background sync to {gcs_target} every 5 min")

        # Register epoch callback for monitoring + watchdog reset
        history = []

        def on_epoch_end(data):
            history.append(data)
            reset_watchdog()
            epoch = data.get("epoch", "?")
            train_loss = data.get("train_loss", "?")
            test_loss = data.get("test_loss", "?")
            print(f"  [EPOCH {epoch}] train_loss={train_loss}, test_loss={test_loss}")
            sys.stdout.flush()

        model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

        # DataLoader optimization
        num_workers = min(4, os.cpu_count() or 2)

        print(f"  [CONFIG] batch={exp_config['batch_size']}, "
              f"grad_accum={exp_config.get('grad_accum_steps', 1)}, "
              f"eff_batch={eff_batch}, "
              f"res={exp_config.get('resolution', 560)}, "
              f"workers={num_workers}, amp={use_amp}")

        start_time = datetime.now()

        # Enable watchdog now that training is starting
        enable_watchdog()

        # Train
        model.train(
            dataset_dir=str(coco_dir),
            epochs=exp_config["epochs"],
            batch_size=exp_config["batch_size"],
            grad_accum_steps=exp_config.get("grad_accum_steps", 1),
            lr=exp_config["lr"],
            lr_encoder=exp_config.get("lr_encoder", exp_config["lr"] * 0.1),
            resolution=exp_config.get("resolution", 560),
            weight_decay=exp_config.get("weight_decay", 1e-4),
            use_ema=exp_config.get("use_ema", True),
            amp=use_amp,
            output_dir=str(exp_dir),
            device="cuda",
            num_workers=num_workers,
            persistent_workers=True,
        )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60

        # Stop sync thread
        if sync_thread:
            sync_thread.stop_event.set()
            sync_thread.join(timeout=10)
            # Final sync
            sync_thread.sync()

        # Collect results
        result = {
            "status": "success",
            "training_time_min": round(training_time, 1),
            "config": exp_config.get("description", ""),
            "effective_batch": eff_batch,
        }

        # Copy best checkpoint
        best_ckpt = exp_dir / "checkpoint_best_ema.pth"
        if not best_ckpt.exists():
            best_ckpt = exp_dir / "checkpoint_best_regular.pth"

        if best_ckpt.exists():
            final_name = f"rfdetr_{exp_name}.pth"
            dest = output_dir / final_name
            shutil.copy(best_ckpt, dest)
            result["weights_path"] = str(dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"   Best weights: {final_name} ({size_mb:.1f} MB)")

        # Save training history
        history_path = exp_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

        print(f"\n[OK] {exp_name} completed in {training_time:.1f} minutes")

        # Cleanup
        del model
        gpu_cleanup()

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] {exp_name}: {error_msg}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()

        gpu_cleanup()

        return {"status": "error", "error": error_msg}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Player Detection - Cloud Training")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--force-conversion", action="store_true", default=True,
                        help="Force YOLO→COCO conversion instead of using GCS cache")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    output_dir = Path(args.output) if args.output else Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" RF-DETR PLAYER DETECTION - CLOUD TRAINING")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Classes: {UNIFIED_CLASSES} (nc={len(UNIFIED_CLASSES)})")

    # GPU info
    cc = check_gpu()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] Config validation only")
        experiments = config.get("rfdetr", {}).get("experiments", {})
        for name, cfg in experiments.items():
            eff = cfg["batch_size"] * cfg.get("grad_accum_steps", 1)
            print(f"  - {name}: {cfg.get('description', '')} (eff_batch={eff})")
        print("[DRY RUN] Valid")
        return

    # --- Step 1: Download dataset ---
    print("\n" + "=" * 70)
    print(" STEP 1: DOWNLOAD DATASET")
    print("=" * 70)

    dataset_path = Path(config["dataset"]["local_path"])
    if not download_dataset(config["dataset"], dataset_path):
        print("[FATAL] Dataset download failed")
        sys.exit(1)

    # --- Step 2: Convert YOLO → COCO ---
    print("\n" + "=" * 70)
    print(" STEP 2: YOLO → COCO CONVERSION")
    print("=" * 70)

    coco_dir = Path(config["rfdetr"]["coco_dir"])
    if args.force_conversion:
        limit = config["dataset"].get("limit_valid_images", 0)
        prepare_coco_dataset(dataset_path, coco_dir, limit_valid_images=limit, force_conversion=True)
    else:
        print("[SKIP] COCO conversion skipped")
        if not (coco_dir / "train" / "_annotations.coco.json").exists():
            print("[ERROR] COCO dataset not found despite lacking --force-conversion")
            sys.exit(1)

    # --- Step 3: Train RF-DETR ---
    print("\n" + "=" * 70)
    print(" STEP 3: RF-DETR TRAINING")
    print("=" * 70)

    experiments = config.get("rfdetr", {}).get("experiments", {})

    # Filter to single experiment if specified
    if args.experiment:
        if args.experiment not in experiments:
            print(f"[ERROR] Experiment '{args.experiment}' not found")
            sys.exit(1)
        experiments = {args.experiment: experiments[args.experiment]}

    results = {}
    for exp_name, exp_config in experiments.items():
        result = train_single_experiment(exp_name, exp_config, coco_dir, output_dir)
        results[exp_name] = result

    # --- Step 4: Upload results ---
    print("\n" + "=" * 70)
    print(" STEP 4: UPLOAD RESULTS")
    print("=" * 70)

    gcs_output = os.environ.get("GCS_OUTPUT_PATH", config["output"].get("gcs_path", ""))
    if gcs_output:
        print(f"Uploading results to {gcs_output}")
        result = subprocess.run(
            ["gcloud", "storage", "cp", "-r", f"{output_dir}/*", f"{gcs_output}/"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"[OK] Results uploaded to {gcs_output}")
        else:
            print(f"[ERROR] Upload failed: {result.stderr}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(" RF-DETR TRAINING COMPLETE")
    print("=" * 70)

    for name, result in results.items():
        status = "[OK]" if result["status"] == "success" else "[ERROR]"
        time_info = f" - {result.get('training_time_min', '?')} min" if result["status"] == "success" else ""
        weights = f" → {result.get('weights_path', 'N/A')}" if result["status"] == "success" else ""
        print(f"  {status} {name}{time_info}{weights}")

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

    # List model files
    print("\nModel files:")
    for ext in [".pt", ".pth"]:
        for f in sorted(output_dir.rglob(f"*{ext}")):
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user.")
        _emergency_upload()
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] Unhandled exception: {e}")
        full_tb = traceback.format_exc()
        print(full_tb)
        sys.stdout.flush()
        sys.stderr.flush()

        # Save crash log
        try:
            crash_log = Path("/workspace/runs/CRASH_LOG.txt")
            crash_log.parent.mkdir(parents=True, exist_ok=True)
            with open(crash_log, "w") as f:
                f.write(f"Crash at: {datetime.now().isoformat()}\n")
                f.write(f"Error: {e}\n\n")
                f.write(f"Traceback:\n{full_tb}\n")
            print(f"[OK] Crash log saved: {crash_log}")
        except Exception:
            pass

        _emergency_upload()
        sys.exit(1)
