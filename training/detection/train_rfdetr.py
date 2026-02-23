"""
RF-DETR Training Script for Cloud.
Trains multiple RF-DETR experiments (Base/Large) with COCO format datasets.
Includes YOLO->COCO conversion and dashboard monitoring.
"""

import os
import sys
import gc
import json
import shutil
import argparse
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from PIL import Image
import threading
import time
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# YOLO -> COCO Conversion
# ============================================================

# Unified classes for RF-DETR
UNIFIED_CLASSES = ["player", "goalkeeper"]
CLASS_MAPPING = {
    0: 0,   # player_left -> player
    1: 0,   # player_right -> player
    2: 1,   # goalkeeper_left -> goalkeeper
    3: 1,   # goalkeeper_right -> goalkeeper
    4: -1,  # referee -> IGNORE
    5: -1,  # ball -> IGNORE
}
COCO_CATEGORIES = [
    {"id": i, "name": name, "supercategory": "football"}
    for i, name in enumerate(UNIFIED_CLASSES)
]


def convert_yolo_to_coco(images_dir: Path, labels_dir: Path, output_dir: Path, split: str) -> Path:
    """Convert YOLO format labels to COCO format for RF-DETR."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    coco = {
        "info": {"description": "VM_FOOTBALL", "version": "2.0"},
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }
    
    ann_id = 0
    images = sorted(images_dir.glob("*.jpg"))
    
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


def _copy_images_only(dataset_path: Path, coco_dir: Path):
    """Copy images to COCO directory without converting labels (used with GCS cache)."""
    # Train images
    train_dir = coco_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    src_train = dataset_path / "images" / "train"
    if src_train.exists():
        for img in src_train.glob("*.jpg"):
            dest = train_dir / img.name
            if not dest.exists():
                shutil.copy(img, dest)
    
    # Valid images (from test split in YOLO format)
    valid_dir = coco_dir / "valid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    src_valid = dataset_path / "images" / "test"
    if src_valid.exists():
        for img in src_valid.glob("*.jpg"):
            dest = valid_dir / img.name
            if not dest.exists():
                shutil.copy(img, dest)
    
    # Test images (subset of valid)
    test_dir = coco_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_json = test_dir / "_annotations.coco.json"
    if test_json.exists():
        with open(test_json) as f:
            test_coco = json.load(f)
        for img_info in test_coco["images"]:
            src = valid_dir / img_info["file_name"]
            dest = test_dir / img_info["file_name"]
            if src.exists() and not dest.exists():
                shutil.copy(src, dest)


def prepare_coco_dataset(dataset_path: Path, coco_dir: Path, force_conversion: bool = False) -> Path:
    """Full YOLO->COCO conversion pipeline with GCS caching."""
    
    # Check if COCO dataset already exists locally (skip expensive conversion)
    train_json = coco_dir / "train" / "_annotations.coco.json"
    valid_json = coco_dir / "valid" / "_annotations.coco.json"
    test_json = coco_dir / "test" / "_annotations.coco.json"
    
    if not force_conversion and train_json.exists() and valid_json.exists() and test_json.exists():
        print("[RF-DETR] COCO dataset already exists locally - skipping conversion")
        for split in ["train", "valid", "test"]:
            n_imgs = len(list((coco_dir / split).glob("*.jpg")))
            print(f"   [CACHED] {split}/: {n_imgs} images")
        return coco_dir
    
    # Check GCS cache (download pre-converted COCO annotations)
    gcs_coco_cache = os.environ.get("GCS_COCO_CACHE", "gs://vm-football-data/coco_cache")
    if not force_conversion:
        try:
            print(f"[RF-DETR] Checking GCS COCO cache: {gcs_coco_cache}")
            result = subprocess.run(
                ["gcloud", "storage", "ls", f"{gcs_coco_cache}/train/_annotations.coco.json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print("[RF-DETR] Found COCO cache in GCS! Downloading...")
                # Download only JSON annotations (images are symlinked/copied separately)
                for split in ["train", "valid", "test"]:
                    split_dir = coco_dir / split
                    split_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        ["gcloud", "storage", "cp",
                         f"{gcs_coco_cache}/{split}/_annotations.coco.json",
                         str(split_dir / "_annotations.coco.json")],
                        capture_output=True, text=True,
                    )
                # Still need to copy/symlink images
                print("[RF-DETR] Annotations downloaded, copying images...")
                _copy_images_only(dataset_path, coco_dir)
                print("[RF-DETR] COCO dataset restored from GCS cache (saved ~15 min)")
                for split in ["train", "valid", "test"]:
                    n_imgs = len(list((coco_dir / split).glob("*.jpg")))
                    print(f"   [GCS-CACHED] {split}/: {n_imgs} images")
                return coco_dir
            else:
                print("[RF-DETR] No GCS cache found, running full conversion...")
        except Exception as e:
            print(f"[RF-DETR] GCS cache check failed ({e}), running full conversion...")
    
    print("[RF-DETR] Converting YOLO dataset to COCO format...")
    
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
    
    # Valid
    convert_yolo_to_coco(
        dataset_path / "images" / "test",
        dataset_path / "labels" / "test",
        coco_dir, "valid",
    )
    
    # Test (RF-DETR requires it) - subset of validation
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
        "info": {"description": "VM_FOOTBALL Test", "version": "2.0"},
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
    
    # Verify
    print("\n   Verifying COCO structure:")
    for split in ["train", "valid", "test"]:
        split_path = coco_dir / split
        json_ok = (split_path / "_annotations.coco.json").exists()
        n_imgs = len(list(split_path.glob("*.jpg")))
        status = "[OK]" if json_ok and n_imgs > 0 else "[ERROR]"
        print(f"   {status} {split}/: {n_imgs} images, json={json_ok}")
    
    # Upload annotations to GCS cache for future runs
    gcs_coco_cache = os.environ.get("GCS_COCO_CACHE", "gs://vm-football-data/coco_cache")
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
        raise ValueError(f"Unknown model class: {class_name}. Options: {list(models.keys())}")
    return models[class_name]


class GCSSyncThread(threading.Thread):
    def __init__(self, local_dir: Path, gcs_target: str, interval: int = 300):
        super().__init__()
        self.local_dir = local_dir
        self.gcs_target = gcs_target
        self.interval = interval
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        print(f"[RF-DETR] Starting background sync to {self.gcs_target}")
        while not self.stop_event.is_set():
            try:
                self.sync()
            except Exception as e:
                print(f"[RF-DETR] Background sync failed: {e}")
            
            # Sleep in small chunks to allow quick stop
            for _ in range(self.interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
        
        # Final sync
        try:
            print("[RF-DETR] Final background sync...")
            self.sync()
        except Exception as e:
            print(f"[RF-DETR] Final sync failed: {e}")

    def sync(self):
        if not self.local_dir.exists():
            return
            
        cmd = ["gcloud", "storage", "rsync", "-r", str(self.local_dir), self.gcs_target]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print(f"[RF-DETR] Synced to {self.gcs_target}") -- too noisy


def train_single_experiment(
    exp_name: str,
    exp_config: dict,
    coco_dir: Path,
    output_dir: Path,
    dashboard_callback=None,
) -> dict:
    """Train a single RF-DETR experiment."""
    print(f"\n{'='*70}")
    print(f"RF-DETR EXPERIMENT: {exp_name}")
    print(f"  {exp_config.get('description', '')}")
    eff_batch = exp_config["batch_size"] * exp_config.get("grad_accum_steps", 1)
    print(f"  batch={exp_config['batch_size']}x{exp_config.get('grad_accum_steps', 1)}={eff_batch}, epochs={exp_config['epochs']}")
    print(f"{'='*70}")
    
    # Clean GPU aggressively
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Notify dashboard
    if dashboard_callback:
        dashboard_callback.total_epochs = exp_config["epochs"]
        dashboard_callback.experiment_name = exp_name
        dashboard_callback.on_train_start(exp_config)
    
    try:
        # Create model
        ModelClass = get_model_class(exp_config["model_class"])
        model = ModelClass()
        
        # Start background sync if GCS path is available
        sync_thread = None
        if os.environ.get("GCS_OUTPUT_PATH"):
            gcs_base = os.environ["GCS_OUTPUT_PATH"]
            gcs_target = f"{gcs_base}/rfdetr/{exp_name}"
            sync_thread = GCSSyncThread(exp_dir, gcs_target, interval=300)
            sync_thread.start()
        
        start_time = datetime.now()
        
        # AMP Strategy:
        # - T4 (cc 7.5): supports FP16 (65 TFLOPS vs 8.1 FP32) but NOT bf16
        # - L4/A100/H100 (cc >= 8.0): supports both FP16 and BF16
        # CRITICAL: rfdetr library hardcodes torch.bfloat16 in autocast.
        # On T4, we monkey-patch autocast to redirect bf16 -> fp16.
        use_amp = True
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            gpu_name = torch.cuda.get_device_name(0)
            if cc < (8, 0):
                # T4: Wrap autocast to intercept bf16 -> fp16
                # Cannot subclass torch.amp.autocast (C++ extension)
                # Instead, use a wrapper function that returns a properly configured instance
                _OrigAutocast = torch.amp.autocast
                def _fp16_autocast(*args, **kwargs):
                    if 'dtype' in kwargs and kwargs['dtype'] == torch.bfloat16:
                        kwargs['dtype'] = torch.float16
                    return _OrigAutocast(*args, **kwargs)
                torch.amp.autocast = _fp16_autocast
                torch.cuda.amp.autocast = _fp16_autocast
                
                torch.set_float32_matmul_precision('medium')
                os.environ['TORCH_AMP_DTYPE'] = 'float16'
                print(f"  [OK] {gpu_name} (cc {cc[0]}.{cc[1]}): Patched autocast bf16->fp16")
                print(f"  [INFO] T4 Tensor Cores: 65 TFLOPS FP16 vs 8.1 TFLOPS FP32")
            else:
                print(f"  [OK] {gpu_name} (cc {cc[0]}.{cc[1]}): AMP enabled with BF16")
        
        # DataLoader optimization: use more CPU workers
        # n1-standard-8 has 8 vCPUs, 4 workers is optimal
        num_workers = min(4, os.cpu_count() or 2)
        
        print(f"  [CONFIG] batch={exp_config['batch_size']}, "
              f"grad_accum={exp_config.get('grad_accum_steps', 1)}, "
              f"eff_batch={exp_config['batch_size'] * exp_config.get('grad_accum_steps', 1)}, "
              f"res={exp_config.get('resolution', 560)}, "
              f"workers={num_workers}, amp={use_amp}")
        
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
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            num_workers=num_workers,
            persistent_workers=True,
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60
        
        # Stop sync thread
        if sync_thread:
            sync_thread.stop_event.set()
            sync_thread.join()
        
        # Collect results
        result = {
            "status": "success",
            "training_time_min": round(training_time, 1),
            "config": exp_config.get("description", ""),
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
        
        if dashboard_callback:
            dashboard_callback.on_train_end(result)
        
        print(f"\n[OK] {exp_name} completed in {training_time:.1f} minutes")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"\n[ERROR] {exp_name}: {error_msg}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if dashboard_callback:
            dashboard_callback.on_error(error_msg)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return {"status": "error", "error": error_msg}


def run_all_experiments(config: dict, output_dir: Path, dashboard_callback=None) -> dict:
    """Run all RF-DETR experiments."""
    dataset_path = Path(config["dataset"]["local_path"])
    coco_dir = Path(config.get("rfdetr", {}).get("coco_dir", "/workspace/dataset_coco"))
    
    # Convert dataset
    prepare_coco_dataset(dataset_path, coco_dir)
    
    experiments = config.get("rfdetr", {}).get("experiments", {})
    if not experiments:
        print("[RF-DETR] No experiments defined")
        return {}
    
    results_all = {}
    
    for exp_name, exp_config in experiments.items():
        results_all[exp_name] = train_single_experiment(
            exp_name=exp_name,
            exp_config=exp_config,
            coco_dir=coco_dir,
            output_dir=output_dir / "rfdetr",
            dashboard_callback=dashboard_callback,
        )
    
    # Save summary
    summary_path = output_dir / "rfdetr" / "experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    
    print(f"\n[RF-DETR] Summary saved: {summary_path}")
    return results_all


def main():
    parser = argparse.ArgumentParser(description="RF-DETR Cloud Training")
    parser.add_argument("--config", type=Path, default=Path("cloud/config.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--skip-conversion", action="store_true",
                        help="Skip YOLO->COCO conversion (if already done)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get("seed", 42))
    
    output_dir = Path(args.output) if args.output else Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[RF-DETR] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        print("[RF-DETR] WARNING: No GPU detected")
    
    if args.dry_run:
        print("[RF-DETR] Dry run - config validation only")
        experiments = config.get("rfdetr", {}).get("experiments", {})
        for name, cfg in experiments.items():
            eff = cfg["batch_size"] * cfg.get("grad_accum_steps", 1)
            print(f"  - {name}: {cfg.get('description', '')} (eff_batch={eff})")
        print("[RF-DETR] Config valid")
        return
    
    # Dashboard callback
    dashboard_callback = None
    dash_config = config.get("dashboard", {})
    if dash_config.get("enabled") and dash_config.get("worker_url"):
        from cloud.callbacks import DashboardCallback
        dashboard_callback = DashboardCallback(
            worker_url=dash_config["worker_url"],
            api_key=dash_config.get("api_key", ""),
            model_type="rfdetr",
            local_log_path=Path(config["output"]["logs_dir"]),
        )
    
    # Filter experiment
    if args.experiment:
        all_exps = config.get("rfdetr", {}).get("experiments", {})
        if args.experiment not in all_exps:
            print(f"[ERROR] Experiment '{args.experiment}' not found")
            return
        config["rfdetr"]["experiments"] = {args.experiment: all_exps[args.experiment]}
    
    results = run_all_experiments(config, output_dir, dashboard_callback)
    
    # Summary
    print(f"\n{'='*70}")
    print("RF-DETR TRAINING COMPLETE")
    print(f"{'='*70}")
    for name, result in results.items():
        status = "[OK]" if result["status"] == "success" else "[ERROR]"
        time_info = f" - {result.get('training_time_min', '?')} min" if result["status"] == "success" else ""
        print(f"  {status} {name}{time_info}")


if __name__ == "__main__":
    main()
