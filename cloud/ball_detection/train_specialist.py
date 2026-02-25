"""
YOLO Ball Detection - Cloud Specialist Training Script
Trains YOLO v11 exclusively for ball detection (class 5) with custom hyperparameters (*kwargs) and P2 topology support.
Designed for Vertex AI Custom Spot Jobs with auto-resume on preemption.

Features:
  - GCS Download: downloads dataset to local SSD for fast I/O
  - Spot Resume: checks GCS for last.pt before training, uses resume=True
  - Per-epoch GCS sync: uploads checkpoints and metrics each epoch
  - Final evaluation: runs predictions on test images with best.pt
  - Global crash handler: uploads error log to GCS on any failure
  - Watchdog: kills process if no progress for 60 min (only during training)
"""

import os
import gc
import sys
import time
import signal
import traceback

# ============================================================
# PHASE 0: Validate ALL imports before anything else
# ============================================================
def validate_imports():
    """Validate all required imports are available. Fails fast."""
    errors = []

    required_stdlib = ["yaml", "subprocess", "random", "json"]
    for mod_name in required_stdlib:
        try:
            __import__(mod_name)
        except ImportError:
            errors.append(f"Missing stdlib module: {mod_name}")

    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
    except ImportError:
        errors.append("torch not installed")

    try:
        import numpy as np
        v = int(np.__version__.split(".")[0])
        if v >= 2:
            errors.append(f"numpy {np.__version__} >= 2.0 will break PyTorch ABI")
        else:
            print(f"[OK] numpy {np.__version__}")
    except ImportError:
        errors.append("numpy not installed")

    try:
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        errors.append("ultralytics not installed (pip install ultralytics)")

    try:
        import cv2
        print(f"[OK] opencv {cv2.__version__}")
    except ImportError:
        errors.append("opencv not installed")

    try:
        import yaml
        print(f"[OK] PyYAML {yaml.__version__}")
    except ImportError:
        errors.append("PyYAML not installed")

    try:
        import PIL
        print(f"[OK] Pillow {PIL.__version__}")
    except ImportError:
        errors.append("Pillow not installed")

    try:
        import matplotlib
        print(f"[OK] matplotlib {matplotlib.__version__}")
    except ImportError:
        errors.append("matplotlib not installed")

    if errors:
        print(f"\n[FATAL] {len(errors)} import error(s):")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"[OK] All 7 imports validated")
    return True


# ============================================================
# Now safe to import everything
# ============================================================
import yaml
import json
import random
import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================
# Watchdog: kill process if stuck during TRAINING (not download)
# ============================================================
WATCHDOG_TIMEOUT = 3600  # 60 minutes (generous for large datasets)
_watchdog_enabled = False


def watchdog_handler(signum, frame):
    """Called by SIGALRM if training appears stuck."""
    if not _watchdog_enabled:
        return  # Watchdog not active yet (still downloading data)
    print("\n[FATAL] WATCHDOG: No progress for 60 minutes. Killing process.")
    print("  This usually means training is hung (OOM, deadlock, data issue).")
    try:
        _emergency_upload()
    except Exception:
        pass
    sys.exit(2)


def enable_watchdog():
    """Enable the watchdog timer (call AFTER data download)."""
    global _watchdog_enabled
    _watchdog_enabled = True
    reset_watchdog()


def reset_watchdog():
    """Reset the watchdog timer. Called periodically during training."""
    if not _watchdog_enabled:
        return
    try:
        signal.alarm(WATCHDOG_TIMEOUT)
    except AttributeError:
        pass  # Windows doesn't have signal.alarm


def _emergency_upload():
    """Upload any existing output to GCS on crash."""
    gcs_output = os.environ.get("GCS_OUTPUT_PATH", "")
    output_dir = os.environ.get("BALL_OUTPUT_DIR", "")
    if gcs_output and output_dir and Path(output_dir).exists():
        print(f"  Emergency upload to {gcs_output}...")
        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", output_dir, gcs_output],
            check=False, timeout=120,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        error_log = Path(output_dir) / "CRASH_LOG.txt"
        if error_log.exists():
            subprocess.run(
                ["gcloud", "storage", "cp", str(error_log), f"{gcs_output}/CRASH_LOG.txt"],
                check=False, timeout=30,
            )


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
    import tarfile

    print("\n" + "=" * 60)
    print(" DATASET DOWNLOAD")
    print("=" * 60)

    tar_path = config_dataset.get("tar_path", "")
    gcs_path = config_dataset.get("gcs_path", "")
    extract_dir = local_path.parent  # Extract into parent, tar contains the folder

    print(f"  Tar source:    {tar_path or 'not configured'}")
    print(f"  Raw source:    {gcs_path or 'not configured'}")
    print(f"  Local target:  {local_path}")

    # --- Check if dataset already exists locally (Spot restart) ---
    if local_path.exists():
        all_ok = True
        for subdir in ["images/train", "images/test", "labels/train", "labels/test"]:
            p = local_path / subdir
            if not p.exists() or len(list(p.glob("*"))) == 0:
                all_ok = False
                break
        if all_ok:
            total = sum(1 for _ in local_path.rglob("*") if _.is_file())
            print(f"  [RESUME] Dataset already exists locally: {total} files")
            return True
        else:
            print(f"  [WARN] Existing dataset incomplete, re-downloading...")

    extract_dir.mkdir(parents=True, exist_ok=True)

    # --- Strategy 1: Download tar.gz (fast!) ---
    if tar_path:
        print(f"\n  [1] Trying tar download: {tar_path}")

        # Check if tar exists in GCS
        check = subprocess.run(
            ["gcloud", "storage", "ls", tar_path],
            capture_output=True, text=True, timeout=30,
        )

        if check.returncode == 0:
            local_tar = Path("/tmp/dataset.tar.gz")

            print(f"  Downloading tar... (single file, should be fast)")
            start = time.time()

            result = subprocess.run(
                ["gcloud", "storage", "cp", tar_path, str(local_tar)],
                capture_output=False,  # Show progress
                timeout=3600,
            )

            if result.returncode != 0:
                print(f"  [WARN] Tar download failed, will try fallback")
            else:
                dl_time = time.time() - start
                tar_size = local_tar.stat().st_size / 1e9
                print(f"  [OK] Tar downloaded: {tar_size:.1f} GB in {dl_time:.0f}s")

                # Extract
                print(f"  Extracting to {extract_dir}...")
                start = time.time()

                with tarfile.open(str(local_tar), "r:gz") as tar:
                    tar.extractall(path=str(extract_dir))

                ext_time = time.time() - start
                print(f"  [OK] Extracted in {ext_time:.0f}s")

                # Cleanup tar to free disk
                local_tar.unlink()
                print(f"  [OK] Tar file removed (freed {tar_size:.1f} GB disk)")

                # Verify
                if local_path.exists():
                    total = sum(1 for _ in local_path.rglob("*") if _.is_file())
                    print(f"  [OK] Dataset ready: {total} files at {local_path}")
                    return True
                else:
                    # Tar might have a different root folder name
                    # Check what was extracted
                    extracted = [d for d in extract_dir.iterdir() if d.is_dir()]
                    print(f"  [WARN] Expected {local_path} but found: {extracted}")
                    if len(extracted) == 1:
                        # Rename to expected path
                        actual = extracted[0]
                        actual.rename(local_path)
                        print(f"  [OK] Renamed {actual.name} -> {local_path.name}")
                        return True
                    print(f"  [ERROR] Could not find extracted dataset")
                    return False
        else:
            print(f"  [INFO] Tar not found in GCS - run prepare_dataset_tar.ps1 first")
            print(f"         Falling back to individual file download (slow)...")

    # --- Strategy 2: Fallback to gcloud cp -r (slow but works) ---
    if gcs_path:
        print(f"\n  [2] Fallback: downloading individual files from {gcs_path}")
        print(f"  WARNING: This will be SLOW for large datasets (42K+ files)")
        print(f"  RECOMMENDATION: Cancel and run prepare_dataset_tar.ps1 first")

        local_path.mkdir(parents=True, exist_ok=True)
        start = time.time()

        result = subprocess.run(
            ["gcloud", "storage", "cp", "-r", f"{gcs_path}/*", str(local_path)],
            capture_output=False,
            timeout=7200,  # 2 hour max
        )

        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  [ERROR] Download failed (exit code {result.returncode})")
            return False

        total = sum(1 for _ in local_path.rglob("*") if _.is_file())
        print(f"  [OK] Download complete: {total} files in {elapsed:.0f}s")
        return True

    print("[FATAL] No dataset source configured (need tar_path or gcs_path)")
    return False


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
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def check_gpu():
    """Validate GPU availability and print diagnostics."""
    print("=" * 60)
    print(" GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[WARN] No GPU detected - training will be extremely slow")
        return False

    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)

    print(f"GPU: {name}")
    print(f"VRAM: {mem_gb:.1f} GB")
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    try:
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        y = x @ x.t()
        loss = y.sum()
        loss.backward()
        del x, y, loss
        torch.cuda.empty_cache()
        print("[OK] GPU forward+backward pass works")
    except Exception as e:
        print(f"[FATAL] GPU compute test failed: {e}")
        sys.exit(1)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    return True


def validate_dataset(dataset_path: Path) -> bool:
    """Verify dataset structure on local disk."""
    print("\n" + "=" * 60)
    print(" DATASET VALIDATION")
    print("=" * 60)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return False

    total_files = 0
    for subdir in ["images/train", "images/test", "labels/train", "labels/test"]:
        p = dataset_path / subdir
        if not p.exists():
            print(f"[ERROR] Missing directory: {p}")
            return False
        n = len(list(p.glob("*")))
        if n == 0:
            print(f"[ERROR] Empty directory: {p}")
            return False
        total_files += n
        print(f"  [OK] {subdir}: {n} files")

    print(f"  [OK] Total: {total_files} files")
    return True


def create_data_yaml(dataset_path: Path, output_path: Path) -> Path:
    """Create YOLO data.yaml for ball detection only (class 5)."""
    data_yaml = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/test",
        "names": {
            0: "player_left",
            1: "player_right",
            2: "goalkeeper_left",
            3: "goalkeeper_right",
            4: "referee",
            5: "ball",
        },
        "nc": 6,
    }

    yaml_path = output_path / "data.yaml"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    with open(yaml_path) as f:
        verify = yaml.safe_load(f)
    assert verify["nc"] == 6, f"data.yaml nc mismatch: {verify['nc']}"
    assert verify["path"] == str(dataset_path), "data.yaml path mismatch"

    print(f"[OK] data.yaml created and verified: {yaml_path}")
    return yaml_path


def check_existing_checkpoint(gcs_output_base: str, exp_name: str, local_dir: Path):
    """
    Check if a previous checkpoint exists in GCS (from a preempted Spot run).
    If found, download it locally for resume.
    Returns local path to last.pt or None.
    """
    if not gcs_output_base:
        return None

    gcs_last_pt = f"{gcs_output_base}/{exp_name}/weights/last.pt"
    local_last_pt = local_dir / exp_name / "weights" / "last.pt"

    print(f"\n  Checking for existing checkpoint: {gcs_last_pt}")

    try:
        result = subprocess.run(
            ["gcloud", "storage", "ls", gcs_last_pt],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode == 0 and gcs_last_pt.replace("gs://", "") in result.stdout.replace("gs://", ""):
            local_last_pt.parent.mkdir(parents=True, exist_ok=True)
            dl_result = subprocess.run(
                ["gcloud", "storage", "cp", gcs_last_pt, str(local_last_pt)],
                capture_output=True, text=True, timeout=120,
            )
            if dl_result.returncode == 0 and local_last_pt.exists():
                size_mb = local_last_pt.stat().st_size / 1e6
                if size_mb < 0.1:
                    print(f"  [WARN] Checkpoint too small ({size_mb:.2f} MB), corrupted. Starting fresh.")
                    local_last_pt.unlink()
                    return None
                print(f"  [RESUME] Found checkpoint: {size_mb:.1f} MB")
                return local_last_pt
            else:
                print(f"  [WARN] Could not download checkpoint: {dl_result.stderr}")
        else:
            print("  [OK] No previous checkpoint - starting fresh")

    except subprocess.TimeoutExpired:
        print("  [WARN] Timeout checking GCS - starting fresh")
    except Exception as e:
        print(f"  [WARN] Error checking checkpoint: {e}")

    return None


def on_train_epoch_end_callback(trainer):
    """Upload results to GCS at the end of each epoch."""
    # Reset watchdog - training is making progress
    reset_watchdog()

    gcs_output = os.environ.get("GCS_OUTPUT_PATH")
    if not gcs_output:
        return

    try:
        save_dir = Path(trainer.save_dir)
        gcs_target = f"{gcs_output}/{save_dir.name}"

        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", str(save_dir), gcs_target],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("  [WARN] Epoch sync timed out (120s), continuing training")
    except Exception as e:
        print(f"  [WARN] Epoch sync failed: {e}")


def train_experiment(
    exp_name: str,
    exp_config: dict,
    data_yaml_path: Path,
    output_dir: Path,
    gcs_output_base: str,
) -> dict:
    """Train a single YOLO experiment with Spot resume support."""
    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f" EXPERIMENT: {exp_name}")
    print(f"  Model:  {exp_config.get('model', 'yolo11n.pt')}")
    print(f"  Epochs: {exp_config.get('epochs', 50)}")
    print(f"  Batch:  {exp_config.get('batch_size', 16)}")
    print(f"  ImgSz:  {exp_config.get('imgsz', 640)}")
    print(f"{'=' * 60}")

    gpu_cleanup()

    model = None
    resume = False
    checkpoint = check_existing_checkpoint(gcs_output_base, exp_name, output_dir)

    try:
        if checkpoint:
            print(f"  [RESUME] Loading checkpoint: {checkpoint}")
            model = YOLO(str(checkpoint))
            resume = True
        else:
            model_name = exp_config.get("model", "yolo11n.pt")
            print(f"  [NEW] Loading pre-trained: {model_name}")
            model = YOLO(model_name)

        if gcs_output_base:
            os.environ["GCS_OUTPUT_PATH"] = gcs_output_base
            model.add_callback("on_train_epoch_end", on_train_epoch_end_callback)

        start_time = datetime.now()

        # Reset watchdog before training starts
        reset_watchdog()

        train_args = {
            "data": str(data_yaml_path),
            "epochs": exp_config.get("epochs", 50),
            "patience": exp_config.get("patience", 15),
            "batch": exp_config.get("batch_size", 16),
            "imgsz": exp_config.get("imgsz", 640),
            "classes": [5],  # Ball only
            "project": str(output_dir),
            "name": exp_name,
            "exist_ok": True,
            "device": "0" if torch.cuda.is_available() else "cpu",
            "save": True,
            "save_period": 10,
            "plots": True,
            "workers": 4,
            "resume": resume,
        }

        # Inject hyperparameters (**kwargs) from config
        hyperparameters = exp_config.get("hyperparameters", {})
        if hyperparameters:
            print(f"  [INFO] Injecting custom hyperparameters: {hyperparameters}")
            train_args.update(hyperparameters)

        results = model.train(**train_args)

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60

        final_metrics = {}
        if hasattr(results, "results_dict"):
            final_metrics = {k: float(v) for k, v in results.results_dict.items()}

        final_metrics["training_time_min"] = round(training_time, 1)
        final_metrics["status"] = "success"
        final_metrics["resumed"] = resume

        # Verify weights were actually saved
        exp_dir = output_dir / exp_name
        best_pt = exp_dir / "weights" / "best.pt"
        last_pt = exp_dir / "weights" / "last.pt"

        if best_pt.exists():
            size_mb = best_pt.stat().st_size / 1e6
            print(f"  [OK] best.pt saved: {size_mb:.1f} MB")
            final_metrics["best_pt_size_mb"] = round(size_mb, 1)
        else:
            print(f"  [WARN] best.pt NOT found at {best_pt}")

        if last_pt.exists():
            print(f"  [OK] last.pt saved: {last_pt.stat().st_size / 1e6:.1f} MB")

        print(f"\n  [OK] {exp_name} completed in {training_time:.1f} minutes")

        # Final full sync to GCS
        if gcs_output_base:
            gcs_target = f"{gcs_output_base}/{exp_name}"
            print(f"  Uploading final results to {gcs_target}...")
            result = subprocess.run(
                ["gcloud", "storage", "rsync", "-r", str(exp_dir), gcs_target],
                check=False, capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                print(f"  [OK] Final results uploaded")
            else:
                print(f"  [WARN] Upload issues: {result.stderr[:200]}")

        return final_metrics

    except Exception as e:
        error_msg = str(e)
        full_traceback = traceback.format_exc()
        print(f"\n  [ERROR] {exp_name}: {error_msg}")
        print(f"  Full traceback:\n{full_traceback}")

        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        error_log = exp_dir / "ERROR_LOG.txt"
        with open(error_log, "w") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Config: {json.dumps(exp_config, indent=2)}\n")
            f.write(f"Resume: {resume}\n\n")
            f.write(f"Error:\n{full_traceback}\n")

        if gcs_output_base:
            try:
                gcs_target = f"{gcs_output_base}/{exp_name}"
                subprocess.run(
                    ["gcloud", "storage", "rsync", "-r", str(exp_dir), gcs_target],
                    check=False, timeout=120,
                )
            except Exception as up_err:
                print(f"  [WARN] Could not upload error logs: {up_err}")

        return {"status": "error", "error": error_msg, "resumed": resume}

    finally:
        if model is not None:
            try:
                del model
            except Exception:
                pass
        gpu_cleanup()


def evaluate_experiments(output_dir: Path, dataset_path: Path, gcs_output_base: str):
    """Evaluate trained models on test images."""
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use("Agg")

    print(f"\n{'=' * 60}")
    print(" EVALUATION")
    print(f"{'=' * 60}")

    test_images_dir = dataset_path / "images" / "test"
    all_test_imgs = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))

    if not all_test_imgs:
        print("[WARN] No test images found for evaluation")
        return

    experiments = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("exp")]

    if not experiments:
        print("[WARN] No experiment directories found for evaluation")
        return

    for exp_dir in experiments:
        exp_name = exp_dir.name
        best_pt = exp_dir / "weights" / "best.pt"

        if not best_pt.exists():
            print(f"  [{exp_name}] No best.pt found, skipping")
            continue

        print(f"\n  [{exp_name}] Evaluating with best.pt...")

        try:
            model = YOLO(str(best_pt))
            sample_imgs = random.sample(all_test_imgs, min(5, len(all_test_imgs)))
            pred_name = f"{exp_name}_predictions"

            model.predict(
                source=[str(i) for i in sample_imgs],
                save=True,
                project=str(output_dir),
                name=pred_name,
                conf=0.25,
                exist_ok=True,
            )

            if gcs_output_base:
                pred_dir = output_dir / pred_name
                if pred_dir.exists():
                    subprocess.run(
                        ["gcloud", "storage", "rsync", "-r",
                         str(pred_dir), f"{gcs_output_base}/{pred_name}"],
                        check=False, timeout=120,
                    )

            print(f"  [{exp_name}] Predictions saved")
            del model
            gpu_cleanup()

        except Exception as e:
            print(f"  [{exp_name}] Evaluation error: {e}")


def main():
    """Main entry point with global crash handler."""
    start_timestamp = datetime.now()

    print("=" * 60)
    print(" YOLO Ball Detection - Vertex AI Training")
    print(f" Started: {start_timestamp.isoformat()}")
    print("=" * 60)

    # --- Phase 0: Validate imports ---
    print("\n[PHASE 0] Validating imports...")
    if not validate_imports():
        print("[FATAL] Missing required libraries. Check setup.py.")
        sys.exit(1)

    # --- Load config ---
    config_path = Path(__file__).parent / "config_specialist.yaml"
    if not config_path.exists():
        config_path = Path("config_specialist.yaml")
    if not config_path.exists():
        print(f"[FATAL] Config file config_specialist.yaml not found.")
        sys.exit(1)

    print(f"\n[OK] Config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate config keys
    for section, key in [("dataset", "local_path"),
                         ("output", "base_dir"), ("output", "gcs_path")]:
        if section not in config or key not in config[section]:
            print(f"[FATAL] Missing config key: {section}.{key}")
            sys.exit(1)

    # Need at least one download source
    if not config["dataset"].get("tar_path") and not config["dataset"].get("gcs_path"):
        print("[FATAL] Need at least dataset.tar_path or dataset.gcs_path in config")
        sys.exit(1)

    set_seed(config.get("seed", 42))

    # Paths
    dataset_local = Path(config["dataset"]["local_path"])
    output_dir = Path(config["output"]["base_dir"])
    gcs_output_base = config["output"]["gcs_path"]

    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GCS_OUTPUT_PATH"] = gcs_output_base
    os.environ["BALL_OUTPUT_DIR"] = str(output_dir)

    # Setup watchdog signal handler (but DON'T enable it yet)
    try:
        signal.signal(signal.SIGALRM, watchdog_handler)
        print("[OK] Watchdog handler registered (will enable after data download)")
    except AttributeError:
        print("[INFO] Watchdog not available (non-POSIX OS)")

    # --- Phase 1: GPU Check ---
    has_gpu = check_gpu()
    if not has_gpu:
        print("[WARN] Training without GPU. This will be very slow.")

    # --- Phase 2: Download Dataset from GCS ---
    if not download_dataset(config["dataset"], dataset_local):
        print("[FATAL] Dataset download failed.")
        sys.exit(1)

    # --- Phase 3: Validate Dataset ---
    if not validate_dataset(dataset_local):
        print("[FATAL] Dataset validation failed after download.")
        sys.exit(1)

    # --- Phase 4: Create data.yaml ---
    data_yaml_path = create_data_yaml(dataset_local, output_dir)

    # --- NOW enable watchdog (data is downloaded, training about to start) ---
    enable_watchdog()
    print("[OK] Watchdog enabled (60 min timeout for training)")

    # --- Phase 5: Train all experiments ---
    experiments = config.get("yolo", {}).get("experiments", {})
    if not experiments:
        print("[FATAL] No experiments defined in config.yaml")
        sys.exit(1)

    print(f"\n[INFO] Will train {len(experiments)} experiment(s):")
    for name, cfg in experiments.items():
        print(f"  - {name}: {cfg.get('model', '?')} | "
              f"{cfg.get('epochs', '?')} epochs | "
              f"batch {cfg.get('batch_size', '?')} | "
              f"imgsz {cfg.get('imgsz', '?')}")

    all_results = {}
    for exp_name, exp_config in experiments.items():
        all_results[exp_name] = train_experiment(
            exp_name=exp_name,
            exp_config=exp_config,
            data_yaml_path=data_yaml_path,
            output_dir=output_dir,
            gcs_output_base=gcs_output_base,
        )

    # --- Phase 6: Evaluate ---
    try:
        evaluate_experiments(output_dir, dataset_local, gcs_output_base)
    except Exception as e:
        print(f"[WARN] Evaluation phase failed (non-fatal): {e}")

    # --- Phase 7: Summary ---
    summary = {
        "started": start_timestamp.isoformat(),
        "finished": datetime.now().isoformat(),
        "total_time_min": round((datetime.now() - start_timestamp).total_seconds() / 60, 1),
        "experiments": all_results,
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if gcs_output_base:
        subprocess.run(
            ["gcloud", "storage", "cp", str(summary_path), f"{gcs_output_base}/training_summary.json"],
            check=False, timeout=30,
        )

    print(f"\n{'=' * 60}")
    print(" TRAINING COMPLETE")
    print(f"{'=' * 60}")

    for name, result in all_results.items():
        status = "[OK]" if result.get("status") == "success" else "[ERROR]"
        resumed = " (resumed)" if result.get("resumed") else ""
        time_info = f" - {result.get('training_time_min', '?')} min" if result.get("status") == "success" else ""
        error_info = f" ({result.get('error', '')})" if result.get("status") == "error" else ""
        print(f"  {status} {name}{resumed}{time_info}{error_info}")

    print(f"\nTotal time: {summary['total_time_min']} min")
    print(f"Results: {output_dir}")
    print(f"GCS: {gcs_output_base}")

    # Disable watchdog
    try:
        signal.alarm(0)
    except AttributeError:
        pass

    failures = [n for n, r in all_results.items() if r.get("status") != "success"]
    if failures:
        print(f"\n[ERROR] {len(failures)} experiment(s) failed: {failures}")
        sys.exit(1)

    print("\n[OK] All experiments succeeded. Exiting.")


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
        print(f"\n{'=' * 60}")
        print(f" UNHANDLED CRASH")
        print(f"{'=' * 60}")
        full_tb = traceback.format_exc()
        print(full_tb)

        try:
            output_dir = os.environ.get("BALL_OUTPUT_DIR", "/workspace/runs")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            crash_log = Path(output_dir) / "CRASH_LOG.txt"
            with open(crash_log, "w") as f:
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Error: {e}\n\n")
                f.write(f"Traceback:\n{full_tb}\n")
            print(f"[OK] Crash log saved: {crash_log}")
        except Exception:
            pass

        _emergency_upload()
        sys.exit(1)
