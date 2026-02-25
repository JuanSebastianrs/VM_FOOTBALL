"""
VM_FOOTBALL - Pipeline Orchestrator
Runs all training stages sequentially and manages outputs.

Usage:
    python cloud/run_pipeline.py --config cloud/config.yaml
    python cloud/run_pipeline.py --config cloud/config.yaml --dry-run
    python cloud/run_pipeline.py --config cloud/config.yaml --only yolo
    python cloud/run_pipeline.py --config cloud/config.yaml --only rfdetr
"""

import os
import sys
import json
import argparse
import subprocess
import yaml
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def is_vertex_ai() -> bool:
    """Check if running inside Vertex AI."""
    return os.environ.get("VERTEX_AI_JOB", "").lower() == "true"


def download_dataset_gcs_cli(gcs_path: str, local_path: str):
    """Download dataset from GCS using gcloud CLI (faster than Python API)."""
    print(f"  Downloading from {gcs_path} to {local_path}...")
    Path(local_path).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_path, local_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [WARN] gcloud storage failed: {result.stderr}")
        print("  Trying Python API...")
        return False
    print(f"  [OK] Dataset downloaded")
    return True


def upload_results_gcs_cli(local_path: str, gcs_path: str):
    """Upload results to GCS using gcloud CLI."""
    print(f"  Uploading from {local_path} to {gcs_path}...")
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", local_path, gcs_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  [OK] Results uploaded to {gcs_path}")
    else:
        print(f"  [ERROR] Upload failed: {result.stderr}")


def print_header(text: str):
    print(f"\n{'#'*70}")
    print(f"# {text}")
    print(f"{'#'*70}\n")


def check_gpu():
    """Check and print GPU info. Returns (has_gpu, supports_bf16)."""
    import torch
    
    print_header("GPU CHECK")
    
    if not torch.cuda.is_available():
        print("[WARN] No GPU detected - training will be very slow")
        return False, False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)
    supports_bf16 = cc >= (8, 0)
    
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
    print(f"bfloat16 support: {supports_bf16}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    return True, supports_bf16


def check_dataset(config: dict) -> bool:
    """Verify dataset exists."""
    dataset_path = Path(config["dataset"]["local_path"])
    
    print_header("DATASET CHECK")
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print(f"Download from GCS: gsutil -m cp -r {config['dataset']['gcs_path']} {dataset_path}")
        return False
    
    # Check structure
    for subdir in ["images/train", "images/test", "labels/train", "labels/test"]:
        p = dataset_path / subdir
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            return False
        n = len(list(p.glob("*")))
        print(f"  [OK] {subdir}: {n} files")
    
    return True


def download_dataset_from_gcs(config: dict):
    """Download dataset from GCS bucket."""
    from google.cloud import storage
    
    bucket_name = config["gcp"]["bucket_name"]
    local_path = Path(config["dataset"]["local_path"])
    
    print(f"Downloading dataset from gs://{bucket_name}/...")
    local_path.mkdir(parents=True, exist_ok=True)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix="reorganized_dataset/"))
    
    print(f"  Found {len(blobs)} files")
    
    for blob in blobs:
        rel_path = blob.name.replace("reorganized_dataset/", "", 1)
        if not rel_path:
            continue
        local_file = local_path / rel_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
    
    print(f"  [OK] Dataset downloaded to {local_path}")


def run_yolo_training(config: dict, output_dir: Path, dashboard_callback=None) -> dict:
    """Run all YOLO experiments."""
    from training.detection.train_yolo import run_all_experiments
    
    print_header("PHASE 1A: YOLO v11 TRAINING")
    return run_all_experiments(config, output_dir, dashboard_callback)


def run_rfdetr_training(config: dict, output_dir: Path, dashboard_callback=None) -> dict:
    """Run all RF-DETR experiments."""
    from training.detection.train_rfdetr import run_all_experiments
    
    print_header("PHASE 1B: RF-DETR TRAINING")
    return run_all_experiments(config, output_dir, dashboard_callback)


def generate_comparison_report(
    yolo_results: dict,
    rfdetr_results: dict,
    output_dir: Path,
):
    """Generate a comparison report between YOLO and RF-DETR."""
    print_header("COMPARISON REPORT")
    
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "yolo_experiments": yolo_results,
        "rfdetr_experiments": rfdetr_results,
    }
    
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("YOLO Results:")
    for name, r in yolo_results.items():
        status = "[OK]" if r.get("status") == "success" else "[ERROR]"
        t = r.get("training_time_min", "?")
        print(f"  {status} {name}: {t} min")
    
    print("\nRF-DETR Results:")
    for name, r in rfdetr_results.items():
        status = "[OK]" if r.get("status") == "success" else "[ERROR]"
        t = r.get("training_time_min", "?")
        print(f"  {status} {name}: {t} min")
    
    print(f"\nReport saved: {report_path}")
    return report


def upload_to_gcs(config: dict, output_dir: Path):
    """Upload trained models and results to GCS."""
    print_header("UPLOADING TO GCS")
    
    try:
        from google.cloud import storage
        
        models_bucket = config["gcp"]["models_bucket"]
        client = storage.Client()
        bucket = client.bucket(models_bucket)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        uploaded = 0
        for ext in [".pt", ".pth", ".json", ".csv", ".png"]:
            for f in output_dir.rglob(f"*{ext}"):
                blob_name = f"run_{timestamp}/{f.relative_to(output_dir)}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(f))
                size_mb = f.stat().st_size / 1e6
                print(f"  Uploaded: {blob_name} ({size_mb:.1f} MB)")
                uploaded += 1
        
        print(f"\n[OK] {uploaded} files uploaded to gs://{models_bucket}/run_{timestamp}/")
        
    except ImportError:
        print("[WARN] google-cloud-storage not installed, skipping GCS upload")
    except Exception as e:
        print(f"[ERROR] GCS upload failed: {e}")


def sync_to_drive(config: dict, output_dir: Path):
    """Sync models to Google Drive."""
    try:
        from cloud.sync_to_drive import sync_models
        sync_models(config, output_dir)
    except ImportError:
        print("[WARN] sync_to_drive not available, skipping Drive sync")
    except Exception as e:
        print(f"[WARN] Drive sync failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="VM_FOOTBALL Training Pipeline")
    parser.add_argument("--config", type=Path, default=Path("cloud/config.yaml"),
                        help="Path to config file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory override")
    parser.add_argument("--only", type=str, choices=["yolo", "rfdetr"],
                        help="Run only specific model type")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip GCS/Drive upload")
    parser.add_argument("--download-dataset", action="store_true",
                        help="Download dataset from GCS before training")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 1 epoch with minimal config to validate pipeline end-to-end")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output) if args.output else Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("VM_FOOTBALL TRAINING PIPELINE")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Time: {datetime.now().isoformat()}")
    
    pipeline_start = time.time()
    
    # --- Step 0: GPU ---
    has_gpu, supports_bf16 = check_gpu()
    
    # Smoke test banner
    if args.smoke_test:
        print_header("SMOKE TEST MODE")
        print("Running 1 epoch with minimal config to validate pipeline end-to-end.")
        print("This validates: GPU -> deps -> dataset -> model init -> forward/backward -> checkpoint")
    
    # bfloat16 info: log warning if RF-DETR will run on non-bf16 GPU
    rfdetr_enabled = config.get("rfdetr", {}).get("enabled", False)
    only_model = args.only
    will_run_rfdetr = rfdetr_enabled and only_model in (None, "rfdetr")
    if will_run_rfdetr and has_gpu and not supports_bf16:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"\n[WARN] RF-DETR will train with amp=False (FP32) on {gpu_name} (cc {cc[0]}.{cc[1]})")
        print(f"  bfloat16 requires cc >= 8.0 (L4/A100/H100)")
        print(f"  Training will work but may be slower. Request NVIDIA_L4 quota for better performance.")
    
    # --- Step 1: Dataset ---
    # Auto-download in Vertex AI or if flag is set
    gcs_dataset = os.environ.get("GCS_DATASET_PATH", config["dataset"]["gcs_path"])
    if is_vertex_ai() or args.download_dataset:
        print_header("DOWNLOADING DATASET FROM GCS")
        if not download_dataset_gcs_cli(gcs_dataset, str(Path(config["dataset"]["local_path"]).parent)):
            download_dataset_from_gcs(config)
    
    dataset_ok = check_dataset(config)
    
    if args.dry_run:
        print_header("DRY RUN COMPLETE")
        print(f"GPU: {'Yes' if has_gpu else 'No'} (bf16: {supports_bf16})")
        print(f"Dataset: {'OK' if dataset_ok else 'MISSING'}")
        
        if config.get("yolo", {}).get("enabled"):
            print("\nYOLO experiments:")
            for name, cfg in config["yolo"]["experiments"].items():
                print(f"  - {name}: {cfg.get('description', '')}")
        
        if config.get("rfdetr", {}).get("enabled"):
            print("\nRF-DETR experiments:")
            for name, cfg in config["rfdetr"]["experiments"].items():
                print(f"  - {name}: {cfg.get('description', '')}")
        
        return
    
    # --- Smoke test overrides ---
    if args.smoke_test:
        print_header("APPLYING SMOKE TEST OVERRIDES")
        print("Overriding all experiments: epochs=1, batch_size=2, resolution=560")
        
        # Override YOLO experiments
        if config.get("yolo", {}).get("experiments"):
            for name in config["yolo"]["experiments"]:
                config["yolo"]["experiments"][name]["epochs"] = 1
                config["yolo"]["experiments"][name]["batch_size"] = 2
                config["yolo"]["experiments"][name]["patience"] = 1
                print(f"  [YOLO] {name}: epochs=1, batch=2")
        
        # Override RF-DETR experiments
        if config.get("rfdetr", {}).get("experiments"):
            for name in config["rfdetr"]["experiments"]:
                config["rfdetr"]["experiments"][name]["epochs"] = 1
                config["rfdetr"]["experiments"][name]["batch_size"] = 2
                config["rfdetr"]["experiments"][name]["grad_accum_steps"] = 1
                # Keep config resolution (don't override — the config already has optimized values)
                res = config["rfdetr"]["experiments"][name].get("resolution", 560)
                print(f"  [RF-DETR] {name}: epochs=1, batch=2, grad_accum=1, res={res} (from config)")
    
    if not dataset_ok:
        print("[ERROR] Dataset not found. Use --download-dataset or verify paths.")
        sys.exit(1)
    
    # --- Setup dashboard callback ---
    dashboard_callback = None
    dash_config = config.get("dashboard", {})
    if dash_config.get("enabled") and dash_config.get("worker_url"):
        from cloud.callbacks import DashboardCallback
        dashboard_callback = DashboardCallback(
            worker_url=dash_config["worker_url"],
            api_key=dash_config.get("api_key", ""),
            local_log_path=Path(config["output"]["logs_dir"]),
        )
    
    # --- Step 2: Train YOLO ---
    yolo_results = {}
    if config.get("yolo", {}).get("enabled") and args.only in (None, "yolo"):
        yolo_results = run_yolo_training(config, output_dir, dashboard_callback)
    
    # --- Step 3: Train RF-DETR ---
    rfdetr_results = {}
    if config.get("rfdetr", {}).get("enabled") and args.only in (None, "rfdetr"):
        rfdetr_results = run_rfdetr_training(config, output_dir, dashboard_callback)
    
    # --- Step 4: Comparison Report ---
    if yolo_results or rfdetr_results:
        generate_comparison_report(yolo_results, rfdetr_results, output_dir)
    
    # --- Step 5: Upload ---
    if not args.skip_upload:
        # In Vertex AI, use GCS_OUTPUT_PATH env var
        gcs_output = os.environ.get("GCS_OUTPUT_PATH", "")
        if gcs_output:
            print_header("UPLOADING RESULTS TO GCS")
            upload_results_gcs_cli(str(output_dir) + "/", gcs_output)
        else:
            upload_to_gcs(config, output_dir)
        sync_to_drive(config, output_dir)
    
    # --- Done ---
    pipeline_time = (time.time() - pipeline_start) / 60
    print_header("PIPELINE COMPLETE")
    print(f"Total time: {pipeline_time:.1f} minutes")
    print(f"Output: {output_dir}")
    
    # List final model files
    print("\nModel files:")
    for ext in [".pt", ".pth"]:
        for f in sorted(output_dir.rglob(f"*{ext}")):
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
