"""
YOLO v11 Training Script for Cloud.
Trains multiple YOLO experiments with hyperparameter configs from config.yaml.
Sends real-time metrics to the Cloudflare dashboard.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_yaml(dataset_path: Path, output_path: Path) -> Path:
    """Create YOLO data.yaml pointing to the dataset."""
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
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"[YOLO] data.yaml created: {yaml_path}")
    return yaml_path


def train_single_experiment(
    exp_name: str,
    exp_config: dict,
    data_yaml_path: Path,
    output_dir: Path,
    dashboard_callback=None,
):
    """Train a single YOLO experiment."""
    from ultralytics import YOLO
    
    print(f"\n{'='*70}")
    print(f"YOLO EXPERIMENT: {exp_name}")
    print(f"  {exp_config.get('description', '')}")
    print(f"{'='*70}")
    
    # Clean GPU
    torch.cuda.empty_cache()
    gc.collect()
    
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Notify dashboard
    if dashboard_callback:
        dashboard_callback.total_epochs = exp_config["epochs"]
        dashboard_callback.experiment_name = exp_name
        dashboard_callback.on_train_start(exp_config)
    
    try:
        # Load model
        model_name = exp_config.get("model", "yolo11n.pt")
        model = YOLO(model_name)
        
        # Register GCS sync callback
        if os.environ.get("GCS_OUTPUT_PATH"):
            model.add_callback("on_train_epoch_end", on_train_epoch_end_callback)
        
        start_time = datetime.now()
        
        # Train
        results = model.train(
            data=str(data_yaml_path),
            epochs=exp_config.get("epochs", 100),
            batch=exp_config.get("batch_size", 16),
            imgsz=exp_config.get("imgsz", 640),
            lr0=exp_config.get("lr0", 0.01),
            lrf=exp_config.get("lrf", 0.01),
            optimizer=exp_config.get("optimizer", "AdamW"),
            augment=exp_config.get("augment", True),
            patience=exp_config.get("patience", 20),
            project=str(output_dir),
            name=exp_name,
            exist_ok=True,
            device="0" if torch.cuda.is_available() else "cpu",
            workers=4,
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60
        
        # Collect final metrics
        final_metrics = {}
        if hasattr(results, "results_dict"):
            final_metrics = {k: float(v) for k, v in results.results_dict.items()}
        
        final_metrics["training_time_min"] = round(training_time, 1)
        
        # Copy best weights
        best_pt = exp_dir / "weights" / "best.pt"
        if best_pt.exists():
            dest = output_dir / f"yolo_{exp_name}_best.pt"
            shutil.copy(best_pt, dest)
            print(f"[YOLO] Best weights saved: {dest}")
            final_metrics["weights_path"] = str(dest)
        
        # Notify dashboard
        if dashboard_callback:
            dashboard_callback.on_train_end(final_metrics)
        
        print(f"\n[OK] {exp_name} completed in {training_time:.1f} minutes")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return {"status": "success", **final_metrics}
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] {exp_name}: {error_msg}")
        if dashboard_callback:
            dashboard_callback.on_error(error_msg)
            
        # Try to upload logs on error
        if os.environ.get("GCS_OUTPUT_PATH"):
            try:
                gcs_base = os.environ["GCS_OUTPUT_PATH"]
                print(f"[YOLO] Uploading crash logs to {gcs_base}...")
                import subprocess
                subprocess.run(["gcloud", "storage", "cp", "-r", str(output_dir), gcs_base], check=False)
            except Exception as up_err:
                print(f"[YOLO] Failed to upload crash logs: {up_err}")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return {"status": "error", "error": error_msg}


def on_train_epoch_end_callback(trainer):
    """Upload results to GCS at the end of each epoch."""
    if not os.environ.get("GCS_OUTPUT_PATH"):
        return
        
    try:
        # Get output directory from trainer
        save_dir = Path(trainer.save_dir)
        gcs_base = os.environ["GCS_OUTPUT_PATH"]
        gcs_target = f"{gcs_base}/yolo/{save_dir.name}"
        
        # Files to sync (results.csv, plots, current weights)
        files_to_sync = [
            save_dir / "results.csv",
            save_dir / "train_batch*.jpg",
            save_dir / "val_batch*.jpg",
            save_dir / "confusion_matrix.png",
            save_dir / "results.png",
            save_dir / "weights" / "last.pt"
        ]
        
        # Construct include pattern for rsync or just cp specific files
        # Using cp for simplicity and specific files
        # However, calling gcloud cp for each file is slow.
        # Better to use rsync on the whole directory but exclude heavy things if needed?
        # Let's just cp the directory recursively but excluding nothing (users want everything)
        # But doing it every epoch might be heavy if weights are huge.
        # last.pt is ~6MB for N, ~50MB for M. It's okay.
        
        # Use gcloud storage rsync which is incremental
        cmd = ["gcloud", "storage", "rsync", "-r", str(save_dir), gcs_target]
        import subprocess
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print(f"[YOLO] Synced to {gcs_target}") -- too noisy
        
    except Exception as e:
        print(f"[YOLO] Sync failed: {e}") 


def run_all_experiments(config: dict, output_dir: Path, dashboard_callback=None):
    """Run all YOLO experiments from config."""
    dataset_path = Path(config["dataset"]["local_path"])
    
    # Create data.yaml
    data_yaml_path = create_data_yaml(dataset_path, output_dir)
    
    experiments = config.get("yolo", {}).get("experiments", {})
    if not experiments:
        print("[YOLO] No experiments defined in config")
        return {}
    
    results_all = {}
    
    for exp_name, exp_config in experiments.items():
        results_all[exp_name] = train_single_experiment(
            exp_name=exp_name,
            exp_config=exp_config,
            data_yaml_path=data_yaml_path,
            output_dir=output_dir / "yolo",
            dashboard_callback=dashboard_callback,
        )
    
    # Save summary
    summary_path = output_dir / "yolo" / "experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    
    print(f"\n[YOLO] Summary saved: {summary_path}")
    return results_all


def main():
    parser = argparse.ArgumentParser(description="YOLO v11 Cloud Training")
    parser.add_argument("--config", type=Path, default=Path("cloud/config.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run specific experiment only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get("seed", 42))
    
    output_dir = Path(args.output) if args.output else Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[YOLO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        print("[YOLO] WARNING: No GPU detected, using CPU")
    
    if args.dry_run:
        print("[YOLO] Dry run - config validation only")
        experiments = config.get("yolo", {}).get("experiments", {})
        for name, cfg in experiments.items():
            print(f"  - {name}: {cfg.get('description', '')}")
        print("[YOLO] Config valid")
        return
    
    # Setup dashboard callback
    dashboard_callback = None
    dash_config = config.get("dashboard", {})
    if dash_config.get("enabled") and dash_config.get("worker_url"):
        from cloud.callbacks import DashboardCallback
        dashboard_callback = DashboardCallback(
            worker_url=dash_config["worker_url"],
            api_key=dash_config.get("api_key", ""),
            model_type="yolo",
            local_log_path=Path(config["output"]["logs_dir"]),
        )
    
    # Filter to single experiment if specified
    if args.experiment:
        all_exps = config.get("yolo", {}).get("experiments", {})
        if args.experiment not in all_exps:
            print(f"[ERROR] Experiment '{args.experiment}' not found")
            print(f"  Available: {list(all_exps.keys())}")
            return
        config["yolo"]["experiments"] = {
            args.experiment: all_exps[args.experiment]
        }
    
    # Run
    results = run_all_experiments(config, output_dir, dashboard_callback)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("YOLO TRAINING COMPLETE")
    print(f"{'='*70}")
    
    any_failure = False
    for name, result in results.items():
        status = "[OK]" if result["status"] == "success" else "[ERROR]"
        if result["status"] != "success":
            any_failure = True
        time_info = f" - {result.get('training_time_min', '?')} min" if result["status"] == "success" else ""
        print(f"  {status} {name}{time_info}")
        
    if any_failure:
        print("\n[ERROR] Some experiments failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
