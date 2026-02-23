"""
Callback system for real-time training monitoring.
Sends metrics to Cloudflare Worker via HTTP POST.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class DashboardCallback:
    """
    Sends training metrics to the Cloudflare Worker dashboard.
    
    Works as a universal callback for both YOLO and RF-DETR training.
    Falls back gracefully if the worker URL is not configured.
    """
    
    def __init__(
        self,
        worker_url: str = "",
        api_key: str = "",
        experiment_name: str = "",
        model_type: str = "",
        total_epochs: int = 0,
        local_log_path: Optional[Path] = None,
    ):
        self.worker_url = worker_url.rstrip("/") if worker_url else ""
        self.api_key = api_key
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.total_epochs = total_epochs
        self.enabled = bool(self.worker_url)
        self.start_time = time.time()
        self.metrics_history = []
        
        # Local log as fallback
        self.local_log_path = local_log_path or Path("outputs/logs")
        self.local_log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.local_log_path / f"{experiment_name}_metrics.jsonl"
        
        if self.enabled:
            print(f"[Dashboard] Connected to: {self.worker_url}")
        else:
            print("[Dashboard] No worker URL configured - logging locally only")
    
    def on_train_start(self, config: Dict[str, Any]) -> None:
        """Called when training begins."""
        payload = {
            "event": "train_start",
            "experiment": self.experiment_name,
            "model_type": self.model_type,
            "total_epochs": self.total_epochs,
            "config": config,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._send(payload)
        self._log_local(payload)
    
    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Called at the end of each epoch."""
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(epoch, 1)) * (self.total_epochs - epoch)
        
        payload = {
            "event": "epoch_end",
            "experiment": self.experiment_name,
            "model_type": self.model_type,
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "progress_pct": round((epoch / max(self.total_epochs, 1)) * 100, 1),
            "metrics": metrics,
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(eta, 1),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.metrics_history.append(payload)
        self._send(payload)
        self._log_local(payload)
    
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training finishes."""
        elapsed = time.time() - self.start_time
        
        payload = {
            "event": "train_end",
            "experiment": self.experiment_name,
            "model_type": self.model_type,
            "total_time_min": round(elapsed / 60, 1),
            "final_metrics": final_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._send(payload)
        self._log_local(payload)
        
        # Save full history
        history_file = self.local_log_path / f"{self.experiment_name}_history.json"
        with open(history_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        print(f"[Dashboard] History saved: {history_file}")
    
    def on_error(self, error_msg: str) -> None:
        """Called when an error occurs."""
        payload = {
            "event": "error",
            "experiment": self.experiment_name,
            "model_type": self.model_type,
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._send(payload)
        self._log_local(payload)
    
    def _send(self, payload: Dict[str, Any]) -> None:
        """Send payload to Cloudflare Worker."""
        if not self.enabled:
            return
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            resp = requests.post(
                f"{self.worker_url}/api/update",
                json=payload,
                headers=headers,
                timeout=5,
            )
            if resp.status_code != 200:
                print(f"[Dashboard] Warning: HTTP {resp.status_code}")
        except requests.RequestException:
            # Silently fail - training should not be interrupted
            pass
    
    def _log_local(self, payload: Dict[str, Any]) -> None:
        """Append payload to local JSONL log."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")


class ConsoleCallback:
    """Simple console printer for training progress."""
    
    def __init__(self, experiment_name: str = ""):
        self.experiment_name = experiment_name
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch: int, total: int, metrics: Dict[str, float]) -> None:
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(epoch, 1)) * (total - epoch)
        
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        eta_min = eta / 60
        
        print(
            f"[{self.experiment_name}] Epoch {epoch}/{total} "
            f"({epoch/total*100:.0f}%) | {metrics_str} | "
            f"ETA: {eta_min:.1f}min"
        )
