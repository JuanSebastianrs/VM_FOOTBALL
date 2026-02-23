"""
Sync trained model weights from local/GCS to Google Drive.
Uses gdown for upload or Google Drive API for programmatic access.

Usage:
    python cloud/sync_to_drive.py --config cloud/config.yaml --output /workspace/outputs
"""

import os
import json
import subprocess
import argparse
import yaml
from pathlib import Path
from datetime import datetime


def sync_with_rclone(output_dir: Path, drive_folder: str):
    """
    Sync using rclone (must be configured beforehand).
    
    Setup (one time):
        rclone config
        -> Choose Google Drive
        -> Name it 'gdrive'
    """
    sync_extensions = [".pt", ".pth", ".json", ".csv", ".png"]
    
    files_to_sync = []
    for ext in sync_extensions:
        files_to_sync.extend(output_dir.rglob(f"*{ext}"))
    
    if not files_to_sync:
        print("[Drive] No files to sync")
        return
    
    print(f"[Drive] Syncing {len(files_to_sync)} files to Drive/{drive_folder}/")
    
    # Check if rclone is available
    try:
        subprocess.run(["rclone", "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[Drive] rclone not installed. Install with:")
        print("  curl https://rclone.org/install.sh | sudo bash")
        print("  rclone config  # Setup Google Drive remote as 'gdrive'")
        print("\nAlternatively, use gcloud/gsutil to copy from GCS to Drive manually.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_path = f"gdrive:{drive_folder}/run_{timestamp}"
    
    for f in files_to_sync:
        rel = f.relative_to(output_dir)
        dest = f"{remote_path}/{rel}"
        size_mb = f.stat().st_size / 1e6
        
        print(f"  Uploading: {rel} ({size_mb:.1f} MB)")
        try:
            subprocess.run(
                ["rclone", "copy", str(f), f"{remote_path}/{rel.parent}"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to upload {rel}: {e}")
    
    print(f"[Drive] Sync complete: {remote_path}")


def sync_with_gdrive_api(output_dir: Path, drive_folder: str):
    """
    Sync using Google Drive API via google-auth + googleapiclient.
    Requires service account or OAuth credentials.
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        print("[Drive] google-api-python-client not installed")
        print("  pip install google-api-python-client google-auth-oauthlib")
        return
    
    # Check for credentials file
    creds_path = Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    if not creds_path.exists():
        creds_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    
    if not creds_path.exists():
        print("[Drive] No credentials found. Run:")
        print("  gcloud auth application-default login")
        return
    
    print(f"[Drive] Using credentials: {creds_path}")
    
    # This is a simplified version - for full implementation,
    # consider using PyDrive2 or gdown for easier Drive interaction
    print("[Drive] API sync not fully implemented - use rclone instead")
    print(f"  Manual command: rclone copy {output_dir} gdrive:{drive_folder}/")


def sync_models(config: dict, output_dir: Path):
    """Main sync function called by the pipeline."""
    drive_config = config.get("drive", {})
    drive_folder = drive_config.get("folder_name", "VM_FOOTBALL_Models")
    
    print(f"\n{'='*50}")
    print(f"SYNCING TO GOOGLE DRIVE: {drive_folder}")
    print(f"{'='*50}")
    
    sync_with_rclone(output_dir, drive_folder)


def main():
    parser = argparse.ArgumentParser(description="Sync models to Google Drive")
    parser.add_argument("--config", type=Path, default=Path("cloud/config.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="Output dir to sync")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    sync_models(config, args.output)


if __name__ == "__main__":
    main()
