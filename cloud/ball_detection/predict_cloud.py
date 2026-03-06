"""
Vertex AI Custom Job: ByteTrack Data Generator for SAM 2

Runs YOLO + ByteTrack on video frame sequences from GCS.
Outputs JSON files with bounding-box coordinates per frame
that serve as bounding-box prompt inputs for SAM 2 fine-tuning.

Modes:
  --mode poc   : Process a single sequence directory (for validation)
  --mode full  : Process ALL images under --gcs_train_prefix, grouped by sequence
"""

import os
import re
import cv2
import json
import glob
import time
import argparse
import shutil
from collections import defaultdict
from google.cloud import storage
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================
CONF_THRESHOLD = 0.15
IOU_NMS = 0.45
BYTETRACK_IMGSZ = 1280  # Safe for T4 16GB VRAM


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    if source_blob_name.endswith('/'):
        blobs = list(storage_client.list_blobs(bucket_name, prefix=source_blob_name))
        os.makedirs(destination_file_name, exist_ok=True)
        count = 0
        for b in blobs:
            if not b.name.endswith('/'):
                rel_path = b.name.replace(source_blob_name, "", 1)
                dest = os.path.join(destination_file_name, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                b.download_to_filename(dest)
                count += 1
        print(f"[GCS] Downloaded {count} files from gs://{bucket_name}/{source_blob_name}")
    else:
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"[GCS] Downloaded gs://{bucket_name}/{source_blob_name}")


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"[GCS] Uploaded → gs://{bucket_name}/{destination_blob_name}")


def download_train_images(bucket_name, prefix, local_dir):
    """Downloads all train images and groups them by sequence prefix.
    
    Images are like SNMOT-060_000001.jpg → group key = SNMOT-060
    Returns dict: {seq_name: local_dir_path}
    """
    storage_client = storage.Client()
    blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
    
    # Group blobs by sequence prefix
    seq_blobs = defaultdict(list)
    for b in blobs:
        fname = os.path.basename(b.name)
        if not fname.lower().endswith('.jpg'):
            continue
        # Extract sequence prefix: SNMOT-060_000001.jpg → SNMOT-060
        match = re.match(r'^(SNMOT-\d+)_', fname)
        if match:
            seq_blobs[match.group(1)].append(b)
        else:
            seq_blobs["unknown"].append(b)
    
    print(f"[GCS] Found {sum(len(v) for v in seq_blobs.values())} images "
          f"across {len(seq_blobs)} sequences")
    
    # Download each sequence into its own folder
    seq_dirs = {}
    for seq_name, blobs_list in sorted(seq_blobs.items()):
        seq_dir = os.path.join(local_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        for b in blobs_list:
            fname = os.path.basename(b.name)
            dest = os.path.join(seq_dir, fname)
            b.download_to_filename(dest)
        seq_dirs[seq_name] = seq_dir
        print(f"  {seq_name}: {len(blobs_list)} frames → {seq_dir}")
    
    return seq_dirs


# ============================================================
# ByteTrack Pipeline
# ============================================================
def run_bytetrack(model, frames_dir, seq_name, output_dir="/tmp/results"):
    """Run ByteTrack pipeline on a sequence of frames.
    
    Returns:
        (json_path, stats_dict) or (None, None) on error
    """
    print(f"\n{'='*60}")
    print(f" BYTETRACK: {seq_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{seq_name}_coordinates.json")

    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not image_files:
        print(f"[ERROR] No .jpg images found for {seq_name}")
        return None, None

    first_frame = cv2.imread(image_files[0])
    img_h, img_w = first_frame.shape[:2]

    print(f"  Resolution: {img_w}x{img_h}")
    print(f"  Frames: {len(image_files)}")

    tracking_data = {}
    model.predictor = None  # Reset tracker state for each sequence

    t_start = time.time()
    total_detections = 0

    for frame_idx, img_path in enumerate(image_files):
        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            fps = frame_idx / elapsed if elapsed > 0 and frame_idx > 0 else 0
            eta = (len(image_files) - frame_idx) / fps if fps > 0 else 0
            print(f"  [{seq_name}] Frame {frame_idx}/{len(image_files)} | "
                  f"{fps:.1f} FPS | ETA: {eta:.0f}s")

        frame = cv2.imread(img_path)
        frame_name = os.path.basename(img_path)
        tracking_data[frame_name] = []

        track_results = model.track(
            source=frame, conf=CONF_THRESHOLD, iou=IOU_NMS,
            imgsz=BYTETRACK_IMGSZ, tracker="bytetrack.yaml",
            persist=True, verbose=False
        )
        total_detections += len(track_results[0].boxes)

        for box in track_results[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            tracking_data[frame_name].append({
                "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                "conf": round(conf, 3),
                "track_id": track_id
            })

    t_total = time.time() - t_start
    avg_fps = len(image_files) / t_total if t_total > 0 else 0

    with open(json_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)

    stats = {
        "sequence": seq_name,
        "total_frames": len(image_files),
        "total_detections": total_detections,
        "total_time_sec": round(t_total, 2),
        "avg_fps": round(avg_fps, 2),
        "resolution": f"{img_w}x{img_h}",
    }

    print(f"  [OK] {seq_name} done in {t_total:.1f}s ({avg_fps:.1f} FPS) | "
          f"{total_detections} detections → {json_path}")
    return json_path, stats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SAM2 Data Generator: ByteTrack Only")
    parser.add_argument("--gcs_bucket", type=str, required=True)
    parser.add_argument("--gcs_model_path", type=str, required=True)
    parser.add_argument("--gcs_sequence_dir", type=str, default="",
                        help="Single sequence dir for POC")
    parser.add_argument("--gcs_train_prefix", type=str, default="",
                        help="Prefix for all train images (flat dir)")
    parser.add_argument("--gcs_output_base", type=str,
                        default="data_generation/sam2_inputs")
    parser.add_argument("--mode", type=str, choices=["poc", "full"], default="poc")
    args = parser.parse_args()

    print("=" * 60)
    print(" SAM2 DATA GENERATOR (ByteTrack Only)")
    print(f" Mode: {args.mode.upper()}")
    print(f" Bucket: {args.gcs_bucket}")
    print(f" Output: gs://{args.gcs_bucket}/{args.gcs_output_base}/")
    print("=" * 60)

    # 1. Download Model
    model_dir = "/tmp/ball_detection/models"
    os.makedirs(model_dir, exist_ok=True)
    local_model_path = os.path.join(model_dir, "best.pt")
    download_blob(args.gcs_bucket, args.gcs_model_path, local_model_path)

    # 2. Load Model
    model = YOLO(local_model_path)
    print(f"[OK] Model loaded: {local_model_path}")

    # GPU check
    import torch
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("[WARN] No GPU detected — running on CPU (will be slow)")

    t_global_start = time.time()
    all_stats = []
    base_dir = "/tmp/ball_detection"

    if args.mode == "poc":
        # POC: Single sequence
        if not args.gcs_sequence_dir:
            print("[ERROR] POC mode requires --gcs_sequence_dir")
            return

        seq_name = args.gcs_sequence_dir.strip('/').split('/')[-1]
        data_dir = os.path.join(base_dir, "data", seq_name)
        out_dir = os.path.join(base_dir, "output", seq_name)

        download_blob(args.gcs_bucket, args.gcs_sequence_dir, data_dir)

        json_path, stats = run_bytetrack(model, data_dir, seq_name, out_dir)
        if json_path:
            upload_blob(args.gcs_bucket, json_path,
                       f"{args.gcs_output_base}/bytetrack/{seq_name}/coordinates.json")
            all_stats.append(stats)

    else:
        # Full: Download all train images, group by sequence
        if not args.gcs_train_prefix:
            print("[ERROR] Full mode requires --gcs_train_prefix")
            return

        data_dir = os.path.join(base_dir, "data", "train")
        out_dir = os.path.join(base_dir, "output")

        print(f"\n[FULL] Downloading all train images...")
        seq_dirs = download_train_images(
            args.gcs_bucket, args.gcs_train_prefix, data_dir
        )

        print(f"\n[FULL] Processing {len(seq_dirs)} sequences...")
        for i, (seq_name, seq_path) in enumerate(sorted(seq_dirs.items())):
            print(f"\n{'#'*60}")
            print(f" SEQUENCE {i+1}/{len(seq_dirs)}: {seq_name}")
            print(f"{'#'*60}")

            try:
                seq_out = os.path.join(out_dir, seq_name)
                json_path, stats = run_bytetrack(model, seq_path, seq_name, seq_out)

                if json_path:
                    upload_blob(args.gcs_bucket, json_path,
                               f"{args.gcs_output_base}/bytetrack/{seq_name}/coordinates.json")
                    all_stats.append(stats)

                # Free disk after each sequence
                shutil.rmtree(seq_path, ignore_errors=True)
                shutil.rmtree(seq_out, ignore_errors=True)

            except Exception as e:
                print(f"[ERROR] Failed on {seq_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Global summary
    t_global = time.time() - t_global_start
    total_frames = sum(s["total_frames"] for s in all_stats)
    total_dets = sum(s["total_detections"] for s in all_stats)

    summary = {
        "mode": args.mode,
        "pipeline": "bytetrack_only",
        "total_time_sec": round(t_global, 2),
        "total_time_min": round(t_global / 60, 2),
        "sequences_processed": len(all_stats),
        "total_frames": total_frames,
        "total_detections": total_dets,
        "overall_fps": round(total_frames / t_global, 2) if t_global > 0 else 0,
        "per_sequence": all_stats
    }

    summary_path = os.path.join(base_dir, "global_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    upload_blob(args.gcs_bucket, summary_path,
                f"{args.gcs_output_base}/global_summary.json")

    print(f"\n{'='*60}")
    print(f" JOB COMPLETE")
    print(f" Total time: {t_global/60:.1f} min")
    print(f" Sequences: {len(all_stats)}")
    print(f" Frames: {total_frames} | Detections: {total_dets}")
    print(f" Results: gs://{args.gcs_bucket}/{args.gcs_output_base}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
