"""
SAM2 Pseudo-Mask Generator (Vertex AI)

Uses pre-trained SAM2 (zero-shot) + ByteTrack bounding boxes to generate
binary segmentation masks for each detection. These masks become the
ground truth for SAM2 fine-tuning.

Input:
  - Original frames from GCS
  - ByteTrack coordinate JSONs from data_generation pipeline

Output per sequence:
  gs://bucket/data_generation/sam2_training/SNMOT-XXX/
  ├── images/SNMOT-XXX_000001.jpg        (symlink/copy)
  ├── masks/SNMOT-XXX_000001.png         (binary mask, ball=255)
  └── prompts.json                        (frame → bbox mapping)
"""

import os
import re
import cv2
import json
import time
import argparse
import shutil
import numpy as np
from collections import defaultdict
from google.cloud import storage


def download_blob(bucket_name, prefix, local_dir):
    """Download all files under a GCS prefix."""
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    os.makedirs(local_dir, exist_ok=True)
    count = 0
    for b in blobs:
        if b.name.endswith('/'):
            continue
        rel = b.name[len(prefix):].lstrip('/')
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        b.download_to_filename(dest)
        count += 1
    print(f"[GCS] Downloaded {count} files from gs://{bucket_name}/{prefix}")
    return count


def upload_folder(bucket_name, local_dir, gcs_prefix):
    """Upload entire folder to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    count = 0
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, local_dir).replace("\\", "/")
            blob = bucket.blob(f"{gcs_prefix}/{rel}")
            blob.upload_from_filename(local_path)
            count += 1
    print(f"[GCS] Uploaded {count} files → gs://{bucket_name}/{gcs_prefix}/")


def upload_blob(bucket_name, local_path, gcs_path):
    """Upload single file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)


def list_sequences(bucket_name, prefix):
    """List all sequence subdirectories under a prefix."""
    client = storage.Client()
    # Use delimiter to get "directories"
    iterator = client.list_blobs(bucket_name, prefix=prefix, delimiter='/')
    # Need to consume the iterator to get prefixes
    list(iterator)  # consume
    return sorted([p.rstrip('/').split('/')[-1] for p in iterator.prefixes])


def generate_masks_for_sequence(
    sam2_predictor, seq_name, frames_dir, json_path, output_dir,
    min_mask_area=10, min_conf=0.20
):
    """Generate binary masks for one sequence using SAM2 zero-shot.

    Args:
        sam2_predictor: SAM2ImagePredictor instance
        seq_name: Sequence name (e.g., SNMOT-060)
        frames_dir: Directory containing the original .jpg frames
        json_path: Path to ByteTrack coordinates.json
        output_dir: Where to save masks and prompts

    Returns:
        stats dict with counts
    """
    with open(json_path, 'r') as f:
        coords = json.load(f)

    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    prompts = {}
    total_masks = 0
    skipped_low_conf = 0
    skipped_small_mask = 0
    skipped_no_detection = 0
    failed = 0

    frame_names = sorted(coords.keys())
    t_start = time.time()

    for idx, frame_name in enumerate(frame_names):
        if idx % 100 == 0:
            elapsed = time.time() - t_start
            fps = idx / elapsed if elapsed > 0 and idx > 0 else 0
            eta = (len(frame_names) - idx) / fps if fps > 0 else 0
            print(f"  [{seq_name}] Frame {idx}/{len(frame_names)} | "
                  f"{fps:.1f} FPS | ETA: {eta:.0f}s")

        detections = coords[frame_name]
        if not detections:
            skipped_no_detection += 1
            continue

        # Take the highest-confidence detection only (1 ball per frame)
        best_det = max(detections, key=lambda d: d["conf"])
        if best_det["conf"] < min_conf:
            skipped_low_conf += 1
            continue

        bbox = best_det["bbox"]  # [x1, y1, x2, y2]

        # Load frame
        frame_path = os.path.join(frames_dir, frame_name)
        if not os.path.exists(frame_path):
            failed += 1
            continue

        try:
            image = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # SAM2 prediction with box prompt
            sam2_predictor.set_image(image_rgb)

            input_box = np.array(bbox)  # [x1, y1, x2, y2]
            masks, scores, logits = sam2_predictor.predict(
                box=input_box[None, :],  # (1, 4)
                multimask_output=True
            )

            # Pick the mask with highest IoU score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]  # (H, W) boolean

            # Validate mask quality
            mask_area = mask.sum()
            if mask_area < min_mask_area:
                skipped_small_mask += 1
                continue

            # Save binary mask as PNG (255 = ball, 0 = background)
            mask_uint8 = (mask.astype(np.uint8)) * 255
            mask_path = os.path.join(mask_dir, frame_name.replace('.jpg', '.png'))
            cv2.imwrite(mask_path, mask_uint8)

            # Copy frame to images dir
            dst_img = os.path.join(img_dir, frame_name)
            if not os.path.exists(dst_img):
                shutil.copy2(frame_path, dst_img)

            # Save prompt for this frame
            prompts[frame_name] = {
                "bbox": bbox,
                "conf": best_det["conf"],
                "mask_area": int(mask_area),
                "sam2_iou_score": float(scores[best_mask_idx])
            }
            total_masks += 1

        except Exception as e:
            print(f"  [WARN] Failed on {frame_name}: {e}")
            failed += 1
            continue

    # Save prompts.json
    prompts_path = os.path.join(output_dir, "prompts.json")
    with open(prompts_path, 'w') as f:
        json.dump(prompts, f, indent=2)

    t_total = time.time() - t_start
    stats = {
        "sequence": seq_name,
        "total_frames": len(frame_names),
        "masks_generated": total_masks,
        "skipped_no_detection": skipped_no_detection,
        "skipped_low_conf": skipped_low_conf,
        "skipped_small_mask": skipped_small_mask,
        "failed": failed,
        "time_sec": round(t_total, 2),
        "fps": round(len(frame_names) / t_total, 2) if t_total > 0 else 0
    }

    print(f"  [OK] {seq_name}: {total_masks} masks in {t_total:.1f}s "
          f"(skip: {skipped_no_detection} no-det, {skipped_low_conf} low-conf, "
          f"{skipped_small_mask} small, {failed} failed)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="SAM2 Pseudo-Mask Generator")
    parser.add_argument("--gcs_bucket", type=str, required=True)
    parser.add_argument("--gcs_frames_prefix", type=str, required=True,
                        help="GCS prefix for original frames, e.g. reorganized_dataset/images/train/")
    parser.add_argument("--gcs_coords_prefix", type=str, required=True,
                        help="GCS prefix for ByteTrack JSONs, e.g. data_generation/sam2_inputs/bytetrack/")
    parser.add_argument("--gcs_output_prefix", type=str,
                        default="data_generation/sam2_training")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="facebook/sam2.1-hiera-small")
    parser.add_argument("--min_conf", type=float, default=0.20,
                        help="Min ByteTrack conf to generate mask")
    args = parser.parse_args()

    print("=" * 60)
    print(" SAM2 PSEUDO-MASK GENERATOR")
    print(f" Checkpoint: {args.sam2_checkpoint}")
    print(f" Min conf: {args.min_conf}")
    print(f" Bucket: {args.gcs_bucket}")
    print("=" * 60)

    # ---- GPU Check ----
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("[WARN] No GPU — will be very slow")

    # ---- Load SAM2 ----
    print("\n[1/3] Loading SAM2 model...")
    import torch

    # Map checkpoint arg to download URL and config
    CHECKPOINT_MAP = {
        "facebook/sam2.1-hiera-small": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        },
        "facebook/sam2.1-hiera-tiny": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        },
        "facebook/sam2.1-hiera-base-plus": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        },
        "facebook/sam2.1-hiera-large": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        },
    }

    predictor = None

    # Attempt 1: HuggingFace from_pretrained
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor.from_pretrained(args.sam2_checkpoint)
        print(f"[OK] SAM2 loaded via from_pretrained: {args.sam2_checkpoint}")
    except Exception as e1:
        print(f"[WARN] from_pretrained failed: {e1}")

    # Attempt 2: Download checkpoint + build_sam2
    if predictor is None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            ckpt_info = CHECKPOINT_MAP.get(args.sam2_checkpoint)
            if ckpt_info is None:
                raise ValueError(f"Unknown checkpoint: {args.sam2_checkpoint}")

            print(f"  Downloading checkpoint from FB CDN...")
            ckpt_path = "/tmp/sam2_ckpt.pt"
            torch.hub.download_url_to_file(ckpt_info["url"], ckpt_path)

            # The sam2 repo was cloned to /tmp/sam2 by startup script
            sam2_repo = "/tmp/sam2"
            config_path = ckpt_info["config"]

            print(f"  Building model from config: {config_path}")
            sam2_model = build_sam2(
                config_path, ckpt_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            predictor = SAM2ImagePredictor(sam2_model)
            print(f"[OK] SAM2 loaded via build_sam2 + checkpoint")
        except Exception as e2:
            print(f"[FATAL] Cannot load SAM2: {e2}")
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)

    # ---- Discover sequences ----
    print("\n[2/3] Discovering sequences...")
    sequences = list_sequences(args.gcs_bucket, args.gcs_coords_prefix)
    print(f"[OK] Found {len(sequences)} sequences with ByteTrack coordinates")

    # ---- Process each sequence ----
    print(f"\n[3/3] Generating pseudo-masks for {len(sequences)} sequences...")
    base_dir = "/tmp/sam2_masks"
    all_stats = []
    t_global = time.time()

    for i, seq_name in enumerate(sequences):
        print(f"\n{'#'*60}")
        print(f" SEQUENCE {i+1}/{len(sequences)}: {seq_name}")
        print(f"{'#'*60}")

        local_frames = os.path.join(base_dir, "frames", seq_name)
        local_coords = os.path.join(base_dir, "coords", seq_name)
        local_output = os.path.join(base_dir, "output", seq_name)

        try:
            # Download original frames for this sequence
            # Frames are flat: reorganized_dataset/images/train/SNMOT-060_*.jpg
            # We need to download only frames matching this sequence prefix
            client = storage.Client()
            blobs = list(client.list_blobs(
                args.gcs_bucket,
                prefix=f"{args.gcs_frames_prefix}{seq_name}_"
            ))
            os.makedirs(local_frames, exist_ok=True)
            for b in blobs:
                fname = os.path.basename(b.name)
                b.download_to_filename(os.path.join(local_frames, fname))
            print(f"  Downloaded {len(blobs)} frames")

            # Download coordinates JSON
            coords_gcs = f"{args.gcs_coords_prefix}{seq_name}/coordinates.json"
            coords_local = os.path.join(local_coords, "coordinates.json")
            os.makedirs(local_coords, exist_ok=True)
            client.bucket(args.gcs_bucket).blob(coords_gcs).download_to_filename(coords_local)

            # Generate masks
            stats = generate_masks_for_sequence(
                predictor, seq_name, local_frames, coords_local, local_output,
                min_conf=args.min_conf
            )
            all_stats.append(stats)

            # Upload results
            upload_folder(
                args.gcs_bucket, local_output,
                f"{args.gcs_output_prefix}/{seq_name}"
            )

            # Cleanup disk
            shutil.rmtree(local_frames, ignore_errors=True)
            shutil.rmtree(local_coords, ignore_errors=True)
            shutil.rmtree(local_output, ignore_errors=True)

        except Exception as e:
            print(f"[ERROR] {seq_name}: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({"sequence": seq_name, "error": str(e)})
            # Cleanup on error too
            shutil.rmtree(os.path.join(base_dir, "frames", seq_name), ignore_errors=True)
            continue

    # ---- Global summary ----
    t_total = time.time() - t_global
    total_masks = sum(s.get("masks_generated", 0) for s in all_stats)
    total_frames = sum(s.get("total_frames", 0) for s in all_stats)
    errors = sum(1 for s in all_stats if "error" in s)

    summary = {
        "total_time_min": round(t_total / 60, 2),
        "sequences_processed": len(all_stats),
        "sequences_failed": errors,
        "total_frames": total_frames,
        "total_masks_generated": total_masks,
        "sam2_checkpoint": args.sam2_checkpoint,
        "min_conf_threshold": args.min_conf,
        "per_sequence": all_stats
    }

    summary_path = os.path.join(base_dir, "mask_generation_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    upload_blob(args.gcs_bucket, summary_path,
                f"{args.gcs_output_prefix}/mask_generation_summary.json")

    print(f"\n{'='*60}")
    print(f" PSEUDO-MASK GENERATION COMPLETE")
    print(f" Time: {t_total/60:.1f} min")
    print(f" Sequences: {len(all_stats)} ({errors} errors)")
    print(f" Masks: {total_masks} / {total_frames} frames")
    print(f" Output: gs://{args.gcs_bucket}/{args.gcs_output_prefix}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
