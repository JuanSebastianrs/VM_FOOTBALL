"""
Build Legibility Dataset (Pre-Sampled Ultra-Optimized Version).
Processes SoccerNet Jersey 2023 and jersey_tracking_v1 to extract:
  - Class 1 (Visible): High-quality crops where the baseline reader matches the GT jersey number.
  - Class 0 (Not Visible): SoccerNet Jersey 2023 with jersey == -1, plus low-quality crops from tracking.

Strictly excludes SNMOT-148 and all 'test' splits from both train/val/test splits to avoid any data leakage.
"""

import os
import sys
import json
import random
import argparse
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.identity.jersey_model import load_jersey_model, compute_jersey_probs_from_logits, TRANSFORM_INFERENCE
from core.identity.jersey_identity_phase import compute_crop_quality


def resolve_tracking_split(sequence, splits_data):
    """Resolve split for tracking sequence using splits.json from tracking dataset."""
    train_seqs = {s for s, v in splits_data.get("train", {}).items() if v}
    val_seqs = {s for s, v in splits_data.get("val", {}).items() if v}
    test_seqs = {s for s, v in splits_data.get("test", {}).items() if v}
    
    if sequence in test_seqs:
        return "test"
    if sequence in val_seqs and (int(hashlib.md5(sequence.encode()).hexdigest(), 16) % 100) < 15:
        return "val"
    if sequence in train_seqs:
        return "train"
    return "unknown"


def predict_batch(model, crops, device):
    """
    Run baseline model on a list of crops.
    crops: list of (img, qual, path)
    Returns: list of (pred_num, conf)
    """
    if not crops:
        return []
    tensors = []
    for img, qual, path in crops:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors.append(TRANSFORM_INFERENCE(img_rgb))
    
    x = torch.stack(tensors).unsqueeze(1).to(device)
    with torch.no_grad():
        out_len, out_tens, out_ones = model(x)
    
    probs = compute_jersey_probs_from_logits(out_len, out_tens, out_ones)
    if probs.ndim == 1:
        probs = np.expand_dims(probs, axis=0) # shape (N, 99)
        
    results = []
    for idx in range(len(crops)):
        pred_num = int(np.argmax(probs[idx])) + 1
        conf = float(probs[idx, pred_num - 1])
        results.append((pred_num, conf))
    return results


def main():
    parser = argparse.ArgumentParser(description="Build Legibility Dataset")
    parser.add_argument("--tracking_dir", type=str, default="datasets/jersey_tracking_v1")
    parser.add_argument("--soccernet_dir", type=str, default="datasets/soccernet/jersey-2023")
    parser.add_argument("--model_path", type=str, default="runs/jersey_digit_mil_v2/best.pt")
    parser.add_argument("--output_dir", type=str, default="datasets/jersey_legibility_v1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading baseline model from {args.model_path}...")
    model = load_jersey_model(args.model_path, device, strict=True)
    model.eval()

    output_path = Path(args.output_dir)
    visible_dir = output_path / "images" / "visible"
    not_visible_dir = output_path / "images" / "not_visible"

    visible_dir.mkdir(parents=True, exist_ok=True)
    not_visible_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # -------------------------------------------------------------------------
    # 1. PROCESS SOCCERNET JERSEY 2023 (EXCLUDE SoccerNet 'test' split entirely)
    # -------------------------------------------------------------------------
    print("\n--- Processing SoccerNet Jersey 2023 (train split only) ---")
    sn_dir = Path(args.soccernet_dir)
    
    # Strictly exclude 'test' split of SoccerNet Jersey 2023
    for split in ["train"]:
        gt_path = sn_dir / split / f"{split}_gt.json"
        if not gt_path.exists():
            gt_path = sn_dir / f"{split}_gt.json"
        if not gt_path.exists():
            print(f"Skipping SoccerNet split {split} because GT is missing.")
            continue

        with open(gt_path) as f:
            gt_data = json.load(f)

        images_dir = sn_dir / split / "images"
        if not images_dir.exists():
            images_dir = sn_dir / split / split / "images"

        if not images_dir.exists():
            print(f"Skipping SoccerNet split {split} because images dir is missing.")
            continue

        player_dirs = sorted(list(images_dir.iterdir()))
        print(f"Found {len(player_dirs)} player tracklets in SoccerNet {split}")

        for pdir in tqdm(player_dirs, desc=f"SoccerNet {split}"):
            if not pdir.is_dir():
                continue
            player_id = pdir.name
            gt_jersey = gt_data.get(player_id, -1)

            image_files = sorted(list(pdir.glob("*.jpg")))
            if not image_files:
                continue
            
            # Determine split by player_id hash (15% validation, 85% train)
            h = int(hashlib.md5(f"sn_{player_id}".encode()).hexdigest(), 16)
            source_split = "val" if (h % 100) < 15 else "train"

            if gt_jersey == -1:
                # True Negatives! No number is visible on this tracklet.
                # Take up to 5 random crops with quality >= 0.1
                sampled_files = random.sample(image_files, min(len(image_files), 10))
                crops_to_process = []
                for fp in sampled_files:
                    img = cv2.imread(str(fp))
                    if img is None:
                        continue
                    qual = compute_crop_quality(img)
                    if qual >= 0.1:
                        crops_to_process.append((img, qual, fp))
                
                # Sample at most 5 valid ones
                crops_to_process = crops_to_process[:5]
                if crops_to_process:
                    preds = predict_batch(model, crops_to_process, device)
                    for idx, (img, qual, fp) in enumerate(crops_to_process):
                        pred_num, conf = preds[idx]
                        out_name = f"sn_{split}_{player_id}_{fp.name}"
                        cv2.imwrite(str(not_visible_dir / out_name), img)
                        
                        try:
                            frame_id = int(fp.stem)
                        except ValueError:
                            frame_id = fp.stem
                            
                        records.append({
                            "filename": f"images/not_visible/{out_name}",
                            "label": 0,
                            "source": f"soccernet_{split}",
                            "source_split": source_split,
                            "sequence": "soccernet",
                            "track_id": player_id,
                            "frame_id": frame_id,
                            "gt_jersey": -1,
                            "quality": qual,
                            "reader_pred": pred_num,
                            "reader_conf": conf,
                            "crop_variant": "torso",
                            "source_path": str(fp)
                        })
            else:
                # Potential Positives or Weak Negatives or Confusos
                sampled_files = random.sample(image_files, min(len(image_files), 10))
                crops_to_process = []
                for fp in sampled_files:
                    img = cv2.imread(str(fp))
                    if img is None:
                        continue
                    qual = compute_crop_quality(img)
                    if qual >= 0.05: # Keep all for routing
                        crops_to_process.append((img, qual, fp))
                
                crops_to_process = crops_to_process[:6]
                if crops_to_process:
                    preds = predict_batch(model, crops_to_process, device)
                    for idx, (img, qual, fp) in enumerate(crops_to_process):
                        pred_num, conf = preds[idx]
                        
                        # Apply clear labeling rules
                        label = None
                        target_dir = None
                        
                        # 1. Positivos (label=1)
                        if qual >= 0.35 and pred_num == gt_jersey and conf >= 0.70:
                            label = 1
                            target_dir = visible_dir
                        # 2. Negativos Débiles (label=0)
                        elif qual < 0.15:
                            label = 0
                            target_dir = not_visible_dir
                        # 3. Excluir Confusos
                        elif pred_num != gt_jersey and conf >= 0.50:
                            # Exclude (label remains None)
                            pass
                        
                        if label is not None:
                            out_name = f"sn_{split}_{player_id}_{fp.name}"
                            cv2.imwrite(str(target_dir / out_name), img)
                            
                            try:
                                frame_id = int(fp.stem)
                            except ValueError:
                                frame_id = fp.stem
                                
                            records.append({
                                "filename": f"images/{'visible' if label == 1 else 'not_visible'}/{out_name}",
                                "label": label,
                                "source": f"soccernet_{split}",
                                "source_split": source_split,
                                "sequence": "soccernet",
                                "track_id": player_id,
                                "frame_id": frame_id,
                                "gt_jersey": gt_jersey,
                                "quality": qual,
                                "reader_pred": pred_num,
                                "reader_conf": conf,
                                "crop_variant": "torso",
                                "source_path": str(fp)
                            })

    # -------------------------------------------------------------------------
    # 2. PROCESS JERSEY TRACKING V1 (exclude SNMOT-148 & tracking test split)
    # -------------------------------------------------------------------------
    print("\n--- Processing Jersey Tracking V1 ---")
    tracking_dir = Path(args.tracking_dir)
    tracklets_json_path = tracking_dir / "tracklets.json"
    splits_json_path = tracking_dir / "splits.json"
    
    if not splits_json_path.exists():
        raise FileNotFoundError(f"Missing splits.json in {tracking_dir}")
    
    with open(splits_json_path) as f:
        splits_data = json.load(f)

    if tracklets_json_path.exists():
        with open(tracklets_json_path) as f:
            tracklets_data = json.load(f)

        print(f"Loaded {len(tracklets_data)} tracklets from tracking dataset.")

        for tracklet in tqdm(tracklets_data, desc="Jersey Tracking V1"):
            seq = tracklet.get("sequence", "")
            if seq == "SNMOT-148":
                # Strict holdout exclusion
                continue

            # Resolve tracking split
            tr_split = resolve_tracking_split(seq, splits_data)
            if tr_split not in ("train", "val"):
                # Strict exclusion of test split and unknown sequences
                continue

            gt_jersey = int(tracklet.get("jersey_number", -1))
            if gt_jersey < 1:
                continue

            crop_paths = tracklet.get("crop_paths", [])
            if not crop_paths:
                continue

            # Pre-sample crops
            sampled_crops = random.sample(crop_paths, min(len(crop_paths), 12))
            crops_to_process = []
            for cpath in sampled_crops:
                full_cpath = tracking_dir / cpath
                if not full_cpath.exists():
                    continue
                img = cv2.imread(str(full_cpath))
                if img is None:
                    continue
                qual = compute_crop_quality(img)
                crops_to_process.append((img, qual, full_cpath))

            crops_to_process = crops_to_process[:8]
            if crops_to_process:
                preds = predict_batch(model, crops_to_process, device)
                for idx, (img, qual, full_cpath) in enumerate(crops_to_process):
                    pred_num, conf = preds[idx]
                    
                    label = None
                    target_dir = None
                    
                    # 1. Positivos (label=1)
                    if qual >= 0.35 and pred_num == gt_jersey and conf >= 0.70:
                        label = 1
                        target_dir = visible_dir
                    # 2. Negativos Débiles (label=0)
                    elif qual < 0.15:
                        label = 0
                        target_dir = not_visible_dir
                    # 3. Excluir Confusos
                    elif pred_num != gt_jersey and conf >= 0.50:
                        pass
                    
                    if label is not None:
                        out_name = f"tr_{seq}_{tracklet['track_id']}_{full_cpath.name}"
                        cv2.imwrite(str(target_dir / out_name), img)
                        
                        try:
                            frame_id = int(full_cpath.stem.split("_")[-1])
                        except (ValueError, IndexError):
                            frame_id = full_cpath.stem
                            
                        records.append({
                            "filename": f"images/{'visible' if label == 1 else 'not_visible'}/{out_name}",
                            "label": label,
                            "source": "tracking_v1",
                            "source_split": tr_split,
                            "sequence": seq,
                            "track_id": str(tracklet['track_id']),
                            "frame_id": frame_id,
                            "gt_jersey": gt_jersey,
                            "quality": qual,
                            "reader_pred": pred_num,
                            "reader_conf": conf,
                            "crop_variant": "torso",
                            "source_path": str(full_cpath)
                        })

    # Strict Validation Check: at least 1 tracking sample must be generated
    tracking_samples = [r for r in records if r["source"] == "tracking_v1"]
    print(f"\nGenerated {len(tracking_samples)} crops from real tracking dataset.")
    if len(tracking_samples) == 0:
        raise ValueError("Assertion Error: Zero tracking samples were generated! The dataset is incomplete.")

    # Save metadata.csv
    df = pd.DataFrame(records)
    df.to_csv(output_path / "metadata.csv", index=False)
    print(f"\nMetadata saved: {output_path / 'metadata.csv'}")
    print(f"Total samples: {len(df)}")
    print(f"  Visible (1): {sum(df['label'] == 1)}")
    print(f"  Not Visible (0): {sum(df['label'] == 0)}")

    # Create splits.json strictly aligned with assigned source_split
    train_files = []
    val_files = []

    for rec in records:
        filename = rec["filename"]
        if rec["source_split"] == "train":
            train_files.append(filename)
        elif rec["source_split"] == "val":
            val_files.append(filename)

    splits = {
        "train": train_files,
        "val": val_files
    }

    with open(output_path / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved: {output_path / 'splits.json'}")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val: {len(val_files)} samples")


if __name__ == "__main__":
    main()
