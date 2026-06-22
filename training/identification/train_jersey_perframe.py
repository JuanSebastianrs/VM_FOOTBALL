"""
Per-frame jersey digit training with legibility filtering.

Motivation (v1.5): the production pipeline runs the digit model PER FRAME
(temporal mode, bags of K=1) but jersey_digit_mil trains on MIL bags of K=16.
This script closes that train/inference mismatch:

  1. Each legible crop is an independent training sample (bag of K=1), so the
     checkpoint stays 100% compatible with DigitCompositionalMIL inference.
  2. Crops are filtered by the trained LegibilityClassifier scores
     (scripts/score_crop_legibility.py) — frames that do not show a readable
     digit are dropped instead of polluting training with tracklet label noise.
  3. The tens-head loss is masked for 1-digit numbers (the tens digit is
     undefined there; the legacy recipe forced target 0, biasing the head).
  4. Label smoothing + AdamW + warmup-cosine + strong per-frame augmentation.

Model selection: best.pt is chosen by tracklet-level fused accuracy on val
(geometric fusion over legible frames — same as production temporal mode),
with per-frame accuracy as tiebreaker.

Usage:
    python training/identification/train_jersey_perframe.py \
        --dataset_dir datasets/jersey_tracking_v1 \
        --output_dir runs/jersey_perframe_v1 \
        --init_from runs/jersey_digit_mil_v2/best.pt \
        --min_legibility 0.5 \
        --epochs 30 --batch_size 64 --lr 2e-4 --device cuda:0
"""

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.identity.jersey_model import (
    DigitCompositionalMIL,
    build_transform_train_perframe,
    build_transform_inference,
    compute_jersey_probs_from_logits,
    save_full_checkpoint,
)
from training.identification.train_jersey_digit_mil import build_splits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jersey_to_targets(jersey):
    """Decompose a jersey number into (length, tens, ones) targets."""
    if jersey <= 9:
        return 0, 0, jersey
    return 1, jersey // 10, jersey % 10


def build_frame_samples(records, legibility_scores, split, min_legibility):
    """
    Flatten tracklets into per-crop samples, keeping only legible crops.

    Returns list of dicts: {crop_path, jersey, legibility, tracklet_key}.
    """
    samples = []
    skipped_tracklets = 0
    for rec in records:
        if rec.get("split") != split:
            continue
        jersey = int(rec.get("jersey_number", -1))
        if jersey < 1 or jersey > 99:
            continue
        key = (rec["sequence"], int(rec["track_id"]))
        kept = 0
        for cp in rec.get("crop_paths", []):
            rel = Path(cp).as_posix()
            leg = legibility_scores.get(rel)
            if leg is None or leg < min_legibility:
                continue
            samples.append({
                "crop_path": rel,
                "jersey": jersey,
                "legibility": leg,
                "tracklet_key": key,
            })
            kept += 1
        if kept == 0:
            skipped_tracklets += 1
    return samples, skipped_tracklets


class JerseyFrameDataset(Dataset):
    """Per-crop dataset; returns bags of K=1 to stay MIL-checkpoint compatible.

    Samples may carry an optional "root" (overrides root_dir, used for mixed
    external data) and "torso_crop": True (apply standard torso fractions to a
    tight player crop, e.g. SoccerNet Jersey images).
    """

    def __init__(self, samples, root_dir, transform):
        self.samples = samples
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        root = Path(s["root"]) if "root" in s else self.root_dir
        img_path = root / s["crop_path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Missing crop: {img_path}")
        img = np.array(Image.open(img_path).convert("RGB"))
        if s.get("torso_crop"):
            h, w = img.shape[:2]
            img = img[int(h * 0.10):int(h * 0.70), int(w * 0.10):int(w * 0.90)]
        x = self.transform(img).unsqueeze(0)  # (1, 3, H, W) — bag of K=1
        length, tens, ones = jersey_to_targets(s["jersey"])
        return x, torch.tensor(length), torch.tensor(tens), torch.tensor(ones)


def make_balanced_sampler(samples):
    """Inverse-frequency sampling over jersey numbers (sqrt-damped)."""
    counts = defaultdict(int)
    for s in samples:
        counts[s["jersey"]] += 1
    weights = [1.0 / math.sqrt(counts[s["jersey"]]) for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def train_epoch(model, loader, optimizer, scheduler, device, label_smoothing, mask_tens):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ce_none = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")
    total_loss, n_batches = 0.0, 0
    for x, len_t, tens_t, ones_t in loader:
        x = x.to(device)
        len_t, tens_t, ones_t = len_t.to(device), tens_t.to(device), ones_t.to(device)
        optimizer.zero_grad()
        out_len, out_tens, out_ones = model(x)
        loss = ce(out_len, len_t) + ce(out_ones, ones_t)
        if mask_tens:
            # Tens digit is only defined for 2-digit numbers
            two_digit = (len_t == 1).float()
            tens_loss = (ce_none(out_tens, tens_t) * two_digit).sum() / two_digit.sum().clamp(min=1.0)
            loss = loss + tens_loss
        else:
            loss = loss + ce(out_tens, tens_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_perframe(model, loader, device):
    """Per-frame jersey accuracy on legible validation crops."""
    model.eval()
    correct, total = 0, 0
    for x, len_t, tens_t, ones_t in loader:
        x = x.to(device)
        out_len, out_tens, out_ones = model(x)
        probs = compute_jersey_probs_from_logits(out_len, out_tens, out_ones)
        probs = np.atleast_2d(probs)
        pred = probs.argmax(axis=1) + 1
        gt_jersey = torch.where(len_t == 0, ones_t, tens_t * 10 + ones_t).numpy()
        correct += int((pred == gt_jersey).sum())
        total += len(gt_jersey)
    return correct / max(total, 1)


@torch.no_grad()
def validate_tracklets(model, samples, root_dir, device, batch_size=128, transform=None):
    """
    Tracklet-level fused accuracy on val (production-style):
    per-frame probs over legible crops -> legibility-weighted geometric fusion.
    """
    model.eval()
    if transform is None:
        transform = build_transform_inference(128)
    by_tracklet = defaultdict(list)
    for s in samples:
        by_tracklet[s["tracklet_key"]].append(s)

    correct_geo, correct_ari, total = 0, 0, 0
    for key, t_samples in by_tracklet.items():
        tensors, legs = [], []
        for s in t_samples:
            img = np.array(Image.open(Path(root_dir) / s["crop_path"]).convert("RGB"))
            tensors.append(transform(img))
            legs.append(s["legibility"])
        probs_list = []
        for i in range(0, len(tensors), batch_size):
            x = torch.stack(tensors[i:i + batch_size]).unsqueeze(1).to(device)  # (B,1,3,H,W)
            out_len, out_tens, out_ones = model(x)
            p = np.atleast_2d(compute_jersey_probs_from_logits(out_len, out_tens, out_ones))
            probs_list.append(p)
        probs = np.concatenate(probs_list, axis=0)  # (T, 99)
        w = np.array(legs, dtype=np.float64)
        w = w / max(w.sum(), 1e-8)

        gt = t_samples[0]["jersey"]
        fused_geo = (w[:, None] * np.log(probs + 1e-10)).sum(axis=0)
        if int(np.argmax(fused_geo)) + 1 == gt:
            correct_geo += 1
        fused_ari = (w[:, None] * probs).sum(axis=0)
        if int(np.argmax(fused_ari)) + 1 == gt:
            correct_ari += 1
        total += 1

    return {
        "tracklet_geo_acc": correct_geo / max(total, 1),
        "tracklet_ari_acc": correct_ari / max(total, 1),
        "tracklets": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Per-frame jersey digit training with legibility filtering")
    parser.add_argument("--dataset_dir", type=str, default="datasets/jersey_tracking_v1")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--legibility_scores", type=str, default=None,
                        help="Default: <dataset_dir>/legibility_scores.json")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Warm-start from an existing DigitCompositionalMIL checkpoint")
    parser.add_argument("--soccernet_index", type=str, default=None,
                        help="Per-frame index JSON from scripts/build_soccernet_perframe_index.py "
                             "(adds SoccerNet Jersey 2023 train crops to training)")
    parser.add_argument("--soccernet_root", type=str, default="datasets/soccernet/jersey-2023")
    parser.add_argument("--soccernet_frac", type=float, default=None,
                        help="Expected fraction of SoccerNet samples per epoch via weighted "
                             "sampling (e.g. 0.4). Default: natural concatenation.")
    parser.add_argument("--min_legibility", type=float, default=0.5)
    parser.add_argument("--img_size", type=int, default=128,
                        help="Model input size; stored in the checkpoint so inference "
                             "auto-selects the matching transform")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--no_mask_tens", action="store_true",
                        help="Disable tens-loss masking for 1-digit numbers (legacy behavior)")
    parser.add_argument("--balanced_sampling", action="store_true",
                        help="Inverse-sqrt-frequency sampling over jersey numbers")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(output_dir / "train.log"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    dataset_dir = Path(args.dataset_dir)
    leg_path = Path(args.legibility_scores) if args.legibility_scores else dataset_dir / "legibility_scores.json"
    with open(leg_path) as f:
        legibility_scores = json.load(f)

    records = build_splits(dataset_dir / "tracklets.json", dataset_dir / "splits.json")
    train_samples, train_skipped = build_frame_samples(records, legibility_scores, "train", args.min_legibility)
    val_samples, val_skipped = build_frame_samples(records, legibility_scores, "val", args.min_legibility)

    log.info(f"Legibility filter (>= {args.min_legibility}):")
    log.info(f"  train: {len(train_samples)} crops "
             f"({sum(1 for r in records if r['split']=='train')} tracklets, {train_skipped} with 0 legible crops)")
    log.info(f"  val:   {len(val_samples)} crops "
             f"({sum(1 for r in records if r['split']=='val')} tracklets, {val_skipped} with 0 legible crops)")

    n_tracking = len(train_samples)
    if args.soccernet_index:
        with open(args.soccernet_index) as f:
            sn_index = json.load(f)
        sn_samples = [{
            "crop_path": e["image_path"],
            "jersey": int(e["jersey"]),
            "legibility": float(e["legibility"]),
            "tracklet_key": ("soccernet", e.get("tracklet_id", "?")),
            "root": args.soccernet_root,
            "torso_crop": True,
        } for e in sn_index if e.get("legibility", 0.0) >= args.min_legibility]
        train_samples = train_samples + sn_samples
        log.info(f"  +soccernet: {len(sn_samples)} legible crops "
                 f"({len({s['tracklet_key'] for s in sn_samples})} tracklets) -> total {len(train_samples)}")

    tf_train = build_transform_train_perframe(args.img_size)
    tf_infer = build_transform_inference(args.img_size)
    train_ds = JerseyFrameDataset(train_samples, dataset_dir, tf_train)
    val_ds = JerseyFrameDataset(val_samples, dataset_dir, tf_infer)

    sampler = None
    if args.soccernet_index and args.soccernet_frac is not None:
        n_sn = len(train_samples) - n_tracking
        if n_tracking > 0 and n_sn > 0:
            w_track = (1.0 - args.soccernet_frac) / n_tracking
            w_sn = args.soccernet_frac / n_sn
            weights = [w_track] * n_tracking + [w_sn] * n_sn
            sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)
            log.info(f"  Mixed sampler: soccernet_frac={args.soccernet_frac}")
    elif args.balanced_sampling:
        sampler = make_balanced_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = DigitCompositionalMIL(pretrained_backbone=(args.init_from is None)).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        log.info(f"Warm-started from {args.init_from}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_key = (-1.0, -1.0)  # (tracklet_geo_acc, frame_acc)
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device,
                                 args.label_smoothing, mask_tens=not args.no_mask_tens)
        frame_acc = validate_perframe(model, val_loader, device)
        tr_metrics = validate_tracklets(model, val_samples, dataset_dir, device, transform=tf_infer)

        log.info(f"Epoch {epoch+1}/{args.epochs} | loss: {train_loss:.4f} | "
                 f"val frame_acc: {frame_acc:.4f} | "
                 f"val tracklet_geo: {tr_metrics['tracklet_geo_acc']:.4f} | "
                 f"val tracklet_ari: {tr_metrics['tracklet_ari_acc']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_frame_acc": frame_acc, **tr_metrics})

        key = (tr_metrics["tracklet_geo_acc"], frame_acc)
        if key > best_key:
            best_key = key
            save_full_checkpoint(model, optimizer, scheduler, epoch, tr_metrics["tracklet_geo_acc"],
                                 output_dir / "best.pt", img_size=args.img_size)
            log.info(f"  -> New best: tracklet_geo={best_key[0]:.4f} frame={best_key[1]:.4f}")

    save_full_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_key[0],
                         output_dir / "final.pt", img_size=args.img_size)
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump({"args": vars(args), "best_tracklet_geo_acc": best_key[0],
                   "best_frame_acc": best_key[1], "history": history}, f, indent=2)
    log.info(f"Best val: tracklet_geo={best_key[0]:.4f} frame={best_key[1]:.4f}")
    log.info(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
