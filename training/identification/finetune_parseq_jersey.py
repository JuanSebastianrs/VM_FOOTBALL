# training/identification/finetune_parseq_jersey.py
"""
Fine-tune de PARSeq (baudm/parseq, torch.hub) en crops de dorsales.

Motivacion: el ensamble v2.1 usa PARSeq generico de texto de escena; la
literatura del reto SoccerNet jersey (Koshkina & Elder 2024, docs/2404.08401v5.pdf)
obtiene sus ganancias fine-tuneando PARSeq en crops de dorsales. Datos LIMPIOS:

  - SoccerNet jersey-2023 **TRAIN** per-frame index (legible, con label).
    El split TEST de jersey-2023 esta PROHIBIDO: fuga confirmada con nuestro
    benchmark de tracking (docs/jersey_perframe_v2.md SS20).
  - Crops del split train de datasets/jersey_tracking_v2_224 (GT nativo,
    legibilidad >= min_legibility).

Seleccion de modelo: exact-match (numero 1..99 parseado de la lectura) sobre
los crops legibles del split VAL del dataset de tracking.

Uso (RTX 2080 8GB, ~30k crops):
  python training/identification/finetune_parseq_jersey.py \
      --model parseq --epochs 8 --batch_size 96 --lr 7e-5 \
      --output_dir runs/parseq_jersey_ft
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scripts.infer_jersey_on_dataset import build_splits  # noqa: E402


def parse_number(label: str):
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits or len(digits) > 2:
        return None
    n = int(digits)
    return n if 1 <= n <= 99 else None


def build_tracking_samples(dataset_dir: Path, split: str, min_legibility: float):
    records = build_splits(dataset_dir / "tracklets.json", dataset_dir / "splits.json")
    with open(dataset_dir / "legibility_scores.json") as f:
        leg = json.load(f)
    samples = []
    for rec in records:
        if rec.get("split") != split:
            continue
        jn = rec.get("jersey_number")
        if jn is None or not (1 <= int(jn) <= 99):
            continue
        for cp in rec.get("crop_paths", []):
            rel = Path(cp).as_posix()
            if leg.get(rel, 0.0) < min_legibility:
                continue
            p = dataset_dir / cp
            if p.exists():
                samples.append((str(p), str(int(jn))))
    return samples


def build_soccernet_samples(index_path: Path, root: Path, min_legibility: float):
    with open(index_path) as f:
        index = json.load(f)
    samples = []
    for it in index:
        if it.get("legibility", 1.0) < min_legibility:
            continue
        jn = it.get("jersey")
        if jn is None or not (1 <= int(jn) <= 99):
            continue
        p = root / it["image_path"]
        if p.exists():
            samples.append((str(p), str(int(jn))))
    return samples


class JerseyCropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_transforms(img_size):
    import torchvision.transforms as T
    norm = T.Normalize(0.5, 0.5)
    train_tf = T.Compose([
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
        T.RandomApply([T.GaussianBlur(3)], p=0.3),
        T.RandomRotation(7, fill=127),
        T.RandomPerspective(distortion_scale=0.15, p=0.3, fill=127),
        T.Resize(img_size, antialias=True),
        T.ToTensor(),
        norm,
    ])
    eval_tf = T.Compose([T.Resize(img_size, antialias=True), T.ToTensor(), norm])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device):
    """Exact match del numero parseado (1..99) sobre crops de val."""
    model.eval()
    hits = total = reads = 0
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds, _ = model.tokenizer.decode(logits.softmax(-1))
        for pred, gt in zip(preds, labels):
            total += 1
            num = parse_number(pred)
            if num is not None:
                reads += 1
                if num == int(gt):
                    hits += 1
    acc = hits / max(total, 1)
    read_rate = reads / max(total, 1)
    return acc, read_rate


def main():
    ap = argparse.ArgumentParser(description="Fine-tune PARSeq en crops de dorsales")
    ap.add_argument("--model", default="parseq", help="parseq (base) | parseq_tiny")
    ap.add_argument("--dataset_dir", default="datasets/jersey_tracking_v2_224")
    ap.add_argument("--soccernet_index",
                    default="datasets/soccernet/jersey-2023/perframe_index_train.json")
    ap.add_argument("--soccernet_root", default="datasets/soccernet/jersey-2023")
    ap.add_argument("--min_legibility", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_frac", type=float, default=0.075)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_dir", default="runs/parseq_jersey_ft")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.hub.load("baudm/parseq", args.model, pretrained=True,
                           trust_repo=True).to(device)
    model.log = lambda *a, **k: None  # training_step llama self.log (no hay Trainer)
    img_size = tuple(model.hparams.img_size)

    dataset_dir = Path(args.dataset_dir)
    train_track = build_tracking_samples(dataset_dir, "train", args.min_legibility)
    train_sn = build_soccernet_samples(Path(args.soccernet_index),
                                       Path(args.soccernet_root), args.min_legibility)
    val_samples = build_tracking_samples(dataset_dir, "val", args.min_legibility)
    train_samples = train_track + train_sn
    print(f"train: {len(train_track)} tracking + {len(train_sn)} soccernet-train "
          f"= {len(train_samples)} | val: {len(val_samples)} | img_size {img_size}")

    train_tf, eval_tf = make_transforms(img_size)
    train_ds = JerseyCropDataset(train_samples, train_tf)
    val_ds = JerseyCropDataset(val_samples, eval_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    acc0, rr0 = evaluate(model, val_dl, device)
    print(f"[baseline pretrained] val exact-match {acc0:.4f} | read rate {rr0:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_total = len(train_dl) * args.epochs
    steps_warmup = max(1, int(steps_total * args.warmup_frac))

    def lr_lambda(step):
        if step < steps_warmup:
            return step / steps_warmup
        t = (step - steps_warmup) / max(1, steps_total - steps_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = [{"epoch": 0, "val_acc": acc0, "read_rate": rr0}]
    best_acc = -1.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (x, labels) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model.training_step((x, list(labels)), i)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running += float(loss.detach())
            if (i + 1) % 50 == 0:
                print(f"  ep{epoch} {i + 1}/{len(train_dl)} "
                      f"loss {running / (i + 1):.4f} lr {scheduler.get_last_lr()[0]:.2e}",
                      flush=True)
        acc, rr = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "val_acc": acc, "read_rate": rr,
                        "train_loss": running / max(1, len(train_dl))})
        print(f"[ep {epoch}/{args.epochs}] loss {running / max(1, len(train_dl)):.4f} "
              f"val exact-match {acc:.4f} read rate {rr:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_name": args.model,
                        "state_dict": model.state_dict(),
                        "val_acc": acc, "epoch": epoch,
                        "img_size": list(img_size)},
                       out_dir / "best.pt")
            print(f"  -> nuevo best.pt (val {acc:.4f})", flush=True)
        with open(out_dir / "history.json", "w") as f:
            json.dump({"args": vars(args), "history": history,
                       "best_val_acc": best_acc}, f, indent=1)

    torch.save({"model_name": args.model, "state_dict": model.state_dict(),
                "val_acc": history[-1]["val_acc"], "epoch": args.epochs,
                "img_size": list(img_size)}, out_dir / "last.pt")
    print(f"FIN. best val exact-match {best_acc:.4f} (baseline {acc0:.4f}) "
          f"-> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
