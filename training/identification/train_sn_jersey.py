"""
Train jersey number classifier on SoccerNet sn-jersey.

Supports two modes:
- processed: uses selected frames from preprocess_sn_jersey.py
- full: uses all frames from the raw dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class FrameItem:
    path: Path
    label: int
    tracklet_id: str


def _load_gt(gt_path: Path) -> Dict[str, int]:
    if not gt_path.exists():
        return {}
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def _list_image_files(folder: Path) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(paths)


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _apply_unsharp(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def _read_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return img


def _build_items_full(
    split_dir: Path,
    gt_map: Dict[str, int],
    skip_no_number: bool,
) -> List[FrameItem]:
    images_dir = split_dir / "images"
    if not images_dir.exists():
        return []

    items: List[FrameItem] = []
    for track_dir in sorted(images_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        track_id = track_dir.name
        label = gt_map.get(track_id)
        if label is None:
            continue
        if skip_no_number and label < 0:
            continue
        for img_path in _list_image_files(track_dir):
            items.append(FrameItem(path=img_path, label=label, tracklet_id=track_id))
    return items


def _build_items_processed(
    split_name: str,
    dataset_root: Path,
    processed_root: Path,
    gt_map: Dict[str, int],
    skip_no_number: bool,
) -> List[FrameItem]:
    selected_csv = processed_root / split_name / "selected.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(f"Missing selected.csv: {selected_csv}")

    split_dir = dataset_root / split_name
    processed_images = processed_root / split_name / "images"

    items: List[FrameItem] = []
    with selected_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row.get("tracklet_id") or ""
            label_str = row.get("label")
            if label_str is None or label_str == "":
                label = gt_map.get(track_id)
            else:
                label = int(label_str)
            if label is None:
                continue
            if skip_no_number and label < 0:
                continue

            img_rel_raw = row.get("image_path") or ""
            img_rel_norm = img_rel_raw.replace("\\", "/")
            img_rel = Path(img_rel_norm) if img_rel_norm else None
            img_name = img_rel.name if img_rel else None
            if not img_name:
                continue

            preferred = processed_images / track_id / img_name
            if preferred.exists():
                img_path = preferred
            else:
                img_path = split_dir / img_rel

            if not img_path.exists():
                continue

            items.append(FrameItem(path=img_path, label=label, tracklet_id=track_id))

    return items


def _split_by_tracklet(
    items: List[FrameItem],
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    track_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        track_to_indices[item.tracklet_id].append(idx)

    tracklets = sorted(track_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(tracklets)

    val_count = int(len(tracklets) * val_split)
    val_set = set(tracklets[:val_count])

    train_indices: List[int] = []
    val_indices: List[int] = []
    for track_id, idxs in track_to_indices.items():
        if track_id in val_set:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)
    return train_indices, val_indices


class TrackletFrameDataset(Dataset):
    def __init__(
        self,
        items: List[FrameItem],
        indices: Optional[List[int]],
        img_size: int,
        train: bool,
        use_clahe: bool,
        use_unsharp: bool,
    ):
        self.items = items if indices is None else [items[i] for i in indices]
        self.img_size = img_size
        self.use_clahe = use_clahe
        self.use_unsharp = use_unsharp

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        img = _read_image(item.path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if self.use_clahe:
            img = _apply_clahe(img)
        if self.use_unsharp:
            img = _apply_unsharp(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        tensor = self.transform(pil)
        return tensor, item.label


def _accuracy_topk(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...]) -> List[float]:
    maxk = max(topk)
    with torch.no_grad():
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / targets.size(0)).item())
        return res


def _build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    else:
        raise ValueError("model must be resnet18 or resnet34")

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_classes),
    )
    return model


def _build_class_weights(items: List[FrameItem], indices: List[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in indices:
        label = items[idx].label
        if 0 <= label < num_classes:
            counts[label] += 1.0

    nonzero = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32)
    if nonzero.any():
        mean_count = counts[nonzero].mean()
        weights[nonzero] = mean_count / counts[nonzero]
    return weights


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        top1, top3 = _accuracy_topk(outputs, labels, (1, 3))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top3 += top3 * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total_samples, total_top1 / total_samples, total_top3 / total_samples


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = labels.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            top1, top3 = _accuracy_topk(outputs, labels, (1, 3))
            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top3 += top3 * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total_samples, total_top1 / total_samples, total_top3 / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sn-jersey classifier")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/sn_jersey_2023/jersey-2023"))
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/sn_jersey_2023/processed"))
    parser.add_argument("--mode", choices=["processed", "full"], default="processed")
    parser.add_argument("--skip-no-number", action="store_true", default=True)
    parser.add_argument("--use-test-as-val", action="store_true", default=False)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-unsharp", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/train/sn_jersey"))
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint_last.pt")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="plateau")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--class-weighted-loss", action="store_true", default=True)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = args.dataset_root
    processed_root = args.processed_root

    train_gt = _load_gt(dataset_root / "train" / "train_gt.json")
    test_gt = _load_gt(dataset_root / "test" / "test_gt.json")

    if args.mode == "processed":
        train_items = _build_items_processed("train", dataset_root, processed_root, train_gt, args.skip_no_number)
        test_items = _build_items_processed("test", dataset_root, processed_root, test_gt, args.skip_no_number)
    else:
        train_items = _build_items_full(dataset_root / "train", train_gt, args.skip_no_number)
        test_items = _build_items_full(dataset_root / "test", test_gt, args.skip_no_number)

    if args.use_test_as_val:
        train_indices = list(range(len(train_items)))
        val_indices = list(range(len(test_items)))
        val_items = test_items
    else:
        train_indices, val_indices = _split_by_tracklet(train_items, args.val_split, args.seed)
        val_items = train_items

    train_dataset = TrackletFrameDataset(
        items=train_items,
        indices=train_indices,
        img_size=args.img_size,
        train=True,
        use_clahe=not args.no_clahe,
        use_unsharp=not args.no_unsharp,
    )
    val_dataset = TrackletFrameDataset(
        items=val_items,
        indices=val_indices,
        img_size=args.img_size,
        train=False,
        use_clahe=not args.no_clahe,
        use_unsharp=not args.no_unsharp,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    if args.class_weighted_loss:
        class_weights = _build_class_weights(train_items, train_indices, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=args.min_lr
        )
    else:
        scheduler = None

    output_dir = args.output / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    history_rows: List[Dict[str, object]] = []
    start_epoch = 1
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_acc = checkpoint.get("best_acc", 0.0)
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[RESUME] Loaded {args.resume} (epoch {start_epoch - 1})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_top1, train_top3 = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top3 = _eval_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} top1 {train_top1:.4f} top3 {train_top3:.4f} | "
            f"val loss {val_loss:.4f} top1 {val_top1:.4f} top3 {val_top3:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top3": train_top3,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top3": val_top3,
            }
        )

        if val_top1 > best_acc:
            best_acc = val_top1
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            if (best_acc - val_top1) > args.early_stopping_min_delta:
                no_improve_epochs += 1

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_top1)
            else:
                scheduler.step()

        checkpoint_path = output_dir / "checkpoint_last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "no_improve_epochs": no_improve_epochs,
            },
            checkpoint_path,
        )

        if no_improve_epochs >= args.early_stopping_patience:
            print(
                f"[EARLY STOP] No meaningful val_top1 improvement for {args.early_stopping_patience} epochs. "
                f"Stopping at epoch {epoch}."
            )
            break

    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_top1", "train_top3", "val_loss", "val_top1", "val_top3"],
        )
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)

    summary = {
        "mode": args.mode,
        "run_name": args.run_name,
        "best_val_top1": best_acc,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "stopped_epoch": history_rows[-1]["epoch"] if history_rows else 0,
        "img_size": args.img_size,
        "skip_no_number": args.skip_no_number,
        "use_test_as_val": args.use_test_as_val,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "class_weighted_loss": args.class_weighted_loss,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
