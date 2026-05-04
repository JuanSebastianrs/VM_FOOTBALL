"""
Preprocess SoccerNet sn-jersey dataset.

- Scores each frame per tracklet using simple quality metrics.
- Selects top-K frames per tracklet.
- Writes metadata CSVs and optionally copies selected frames.

This keeps the original tracklet structure to support tracklet-level training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ROI_X0 = 0.20
ROI_X1 = 0.80
ROI_Y0 = 0.20
ROI_Y1 = 0.70


def _list_image_files(folder: Path) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(paths)


def _read_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return img


def _score_frame(img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = float(np.mean(np.hypot(sobel_x, sobel_y)))

    h, w = gray.shape
    x0 = int(w * ROI_X0)
    x1 = int(w * ROI_X1)
    y0 = int(h * ROI_Y0)
    y1 = int(h * ROI_Y1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        roi = gray

    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(roi_blur, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    try:
        roi_area = float(roi.shape[0] * roi.shape[1])
        mser = cv2.MSER_create(_delta=5, _min_area=30, _max_area=int(0.2 * roi_area))
        regions, _ = mser.detectRegions(roi)
        mser_area = float(sum(len(r) for r in regions))
        mser_ratio = mser_area / roi_area if roi_area > 0 else 0.0
    except Exception:
        mser_ratio = 0.0

    brightness_penalty = abs(brightness - 128.0) / 128.0

    # Score is used only for ranking within a tracklet.
    score = (
        math.log1p(sharp)
        + 0.7 * math.log1p(contrast)
        + 0.4 * math.log1p(grad)
        + 0.8 * math.log1p(edge_density * 1000.0)
        + 0.6 * math.log1p(mser_ratio * 1000.0)
        - 0.8 * brightness_penalty
    )

    return {
        "sharp": sharp,
        "contrast": contrast,
        "brightness": brightness,
        "grad": grad,
        "edge_density": edge_density,
        "mser_ratio": mser_ratio,
        "score": score,
    }


def _select_k(
    n: int,
    fixed_k: int,
    keep_ratio: float,
    min_keep: int,
    max_keep: int,
) -> int:
    if n <= 0:
        return 0
    if fixed_k > 0:
        return min(n, fixed_k)
    k = int(round(n * keep_ratio))
    k = max(k, min_keep)
    k = min(k, max_keep)
    return min(n, k)


def _load_gt(gt_path: Path) -> Dict[str, int]:
    if not gt_path.exists():
        return {}
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _copy_selected(
    src: Path,
    dst: Path,
) -> None:
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _process_split(
    split_name: str,
    split_dir: Path,
    output_root: Path,
    gt_map: Dict[str, int],
    fixed_k: int,
    keep_ratio: float,
    min_keep: int,
    max_keep: int,
    copy_selected: bool,
    export_class_folders: bool,
    keep_all: bool,
) -> Dict[str, int]:
    images_dir = split_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    split_out = output_root / split_name
    _ensure_dir(split_out)
    out_images = split_out / "images"

    rows_all: List[Dict[str, object]] = []
    rows_sel: List[Dict[str, object]] = []

    tracklets = [p for p in images_dir.iterdir() if p.is_dir()]
    tracklets = sorted(tracklets, key=lambda p: p.name)

    total_frames = 0
    total_selected = 0

    for track_dir in tracklets:
        track_id = track_dir.name
        label = gt_map.get(track_id)
        img_paths = _list_image_files(track_dir)
        if not img_paths:
            continue

        scored: List[Tuple[Path, Dict[str, float]]] = []
        for img_path in img_paths:
            img = _read_image(img_path)
            if img is None:
                continue
            metrics = _score_frame(img)
            h, w = img.shape[:2]
            metrics["width"] = float(w)
            metrics["height"] = float(h)
            scored.append((img_path, metrics))

        if not scored:
            continue

        scored.sort(key=lambda x: x[1]["score"], reverse=True)
        if keep_all:
            k = len(scored)
        else:
            k = _select_k(len(scored), fixed_k, keep_ratio, min_keep, max_keep)
        selected_set = {p for p, _ in scored[:k]}

        for rank, (img_path, metrics) in enumerate(scored, start=1):
            is_selected = img_path in selected_set
            row = {
                "split": split_name,
                "tracklet_id": track_id,
                "image_path": str(img_path.relative_to(split_dir)),
                "label": label if label is not None else "",
                "width": int(metrics["width"]),
                "height": int(metrics["height"]),
                "sharp": round(metrics["sharp"], 4),
                "contrast": round(metrics["contrast"], 4),
                "brightness": round(metrics["brightness"], 4),
                "grad": round(metrics["grad"], 4),
                "edge_density": round(metrics["edge_density"], 6),
                "mser_ratio": round(metrics["mser_ratio"], 6),
                "score": round(metrics["score"], 6),
                "rank": rank,
                "selected": 1 if is_selected else 0,
            }
            rows_all.append(row)
            total_frames += 1

            if is_selected:
                rows_sel.append(row)
                total_selected += 1

                if copy_selected:
                    dst = out_images / track_id / img_path.name
                    _copy_selected(img_path, dst)

                if export_class_folders and label is not None and label >= 0:
                    class_dir = output_root / f"{split_name}_cls" / f"{label:02d}"
                    _ensure_dir(class_dir)
                    dst_name = f"{track_id}_{img_path.name}"
                    _copy_selected(img_path, class_dir / dst_name)

    fieldnames = [
        "split",
        "tracklet_id",
        "image_path",
        "label",
        "width",
        "height",
        "sharp",
        "contrast",
        "brightness",
        "grad",
        "edge_density",
        "mser_ratio",
        "score",
        "rank",
        "selected",
    ]

    _write_csv(split_out / "metadata.csv", rows_all, fieldnames)
    _write_csv(split_out / "selected.csv", rows_sel, fieldnames)

    summary = {
        "split": split_name,
        "tracklets": len(tracklets),
        "frames_total": total_frames,
        "frames_selected": total_selected,
        "fixed_k": fixed_k,
        "keep_ratio": keep_ratio,
        "min_keep": min_keep,
        "max_keep": max_keep,
    }

    with (split_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "frames_total": total_frames,
        "frames_selected": total_selected,
        "tracklets": len(tracklets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess sn-jersey tracklets")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/sn_jersey_2023/jersey-2023"),
        help="Path to jersey-2023 root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/sn_jersey_2023/processed"),
        help="Output folder for processed data",
    )
    parser.add_argument("--fixed-k", type=int, default=0, help="Fixed K frames per tracklet")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Ratio of frames to keep")
    parser.add_argument("--min-keep", type=int, default=4, help="Min frames to keep")
    parser.add_argument("--max-keep", type=int, default=12, help="Max frames to keep")
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy selected frames, only write metadata",
    )
    parser.add_argument(
        "--export-class-folders",
        action="store_true",
        help="Also export selected frames to class folders for ImageFolder training",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all frames (disable top-K filtering) while still generating metadata.",
    )

    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    _ensure_dir(output_root)

    splits = [
        ("train", input_root / "train"),
        ("test", input_root / "test"),
        ("challenge", input_root / "challenge"),
    ]

    overall = {}
    for split_name, split_dir in splits:
        if not split_dir.exists():
            print(f"Skip missing split: {split_dir}")
            continue

        gt_path = split_dir / f"{split_name}_gt.json"
        gt_map = _load_gt(gt_path)

        stats = _process_split(
            split_name=split_name,
            split_dir=split_dir,
            output_root=output_root,
            gt_map=gt_map,
            fixed_k=args.fixed_k,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            max_keep=args.max_keep,
            copy_selected=not args.no_copy,
            export_class_folders=args.export_class_folders,
            keep_all=args.keep_all,
        )
        overall[split_name] = stats
        print(
            f"[{split_name}] tracklets={stats['tracklets']} frames={stats['frames_total']} selected={stats['frames_selected']}"
        )

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)


if __name__ == "__main__":
    main()
