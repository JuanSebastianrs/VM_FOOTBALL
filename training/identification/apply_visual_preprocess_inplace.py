"""
Apply visual preprocessing in-place to sn-jersey image folders.

Pipeline per image:
1) CLAHE on luminance channel
2) Mild denoising
3) Unsharp masking

This script overwrites images in-place (no dataset copy).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _allowed_split(path: Path, root: Path, splits: Optional[List[str]]) -> bool:
    if not splits:
        return True
    rel = path.relative_to(root).as_posix()
    for split in splits:
        if rel.startswith(f"{split}/"):
            return True
    return False


def iter_images(root: Path, splits: Optional[List[str]] = None) -> Iterable[Path]:
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort(key=lambda x: x.as_posix())
    for p in paths:
        if _allowed_split(p, root, splits):
            yield p


def preprocess_image(
    img,
    clahe_clip: float,
    clahe_grid: int,
    denoise_h: int,
    unsharp_sigma: float,
    unsharp_amount: float,
):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if denoise_h > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, denoise_h, denoise_h, 7, 21)

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=unsharp_sigma)
    img = cv2.addWeighted(img, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)
    return img


def process_root(
    root: Path,
    clahe_clip: float,
    clahe_grid: int,
    denoise_h: int,
    unsharp_sigma: float,
    unsharp_amount: float,
    splits: Optional[List[str]],
    start_after: Optional[str],
    log_every: int,
    log_each_image: bool,
) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Missing root: {root}")

    start_after_norm = start_after.replace("\\", "/") if start_after else None

    total = 0
    failed = 0
    skipped = 0
    for img_path in iter_images(root, splits=splits):
        rel = img_path.relative_to(root).as_posix()
        if start_after_norm and rel <= start_after_norm:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            failed += 1
            continue

        out = preprocess_image(
            img=img,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
            denoise_h=denoise_h,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )
        ok = cv2.imwrite(str(img_path), out)
        if not ok:
            failed += 1
        total += 1

        if log_each_image:
            print(f"[{root}] {total} -> {rel}")
        elif log_every > 0 and total % log_every == 0:
            print(f"[{root}] processed={total} skipped={skipped} failed={failed} current={rel}")

    print(f"[DONE][{root}] processed={total} skipped={skipped} failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply visual preprocessing in-place")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "datasets/sn_jersey_2023/preprocessed_full_keepall",
            "datasets/sn_jersey_2023/preprocessed_processed_keepall",
        ],
        help="One or more roots to process in-place",
    )
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    parser.add_argument("--denoise-h", type=int, default=3)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=0.4)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "test"],
        help="Split filter. Default is train/test only (challenge is intentionally ignored).",
    )
    parser.add_argument(
        "--start-after",
        type=str,
        default=None,
        help="Resume cursor (relative path inside each root), e.g. challenge/images/556/556_709.jpg",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N processed images (default: 500). Use 0 to disable periodic logs.",
    )
    parser.add_argument(
        "--log-each-image",
        action="store_true",
        help="Print one line per processed image.",
    )
    args = parser.parse_args()

    for root_str in args.roots:
        process_root(
            root=Path(root_str),
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            denoise_h=args.denoise_h,
            unsharp_sigma=args.unsharp_sigma,
            unsharp_amount=args.unsharp_amount,
            splits=args.splits,
            start_after=args.start_after,
            log_every=args.log_every,
            log_each_image=args.log_each_image,
        )


if __name__ == "__main__":
    main()
