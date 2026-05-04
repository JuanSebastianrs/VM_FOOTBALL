"""
Run sn-jersey preprocessing for two sources:
1) Full dataset root (raw tracklets)
2) Already-processed dataset root (optional reprocessing)

This is a launcher around preprocess_sn_jersey.py so we can keep one
canonical preprocessing implementation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_existing_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    # Common typo compatibility: only replace the final path segment.
    if p.name == "jersey_2023":
        alt = p.with_name("jersey-2023")
        if alt.exists():
            return alt
    return p


def _validate_root(root: Path, tag: str) -> None:
    required = ["train", "test", "challenge"]
    missing = [split for split in required if not (root / split / "images").exists()]
    if missing:
        raise FileNotFoundError(
            f"[{tag}] Missing split/images folders in {root}. Missing: {missing}"
        )


def _warn_missing_gt(root: Path, tag: str) -> None:
    for split in ["train", "test", "challenge"]:
        gt = root / split / f"{split}_gt.json"
        if not gt.exists():
            print(f"[WARN][{tag}] Missing GT file: {gt}")


def _run_preprocess(
    input_root: Path,
    output_root: Path,
    fixed_k: int,
    keep_ratio: float,
    min_keep: int,
    max_keep: int,
    no_copy: bool,
    export_class_folders: bool,
    keep_all: bool,
    tag: str,
) -> None:
    script_path = Path(__file__).with_name("preprocess_sn_jersey.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--fixed-k",
        str(fixed_k),
        "--keep-ratio",
        str(keep_ratio),
        "--min-keep",
        str(min_keep),
        "--max-keep",
        str(max_keep),
    ]
    if no_copy:
        cmd.append("--no-copy")
    if export_class_folders:
        cmd.append("--export-class-folders")
    if keep_all:
        cmd.append("--keep-all")

    print(f"\n[{tag}] Running preprocessing")
    print(f"[{tag}] Input:  {input_root}")
    print(f"[{tag}] Output: {output_root}")
    print(f"[{tag}] Cmd: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing for both full and processed sn-jersey roots."
    )

    parser.add_argument(
        "--full-input-root",
        type=str,
        default="datasets/sn_jersey_2023/jersey_2023",
        help="Path to full dataset root (supports jersey_2023 typo fallback).",
    )
    parser.add_argument(
        "--full-output-root",
        type=Path,
        default=Path("datasets/sn_jersey_2023/processed_from_full"),
        help="Output root for full dataset preprocessing.",
    )

    parser.add_argument(
        "--processed-input-root",
        type=str,
        default="datasets/processed",
        help="Path to processed dataset root to preprocess again.",
    )
    parser.add_argument(
        "--processed-output-root",
        type=Path,
        default=Path("datasets/sn_jersey_2023/reprocessed_from_processed"),
        help="Output root for processed dataset reprocessing.",
    )

    parser.add_argument("--fixed-k", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=0.2)
    parser.add_argument("--min-keep", type=int, default=4)
    parser.add_argument("--max-keep", type=int, default=12)
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--export-class-folders", action="store_true")
    parser.add_argument(
        "--keep-all",
        action="store_true",
        default=True,
        help="Disable frame filtering and keep all frames (recommended for fair full vs processed comparison).",
    )

    parser.add_argument(
        "--skip-processed-pass",
        action="store_true",
        help="Skip the second pass over processed input root.",
    )

    args = parser.parse_args()

    full_input = _resolve_existing_path(args.full_input_root)
    processed_input = _resolve_existing_path(args.processed_input_root)

    _validate_root(full_input, "FULL")
    _warn_missing_gt(full_input, "FULL")
    _run_preprocess(
        input_root=full_input,
        output_root=args.full_output_root,
        fixed_k=args.fixed_k,
        keep_ratio=args.keep_ratio,
        min_keep=args.min_keep,
        max_keep=args.max_keep,
        no_copy=args.no_copy,
        export_class_folders=args.export_class_folders,
        keep_all=args.keep_all,
        tag="FULL",
    )

    if args.skip_processed_pass:
        print("\n[INFO] Processed pass skipped by flag.")
        return

    _validate_root(processed_input, "PROCESSED")
    _warn_missing_gt(processed_input, "PROCESSED")
    _run_preprocess(
        input_root=processed_input,
        output_root=args.processed_output_root,
        fixed_k=args.fixed_k,
        keep_ratio=args.keep_ratio,
        min_keep=args.min_keep,
        max_keep=args.max_keep,
        no_copy=args.no_copy,
        export_class_folders=args.export_class_folders,
        keep_all=args.keep_all,
        tag="PROCESSED",
    )

    print("\n[OK] Dual preprocessing finished.")


if __name__ == "__main__":
    main()
