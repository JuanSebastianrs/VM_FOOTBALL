"""
Validate preprocessed sn-jersey datasets and optionally package them.

Checks:
- split existence (train/test)
- image file counts
- zero-byte files
- required metadata files per mode

Optional:
- create tar.gz archives for upload to GCS
"""

from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class SplitStats:
    files: int
    zero_bytes: int


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def split_stats(root: Path, split: str) -> SplitStats:
    split_dir = root / split
    files = 0
    zero = 0
    for p in iter_images(split_dir):
        files += 1
        if p.stat().st_size == 0:
            zero += 1
    return SplitStats(files=files, zero_bytes=zero)


def validate_required_files(full_root: Path, processed_root: Path) -> List[str]:
    errors: List[str] = []
    for split in ["train", "test"]:
        if not (full_root / split / f"{split}_gt.json").exists():
            errors.append(f"Missing file: {full_root / split / f'{split}_gt.json'}")
        if not (processed_root / split / "selected.csv").exists():
            errors.append(f"Missing file: {processed_root / split / 'selected.csv'}")
    return errors


def make_tar(source_dir: Path, output_tar_gz: Path) -> None:
    output_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and package preprocessed sn-jersey datasets")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/sn_jersey_2023"))
    parser.add_argument("--full-source", type=str, default="jersey-2023")
    parser.add_argument("--processed-source", type=str, default="processed")
    parser.add_argument("--full-preprocessed", type=str, default="preprocessed_full_keepall")
    parser.add_argument("--processed-preprocessed", type=str, default="preprocessed_processed_keepall")
    parser.add_argument("--report-path", type=Path, default=Path("tmp/preprocessed_dataset_validation_report.json"))
    parser.add_argument("--create-archives", action="store_true")
    parser.add_argument("--archive-dir", type=Path, default=Path("tmp"))
    parser.add_argument("--archive-suffix", type=str, default="v2")
    args = parser.parse_args()

    src_full = args.base_dir / args.full_source
    src_processed = args.base_dir / args.processed_source
    pp_full = args.base_dir / args.full_preprocessed
    pp_processed = args.base_dir / args.processed_preprocessed

    roots: Dict[str, Path] = {
        "source_full": src_full,
        "source_processed": src_processed,
        "preprocessed_full": pp_full,
        "preprocessed_processed": pp_processed,
    }

    report: Dict[str, object] = {"roots": {}, "required_file_errors": [], "count_mismatches": []}

    for name, root in roots.items():
        if not root.exists():
            raise FileNotFoundError(f"Missing root: {root}")
        root_data: Dict[str, object] = {}
        for split in ["train", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Missing split: {split_dir}")
            stats = split_stats(root, split)
            root_data[split] = {"files": stats.files, "zero_bytes": stats.zero_bytes}
        report["roots"][name] = root_data

    report["required_file_errors"] = validate_required_files(pp_full, pp_processed)

    for split in ["train", "test"]:
        src = report["roots"]["source_full"][split]["files"]
        pp = report["roots"]["preprocessed_full"][split]["files"]
        if src != pp:
            report["count_mismatches"].append(f"full {split}: source={src} preprocessed={pp}")
        src_p = report["roots"]["source_processed"][split]["files"]
        pp_p = report["roots"]["preprocessed_processed"][split]["files"]
        if src_p != pp_p:
            report["count_mismatches"].append(f"processed {split}: source={src_p} preprocessed={pp_p}")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Report saved to {args.report_path}")
    if report["required_file_errors"]:
        print("[WARN] Required file issues detected:")
        for err in report["required_file_errors"]:
            print(f"  - {err}")
    if report["count_mismatches"]:
        print("[WARN] Count mismatches detected:")
        for msg in report["count_mismatches"]:
            print(f"  - {msg}")

    if args.create_archives:
        full_tar = args.archive_dir / f"sn_jersey_2023_{pp_full.name}_{args.archive_suffix}.tar.gz"
        proc_tar = args.archive_dir / f"sn_jersey_2023_{pp_processed.name}_{args.archive_suffix}.tar.gz"
        make_tar(pp_full, full_tar)
        make_tar(pp_processed, proc_tar)
        print(f"[OK] Created archive: {full_tar}")
        print(f"[OK] Created archive: {proc_tar}")


if __name__ == "__main__":
    main()
