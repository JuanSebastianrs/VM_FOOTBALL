# scripts/migrate_outputs_layout.py
"""
Migra `outputs/` al layout canonico (ver core/scanning_v2/paths.py):

  1. outputs/scanning_v2/<SEQ>/*          -> outputs/<SEQ>/scanning/
  2. outputs/scanning_v2_supervised_weak  -> outputs/scanning_training
  3. outputs/scanning_v2_supervised       -> outputs/_archive/scanning_v2_supervised
  4. outputs/scanning (v1)                -> outputs/_archive/scanning_v1
  5. outputs/*.log, outputs/*.err         -> outputs/_archive/logs/

Idempotente: lo ya migrado se salta. Nada se borra; lo legado va a _archive/.

  python scripts/migrate_outputs_layout.py [--outputs_root outputs] [--dry_run]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _move(src: Path, dst: Path, dry: bool) -> None:
    print(f"  {src} -> {dst}")
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def migrate(root: Path, dry: bool = False) -> None:
    archive = root / "_archive"

    # 1) scanning_v2 por secuencia -> outputs/<SEQ>/scanning
    v2 = root / "scanning_v2"
    if v2.is_dir():
        print("[migrate] scanning_v2/<SEQ> -> <SEQ>/scanning")
        for seq_dir in sorted(p for p in v2.iterdir() if p.is_dir()):
            dst = root / seq_dir.name / "scanning"
            if dst.exists():
                print(f"  SKIP {seq_dir.name}: {dst} ya existe")
                continue
            _move(seq_dir, dst, dry)
        if not dry and not any(v2.iterdir()):
            v2.rmdir()

    # 2) entrenamiento weak -> scanning_training
    old_train = root / "scanning_v2_supervised_weak"
    new_train = root / "scanning_training"
    if old_train.is_dir() and not new_train.exists():
        print("[migrate] scanning_v2_supervised_weak -> scanning_training")
        _move(old_train, new_train, dry)

    # 3-4) legado -> _archive
    for legacy, name in [(root / "scanning_v2_supervised", "scanning_v2_supervised"),
                         (root / "scanning", "scanning_v1")]:
        if legacy.is_dir():
            print(f"[migrate] {legacy.name} -> _archive/{name}")
            _move(legacy, archive / name, dry)

    # 5) logs sueltos en la raiz de outputs
    logs = [p for p in root.glob("*.log")] + [p for p in root.glob("*.err")]
    if logs:
        print("[migrate] logs sueltos -> _archive/logs")
        for p in logs:
            _move(p, archive / "logs" / p.name, dry)

    print("[migrate] DONE" + (" (dry run)" if dry else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    migrate(Path(a.outputs_root), a.dry_run)
