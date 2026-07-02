# scripts/scanning_v2/validate_scanning_annotations.py
"""
Valida la anotacion humana de ventanas de scanning y su sincronizacion con los
eventos V2 actuales.

Ejemplo:
  python scripts/scanning_v2/validate_scanning_annotations.py \
    --video_id SNMOT-148 --outputs_dir outputs/SNMOT-148/scanning \
    --annotations data/annotations/scanning_windows_gt.csv \
    --output_dir outputs/scanning_training_gt/reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised.annotation_validator import (   # noqa: E402
    render_markdown, validate_annotations)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--outputs_dir", required=True, help="outputs V2 del video")
    p.add_argument("--annotations", default="data/annotations/scanning_windows_gt.csv")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    a = get_args()
    odir = Path(a.outputs_dir)
    events_path = odir / "pass_reception_events.parquet"
    current_ids = set()
    if events_path.exists():
        ev = pd.read_parquet(events_path)
        current_ids = set(ev["event_id"].astype(str).tolist())

    ann = pd.read_csv(a.annotations) if Path(a.annotations).exists() else None
    pack_dir = odir / "annotation_pack"   # clip_path es relativo al annotation_pack

    rep = validate_annotations(ann, current_ids, a.video_id,
                               annotation_pack_dir=str(pack_dir))
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotation_validation.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    (out / "annotation_validation.md").write_text(
        render_markdown(rep), encoding="utf-8")
    print(f"[validate] total={rep['total_rows']} labeled={rep['labeled_rows']} "
          f"pos={rep['positives']} neg={rep['negatives']} "
          f"invalid={rep['n_invalid_labels']} sync={rep['synchronized']}")
    print(f"[validate] -> {out / 'annotation_validation.md'}")


if __name__ == "__main__":
    main()
