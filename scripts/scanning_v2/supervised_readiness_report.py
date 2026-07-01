# scripts/scanning_v2/supervised_readiness_report.py
"""
Genera el reporte de readiness supervisado: dice si hay evidencia suficiente para
entrenar, cuanto falta, si los clips/anotaciones estan listos y el estado del modelo.

Ejemplo:
  python scripts/scanning_v2/supervised_readiness_report.py \
    --video_id SNMOT-148 --v2_outputs outputs/scanning_v2/SNMOT-148 \
    --supervised_outputs outputs/scanning_v2_supervised/SNMOT-148 \
    --annotations data/annotations/scanning_windows_gt.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning_v2.supervised.readiness import compute_readiness, render_markdown  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--v2_outputs", required=True)
    p.add_argument("--supervised_outputs", required=True)
    p.add_argument("--annotations", default="data/annotations/scanning_windows_gt.csv")
    p.add_argument("--config", default="configs/scanning_v2_supervised.yaml")
    return p.parse_args()


def main():
    a = get_args()
    min_lab, min_pos = 20, 3
    if Path(a.config).exists():
        with open(a.config, "r", encoding="utf-8") as f:
            m = (yaml.safe_load(f) or {}).get("model", {})
        min_lab = int(m.get("min_labeled_samples", 20))
        min_pos = int(m.get("min_positive_samples", 3))

    r = compute_readiness(a.video_id, a.v2_outputs, a.supervised_outputs,
                          a.annotations, min_lab, min_pos)
    out = Path(a.supervised_outputs) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "supervised_readiness.json").write_text(
        json.dumps(r, indent=2, default=str), encoding="utf-8")
    (out / "supervised_readiness.md").write_text(render_markdown(r), encoding="utf-8")
    print(f"[readiness] labeled={r['labeled_effective']}/{min_lab} "
          f"pos={r['positives']}/{min_pos} can_train={r['can_train']}")
    print(f"[readiness] accion: {r['recommended_action']}")
    print(f"[readiness] -> {out / 'supervised_readiness.md'}")


if __name__ == "__main__":
    main()
