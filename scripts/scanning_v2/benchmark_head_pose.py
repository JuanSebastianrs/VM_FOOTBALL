# scripts/scanning_v2/benchmark_head_pose.py
"""
Benchmark de backends de head pose (V2).

Reporta, a partir de head_pose.parquet, la distribucion de backends realmente
usados, su confianza media y la validez de los head crops; ademas comprueba la
DISPONIBILIDAD real de cada backend (6DRepNet / MediaPipe FaceLandmarker) en el
entorno. No es gaze real: el yaw es camara-relativo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def backend_availability():
    return {
        "sixdrepnet": importlib.util.find_spec("sixdrepnet") is not None,
        "mediapipe": importlib.util.find_spec("mediapipe") is not None,
        "yolo_pose_body": importlib.util.find_spec("ultralytics") is not None,
    }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head_pose", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    a = get_args()
    hp = pd.read_parquet(a.head_pose)
    avail = backend_availability()
    used = hp["head_pose_backend_used"].value_counts().to_dict()
    by_backend = {}
    for bk, g in hp.groupby("head_pose_backend_used"):
        by_backend[str(bk)] = {
            "n": int(len(g)),
            "mean_confidence": round(float(g["head_pose_confidence"].mean()), 3),
            "head_crop_valid_rate": round(float(g["head_crop_valid"].mean()), 3),
            "mean_head_crop_quality": round(float(g["head_crop_quality"].mean()), 3),
        }
    result = {
        "backend_availability": avail,
        "n_head_pose_rows": int(len(hp)),
        "backend_used_dist": {str(k): int(v) for k, v in used.items()},
        "per_backend": by_backend,
        "mean_confidence_overall": round(float(hp["head_pose_confidence"].mean()), 3),
        "head_crop_valid_rate_overall": round(float(hp["head_crop_valid"].mean()), 3),
    }
    out = Path(a.output_dir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "head_pose_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Benchmark de head pose (V2)", "",
             "> yaw camara-relativo, NO gaze real.", "",
             "## Disponibilidad de backends",
             *[f"- {k}: {'OK' if v else 'NO instalado'}" for k, v in avail.items()],
             "", "## Backend usado por head-pose row", "",
             "| backend | n | conf | head_crop_valid | crop_quality |",
             "|---|---|---|---|---|"]
    for bk, m in by_backend.items():
        lines.append(f"| {bk} | {m['n']} | {m['mean_confidence']} | "
                     f"{m['head_crop_valid_rate']} | {m['mean_head_crop_quality']} |")
    (out / "head_pose_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
