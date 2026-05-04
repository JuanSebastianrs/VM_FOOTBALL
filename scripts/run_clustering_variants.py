"""
Run multiple clustering variants on clip folders to compare GK handling.
Generates one video + JSON per variant per clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import sys


VARIANTS = [
    ("hsv_fused", "hsv", "fused", True),
    ("hsv_legacy", "hsv", "legacy", True),
    ("dbscan_fused", "dbscan", "fused", True),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from core.clustering import team_clustering_phase

    parser = argparse.ArgumentParser(description="Run clustering variants on clips")
    parser.add_argument("--clips-root", type=Path, default=Path("clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clustering_variants"))
    parser.add_argument("--rfdetr-weights", type=Path,
                        default=Path("weights/rfdetr_player_gk_ref_best_total.pth"))
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    clips = [p for p in args.clips_root.iterdir() if p.is_dir()]
    if not clips:
        raise FileNotFoundError(f"No se encontraron subcarpetas en {args.clips_root}")

    for clip_dir in clips:
        for name, mode, gk_mode, use_gk_class in VARIANTS:
            out_dir = args.output_dir / clip_dir.name / name
            out_dir.mkdir(parents=True, exist_ok=True)
            output_video = out_dir / f"{clip_dir.name}_{name}.mp4"
            output_json = out_dir / f"{clip_dir.name}_{name}.json"

            print(f"[RUN] {clip_dir.name} | {name}")
            track_to_team, goalie_tracks, referee_tracks = team_clustering_phase.run(
                mode=mode,
                conf_threshold=args.conf,
                allow_unknown=True,
                n_clusters=args.k,
                use_gk_class=use_gk_class,
                gk_assignment_mode=gk_mode,
                cluster_referee=False,
                model_path=str(args.rfdetr_weights),
                sequence_folder=str(clip_dir),
                output_video=str(output_video),
                debug_dir=str(out_dir / "team_clustering_debug"),
                detections_json_path=None,
            )

            assignments = []
            gk_set = set(goalie_tracks)
            ref_set = set(referee_tracks)
            for tid, team_id in track_to_team.items():
                role = "player"
                if tid in gk_set:
                    role = "goalkeeper"
                elif tid in ref_set:
                    role = "referee"
                assignments.append({"track_id": int(tid), "team_id": int(team_id), "role": role})
            for ref_id in ref_set:
                if ref_id not in track_to_team:
                    assignments.append({"track_id": int(ref_id), "team_id": -2, "role": "referee"})

            output_json.write_text(
                "\n".join([
                    "[",
                    ",\n".join(["  {\"track_id\": %d, \"team_id\": %d, \"role\": \"%s\"}" % (
                        a["track_id"], a["team_id"], a["role"]
                    ) for a in assignments]),
                    "\n]"
                ]),
                encoding="utf-8",
            )

            print(f"[OK] {output_video}")


if __name__ == "__main__":
    main()
