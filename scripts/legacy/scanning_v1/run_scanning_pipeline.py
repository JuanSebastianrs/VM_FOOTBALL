# scripts/scanning/run_scanning_pipeline.py
"""
Corre el pipeline de scanning sobre una secuencia ya procesada por el pipeline
tactico (detecciones + trayectoria + equipos + calibracion) y exporta:

  outputs/scanning/<video_id>/player_orientation.parquet
  outputs/scanning/<video_id>/scanning_events.parquet
  outputs/scanning/<video_id>/reception_events.parquet
  (opcional) debug_video.mp4 / minimap_scanning.mp4

Ejemplo:
  python scripts/scanning/run_scanning_pipeline.py \
      --video_id SNMOT-148 \
      --detections outputs/SNMOT-148/SNMOT-148_detections.json \
      --trajectory outputs/SNMOT-148/SNMOT-148_trajectory.json \
      --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
      --calibration outputs/SNMOT-148/calibration_hinv.json \
      --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
      --visualize
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning import load_config, load_sequence              # noqa: E402
from core.scanning.exporter import Exporter                       # noqa: E402
from core.scanning.pipeline import ScanningPipeline               # noqa: E402


def get_args():
    p = argparse.ArgumentParser(description="Pipeline de visual scanning")
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", default=None,
                   help="Dir con img1/ (requerido para keypoints y video overlay)")
    p.add_argument("--config", default=None)
    p.add_argument("--keypoint-backend", dest="keypoint_backend",
                   choices=["yolo_pose", "mediapipe", "hybrid"], default=None,
                   help="Sobrescribe scanning.keypoint_backend del YAML (en memoria, "
                        "no modifica el archivo). Si se omite, usa el valor del config.")
    p.add_argument("--output_dir", default=None,
                   help="Directorio de salida EXACTO. Si se omite, usa "
                        "<data.output_dir>/<video_id>.")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--no_pose", action="store_true",
                   help="Desactiva keypoints (orientacion solo por movimiento)")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--also_csv", action="store_true",
                   help="Exporta tambien CSV ademas del formato principal.")
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    sc = config["scanning"]
    # Override de backend en memoria (no toca el YAML en disco)
    if args.keypoint_backend:
        sc["keypoint_backend"] = args.keypoint_backend
    print(f"[scanning] keypoint_backend={sc.get('keypoint_backend', 'yolo_pose')}")
    # --output_dir, si se da, es la ruta EXACTA de salida (no se le anade video_id)
    out_dir = args.output_dir or os.path.join(config["data"]["output_dir"], args.video_id)

    print(f"[scanning] cargando secuencia {args.video_id} ...")
    seq = load_sequence(
        args.video_id, args.detections, args.trajectory, args.team_assignments,
        args.calibration, sequence_dir=args.sequence_dir, fps=sc.get("fps", 25.0))
    print(f"  frames={len(seq.frame_ids)} tracks={len(seq.track_ids())} "
          f"homografia={'si' if seq.homography.available else 'no'} "
          f"imagenes={len(seq.image_paths)}")

    enable_pose = (not args.no_pose) and len(seq.image_paths) > 0
    if not enable_pose and not args.no_pose:
        print("  [aviso] sin imagenes -> se desactiva pose (solo movimiento).")

    pipe = ScanningPipeline(config, enable_pose=enable_pose)
    records, receptions, scans = pipe.run(seq, max_frames=args.max_frames)

    ext = "csv" if args.format == "csv" else "parquet"
    exporter = Exporter(out_dir)
    p1 = exporter.export_orientation(records, os.path.join(out_dir, f"player_orientation.{ext}"))
    p2 = exporter.export_scanning([m.to_dict() for m in scans],
                                  os.path.join(out_dir, f"scanning_events.{ext}"))
    p3 = exporter.export_receptions([e.to_dict() for e in receptions],
                                    os.path.join(out_dir, f"reception_events.{ext}"))
    print(f"[scanning] exportado:\n  {p1}\n  {p2}\n  {p3}")
    if args.also_csv and ext != "csv":
        c1 = exporter.export_orientation(records, os.path.join(out_dir, "player_orientation.csv"))
        c2 = exporter.export_scanning([m.to_dict() for m in scans],
                                      os.path.join(out_dir, "scanning_events.csv"))
        print(f"  + csv:\n  {c1}\n  {c2}")
    n_scan = sum(1 for m in scans if m.scan_count > 0)
    print(f"  recepciones={len(receptions)}  con_scan={n_scan}")

    if args.visualize:
        import pandas as pd
        from core.scanning.visualization import ScanningVisualizer
        df = pd.read_parquet(p1) if ext == "parquet" else pd.read_csv(p1)
        events = [m.to_dict() for m in scans]
        viz = ScanningVisualizer(sc)
        if seq.image_paths:
            vp = viz.render_video(seq, df, events,
                                  os.path.join(out_dir, "debug_video.mp4"),
                                  max_frames=args.max_frames)
            print(f"  video: {vp}")
        mp = viz.render_minimap(seq, df, events,
                                os.path.join(out_dir, "minimap_scanning.mp4"),
                                max_frames=args.max_frames)
        print(f"  minimap: {mp}")


if __name__ == "__main__":
    main()
