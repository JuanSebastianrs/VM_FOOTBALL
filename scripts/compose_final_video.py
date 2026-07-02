# scripts/compose_final_video.py
"""
Compone el VIDEO FINAL de una secuencia (todos los modelos en un solo mp4) y
lo publica en `outputs/final/<seq>_FINAL.mp4`:

  seccion 1: <seq>_2d_map.mp4  (deteccion + tracking + equipos + dorsales
             + balon + minimapa metrico sincronizado)
  seccion 2: clips de scanning por recepcion (receptor resaltado + panel de
             orientacion), si existen.

Con --archive_debug mueve los mp4 intermedios/debug de la secuencia
(yolo_raw, team_clustering, variantes viejas) a `<seq>/_archive/` para que
el output visible quede limpio.

  python scripts/compose_final_video.py --video_id COL-POR-2026
  python scripts/compose_final_video.py --video_id SNMOT-148 --archive_debug
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.scanning_v2.paths import resolve_scanning_dir  # noqa: E402

# mp4 de depuracion / intermedios que NO forman parte del output final
DEBUG_VIDEO_PATTERNS = [
    "*_yolo_raw.mp4", "*_team_clustering.mp4", "*_2d_map_v*.mp4",
    "*_2d_map_bidirectional*.mp4", "*_2d_map_lie*.mp4", "*_minimap_phase*.mp4",
    "*_jersey_overlay*.mp4",
]


def _run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def compose(video_id: str, outputs_root: str = "outputs", fps: int = 25,
            overwrite: bool = False, archive_debug: bool = False) -> Path:
    seq_dir = Path(outputs_root) / video_id
    final_dir = Path(outputs_root) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    out = final_dir / f"{video_id}_FINAL.mp4"

    main = seq_dir / f"{video_id}_2d_map.mp4"
    if not main.exists():
        raise FileNotFoundError(
            f"No existe {main}. Corre el pipeline con --render primero.")

    clips_dir = resolve_scanning_dir(outputs_root, video_id) / "event_clips"
    clips = sorted(clips_dir.glob("*_video.mp4")) if clips_dir.is_dir() else []

    if out.exists() and not overwrite:
        print(f"[final] ya existe {out} (usa --overwrite)")
    else:
        # canvas = resolucion del video principal
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(main)],
            capture_output=True, text=True, check=True)
        w, h = (int(x) for x in probe.stdout.strip().split(","))

        inputs, filters, labels = ["-i", str(main)], [f"[0:v]fps={fps},setsar=1[v0]"], ["[v0]"]
        for i, c in enumerate(clips, start=1):
            inputs += ["-i", str(c)]
            filters.append(
                f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1[v{i}]")
            labels.append(f"[v{i}]")
        filters.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[out]")
        _run(["ffmpeg", "-y", "-v", "error", *inputs,
              "-filter_complex", ";".join(filters), "-map", "[out]",
              "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out])
        print(f"[final] {out} ({len(clips)} clips de scanning anexados)")

    if archive_debug:
        arch = seq_dir / "_archive"
        moved = 0
        for pat in DEBUG_VIDEO_PATTERNS:
            for f in seq_dir.glob(pat):
                arch.mkdir(exist_ok=True)
                shutil.move(str(f), str(arch / f.name))
                moved += 1
        if moved:
            print(f"[final] {moved} videos de debug -> {arch}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--archive_debug", action="store_true")
    a = ap.parse_args()
    compose(a.video_id, a.outputs_root, a.fps, a.overwrite, a.archive_debug)
