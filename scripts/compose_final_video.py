# scripts/compose_final_video.py
"""
Compone el VIDEO FINAL de una secuencia (todos los modelos en un solo mp4)
dentro de su propia carpeta: `outputs/<seq>/<seq>_FINAL.mp4`:

  seccion 1: <seq>_2d_map.mp4  (deteccion + tracking + equipos + dorsales
             + balon + minimapa metrico sincronizado)
  seccion 2: clips de scanning por recepcion (receptor resaltado + panel de
             orientacion), si existen.

Limpieza:
  --archive_debug  mueve mp4 intermedios/debug a `<seq>/_archive/`
  --tidy           BORRA el ruido no re-usable por el pipeline: crops de
                   ventanas de scanning, mascaras SAM2, carpetas de debug,
                   annotation packs y _archive. Conserva los MP4 importantes
                   (FINAL, 2d_map, tacticalvision_refined, event_clips) y los
                   datos minimos que alimentan el cache incremental
                   (JSON/CSV/parquet: detections, trajectory, teams, jersey,
                   calibracion, metricas, scanning).

  python scripts/compose_final_video.py --video_id COL-POR-2026 --tidy
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


# ruido re-generable que --tidy BORRA (no lo usa ninguna fase del pipeline)
TIDY_DELETE_DIRS = [
    "team_clustering_debug", "*_sam2_masks", "_archive",
    "scanning/windows", "scanning/annotation_pack", "scanning/reports",
    "jersey_crops*", "gt_metrics", "jersey_eval*",
]


def tidy(seq_dir: Path) -> None:
    import glob
    deleted = 0
    for pat in TIDY_DELETE_DIRS:
        for p in glob.glob(str(seq_dir / pat)):
            pp = Path(p)
            if pp.is_dir():
                shutil.rmtree(pp)
                deleted += 1
            elif pp.is_file():
                pp.unlink()
                deleted += 1
    if deleted:
        print(f"[tidy] {deleted} carpetas/archivos de ruido borrados en {seq_dir}")


def compose(video_id: str, outputs_root: str = "outputs", fps: int = 25,
            overwrite: bool = False, archive_debug: bool = False,
            do_tidy: bool = False) -> Path:
    seq_dir = Path(outputs_root) / video_id
    out = seq_dir / f"{video_id}_FINAL.mp4"

    main = seq_dir / f"{video_id}_2d_map.mp4"
    if not main.exists():
        raise FileNotFoundError(
            f"No existe {main}. Corre el pipeline con --render primero.")

    scan_dir = resolve_scanning_dir(outputs_root, video_id)
    # clips con etiqueta del MODELO entrenado > clips con etiqueta heuristica
    clips_dir = scan_dir / "event_clips_model"
    if not clips_dir.is_dir():
        clips_dir = scan_dir / "event_clips"
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
    if do_tidy:
        tidy(seq_dir)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--archive_debug", action="store_true")
    ap.add_argument("--tidy", action="store_true")
    a = ap.parse_args()
    compose(a.video_id, a.outputs_root, a.fps, a.overwrite, a.archive_debug,
            a.tidy)
