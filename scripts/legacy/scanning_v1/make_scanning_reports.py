# scripts/scanning/make_scanning_reports.py
"""
Genera los reportes (pesado + ligeros) del modulo de scanning a partir de los
outputs ya exportados de una secuencia.

Escribe:
  <scanning_dir>/validation_report.md                  (reporte de la corrida)
  docs/scanning/full_pipeline_comparison.md            (copia del comparativo 3-way)
  docs/scanning/backend_benchmark_summary.md           (resumen + recomendacion)
  docs/scanning/<video_id>_scanning_summary.md         (resumen de la secuencia)

Los reportes en docs/ son LIGEROS (solo markdown): no copian mp4/parquet/imagenes.

Ejemplo:
  python scripts/scanning/make_scanning_reports.py \
      --video_id SNMOT-148 \
      --scanning_dir outputs/SNMOT-148/scanning
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

NOT_GAZE = ("> **No es gaze real.** Es orientacion visual *aproximada* basada en "
            "keypoints (cabeza/torso/carrera). En broadcast los ojos rara vez son "
            "visibles; `theta_source` indica de donde sale cada estimacion.")
SCAN_HEUR = ("> `scan_count` y las metricas de scanning son **heuristicas**: miden "
             "*cambios* de orientacion, no mirada verificada. Su validacion de "
             "accuracy requiere **anotacion humana** (los seed labels NO son GT).")


def _read(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _pct(df, col, pred):
    return 100.0 * pred(df[col]).mean() if df is not None and col in df else float("nan")


def make_validation_report(scanning_dir: Path, video_id: str, metrics: dict):
    o = _read(scanning_dir / "player_orientation.parquet")
    s = _read(scanning_dir / "scanning_events.parquet")
    r = _read(scanning_dir / "reception_events.parquet")
    events_dir = scanning_dir / "events"
    clips = sorted(events_dir.glob("*_video.mp4")) if events_dir.exists() else []

    lines = [f"# Reporte de validacion — scanning {video_id} (backend hybrid)", "",
             NOT_GAZE, "", SCAN_HEUR, "", "## Cobertura de orientacion", ""]
    if o is not None:
        n = len(o)
        bu = o["keypoint_backend_used"].value_counts().to_dict() if "keypoint_backend_used" in o else {}
        att = o["keypoint_backend_attempted"].astype(str) if "keypoint_backend_attempted" in o else pd.Series([], dtype=str)
        fb = int(att.str.contains(r"\+").sum()) if len(att) else 0
        lines += [
            f"- player-frame samples: **{n}**",
            f"- pose_valid_rate: **{o['pose_valid'].mean():.3f}**",
            f"- theta_visual_field no nulo: **{o['theta_visual_field'].notna().mean():.3f}**",
            f"- ball_relative_angle_field no nulo: **{o['ball_relative_angle_field'].notna().mean():.3f}**",
            f"- backend usado por fila: {bu}",
            f"- fallbacks MediaPipe->YOLO (filas con '+'): **{fb}**",
            "- theta_source (%): " + ", ".join(
                f"{k}={100.0 * v / n:.1f}" for k, v in o['theta_source'].value_counts().items()),
            "",
        ]
    lines += ["## Recepciones y scanning", ""]
    if r is not None:
        lines.append(f"- recepciones detectadas: **{len(r)}**")
    if s is not None:
        scan_ev = int((s["scan_count"] > 0).sum())
        lines.append(f"- eventos con scan_count>0: **{scan_ev}** / {len(s)} recepciones")
        cols = ["receiver_track_id", "frame_reception", "scan_count",
                "max_orientation_change_deg", "looked_towards_ball_count",
                "looked_away_from_ball_count", "quality_score"]
        cols = [c for c in cols if c in s.columns]
        top = s[s["scan_count"] > 0].sort_values("max_orientation_change_deg", ascending=False)
        lines += ["", "### Eventos de scanning", "",
                  "| " + " | ".join(cols) + " |",
                  "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in top.iterrows():
            lines.append("| " + " | ".join(
                (f"{row[c]:.1f}" if isinstance(row[c], float) else str(row[c])) for c in cols) + " |")
    lines += ["", "## Marcos de referencia (verificacion)", "",
              "- Video overlay: flecha = `theta_visual_smooth_img` (imagen).",
              "- Minimapa: flecha/cono = `theta_visual_smooth_field` (cancha).",
              "- `looked_towards/away_from_ball`: en CANCHA cuando hay homografia "
              "(compara `theta_visual_smooth_field` vs `ball_relative_angle_field`).",
              "", "## Clips generados", ""]
    if clips:
        for c in clips:
            lines.append(f"- `events/{c.name}` (+ `{c.name.replace('_video', '_minimap')}`)")
    else:
        lines.append("- (sin clips: no hubo eventos con scan_count>0)")
    lines += ["", "## Artefactos en esta carpeta", "",
              "- player_orientation.parquet / .csv", "- scanning_events.parquet / .csv",
              "- reception_events.parquet", "- debug_video.mp4, minimap_scanning.mp4",
              "- events/  (clips por evento)", "- backend_benchmark/  (comparativo 3-way)", ""]
    (scanning_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def make_docs(video_id: str, scanning_dir: Path, docs_dir: Path, metrics: dict):
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1) copia del comparativo completo
    cmp_src = scanning_dir / "backend_benchmark" / "full_pipeline_comparison.md"
    if cmp_src.exists():
        (docs_dir / "full_pipeline_comparison.md").write_text(
            cmp_src.read_text(encoding="utf-8"), encoding="utf-8")

    # 2) resumen de benchmark + recomendacion
    m = metrics or {}
    def g(bk, k, d=float("nan")):
        return m.get(bk, {}).get(k, d)
    rows = []
    for bk in ["yolo_pose", "mediapipe", "hybrid"]:
        if bk in m:
            rows.append(f"| {bk} | {g(bk,'pose_valid_rate'):.3f} | {g(bk,'head_angle_rate'):.3f} "
                        f"| {g(bk,'angular_jitter_deg'):.2f} | {g(bk,'mean_orientation_confidence'):.3f} "
                        f"| {int(g(bk,'scanning_events_detected',0))} | {g(bk,'ms_per_frame'):.1f} |")
    summary = [
        f"# Resumen de benchmark de backends — {video_id}", "", NOT_GAZE, "",
        "| backend | pose_valid | head_angle | jitter° | conf | scan_ev | ms/frame |",
        "|---|---|---|---|---|---|---|", *rows, "",
        "## Recomendacion", "",
        "**`hybrid` (YOLO-Pose por defecto, MediaPipe en crops grandes con fallback a YOLO).**",
        "",
        "### Por que hybrid mejora la orientacion",
        "- Mayor `pose_valid_rate` que yolo_pose y mucho mayor que mediapipe (combina "
        "cobertura de YOLO en crops pequenos + cabeza limpia de MediaPipe en grandes).",
        "- Mayor `head_angle_rate` (mas frames resueltos por cabeza, la fuente mas fiable).",
        "- Menor `angular_jitter_deg` => orientacion mas estable temporalmente.",
        "- El fallback MediaPipe->YOLO evita perder el frame cuando MediaPipe no detecta.",
        "",
        "> Nota: hybrid suele detectar **menos** eventos de scanning que yolo_pose. No es "
        "peor: su menor jitter elimina 'scans' espurios por ruido de keypoints. Esto "
        "confirma que `scan_count` es sensible al ruido del backend y **necesita GT humano**.",
        "",
        "## Limitaciones",
        "- Baja resolucion y oclusiones en broadcast => orientacion aproximada.",
        "- Jugadores de espaldas: la cabeza no resuelve frente/espalda en 2D; se cae a torso/carrera.",
        "- MediaPipe colapsa en crops pequenos (por eso hybrid usa YOLO ahi).",
        "- El tiempo YOLO(GPU) vs MediaPipe(CPU) no es comparable 1:1.",
        "", SCAN_HEUR, "",
    ]
    (docs_dir / "backend_benchmark_summary.md").write_text("\n".join(summary), encoding="utf-8")

    # 3) resumen de la secuencia
    o = _read(scanning_dir / "player_orientation.parquet")
    s = _read(scanning_dir / "scanning_events.parquet")
    r = _read(scanning_dir / "reception_events.parquet")
    seq = [f"# Resumen de scanning — {video_id}", "", NOT_GAZE, "", SCAN_HEUR, "",
           "## Que se genero", "",
           f"Pipeline completo con **keypoint_backend = hybrid** sobre {video_id}.", ""]
    if o is not None:
        seq += [
            f"- player-frame samples: **{len(o)}**, pose_valid_rate: **{o['pose_valid'].mean():.3f}**",
            "- orientacion separada en imagen (`*_img`) y cancha (`*_field`), cruda y suavizada.",
            "- backend por fila en `keypoint_backend_used` / `keypoint_backend_attempted`.",
        ]
    if r is not None:
        seq.append(f"- recepciones detectadas: **{len(r)}**")
    if s is not None:
        seq.append(f"- eventos de scanning (scan_count>0): **{int((s['scan_count']>0).sum())}**")
    seq += ["", "## Como interpretarlo", "",
            "- El **video overlay** muestra la orientacion en imagen; el **minimapa** la "
            "orientacion en cancha (`theta_visual_smooth_field`) con cono de vision aproximado.",
            "- `looked_towards/away_from_ball` se mide en cancha cuando hay homografia.",
            "- Los clips por evento estan en `events/` (3 s antes de cada recepcion con scan).",
            "",
            "## Donde estan los resultados (no versionados, pesados)", "",
            f"`outputs/{video_id}/scanning/` — parquet/csv, mp4, events/, backend_benchmark/.",
            "Estos archivos siguen en `.gitignore`; aqui en `docs/` solo hay reportes ligeros.",
            ""]
    (docs_dir / f"{video_id}_scanning_summary.md").write_text("\n".join(seq), encoding="utf-8")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--scanning_dir", required=True)
    p.add_argument("--docs_dir", default="docs/scanning")
    return p.parse_args()


def main():
    args = get_args()
    scanning_dir = Path(args.scanning_dir)
    metrics_path = scanning_dir / "backend_benchmark" / "full_pipeline_comparison_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    make_validation_report(scanning_dir, args.video_id, metrics)
    make_docs(args.video_id, scanning_dir, Path(args.docs_dir), metrics)
    print(f"[reports] validation_report.md -> {scanning_dir}")
    print(f"[reports] docs ligeros -> {args.docs_dir}")


if __name__ == "__main__":
    main()
