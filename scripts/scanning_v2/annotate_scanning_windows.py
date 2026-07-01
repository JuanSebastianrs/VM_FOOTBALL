# scripts/scanning_v2/annotate_scanning_windows.py
"""
FASE 10 — Pack de anotacion humana de ventanas de scanning.

Genera outputs/.../annotation_pack/ con:
  clips/                         (copias/links de los clips por evento)
  annotation_template.csv        (una fila por evento a anotar)
  README_annotation_guidelines.md

El CSV sigue el esquema de data/annotations/scanning_windows_gt.csv. La columna
scan_label_gt empieza vacia: la llena el anotador humano (NO se rellena con la
prediccion heuristica para no sesgar).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

GT_COLUMNS = [
    "sample_id", "event_id", "video_id", "frame_start", "frame_reception",
    "frame_end", "receiver_track_id", "receiver_role", "team_id", "event_source",
    "clip_path", "scan_label_gt", "head_turn_count_gt", "turn_direction_gt",
    "visibility", "confidence", "notes",
]

# campos que llena el anotador humano (se preservan al regenerar el template)
HUMAN_FIELDS = ["scan_label_gt", "head_turn_count_gt", "turn_direction_gt",
                "visibility", "confidence", "notes"]

GUIDELINES = """# Guia de anotacion — head-turn / visual scanning

Objetivo: marcar, viendo el clip de 3 s antes de la recepcion, si el jugador
**receptor** giro la cabeza para escanear el campo. Esto NO es medir su mirada
exacta; es juzgar si hubo un head-turn observable.

## Que CUENTA como head-turn (scan_label_gt = 1)
- Giro claro de la cabeza a un lado y/o de vuelta, antes de recibir.
- Varias miradas (barrido) sobre los hombros.

## Que NO cuenta (scan_label_gt = 0)
- Mirar fijo al balon todo el tiempo.
- Micro-movimientos o vibracion de la deteccion (no es head-turn real).
- Giro del cuerpo entero sin girar la cabeza (anotar 0 para head-turn).

## Campos
- scan_label_gt: 0 = no / 1 = si.
- head_turn_count_gt: numero de giros claros (0,1,2,...).
- turn_direction_gt: left | right | both | unclear.
- visibility: high | medium | low (resolucion/oclusion de la cabeza).
- confidence: high | medium | low (que tan seguro estas de tu etiqueta).

## Casos especiales
- Camara muy lejana / cabeza diminuta: visibility = low; si no puedes juzgar,
  scan_label_gt vacio o confidence = low.
- Oclusion por otro jugador: anota lo que veas; visibility = low.
- Caso dudoso: turn_direction_gt = unclear y confidence = low.
- El receptor marcado nunca debe ser un arbitro; si lo parece, anota en notes.
"""


def _existing_labels(master_path: Path, video_id: str) -> dict:
    """Labels humanos ya escritos en el master, por event_id (de este video)."""
    if not master_path.exists():
        return {}
    try:
        m = pd.read_csv(master_path)
    except Exception:
        return {}
    if "event_id" not in m.columns:
        return {}
    if "video_id" in m.columns:
        m = m[m["video_id"].astype(str) == str(video_id)]
    out = {}
    for _, r in m.drop_duplicates("event_id").iterrows():
        out[str(r["event_id"])] = {f: r.get(f) for f in HUMAN_FIELDS if f in m.columns}
    return out


def build_annotation_pack(out_dir, video_id,
                          master_path="data/annotations/scanning_windows_gt.csv") -> Path:
    out = Path(out_dir)
    events = pd.read_parquet(out / "pass_reception_events.parquet")
    events = events.drop_duplicates("event_id")     # nunca duplicar event_id
    pack = out / "annotation_pack"
    (pack / "clips").mkdir(parents=True, exist_ok=True)

    # copiar clips de video si existen
    clips_src = out / "event_clips"
    if clips_src.exists():
        for mp4 in clips_src.glob("*_video.mp4"):
            shutil.copy2(mp4, pack / "clips" / mp4.name)

    try:
        scanning = pd.read_parquet(out / "scanning_events.parquet").set_index("event_id")
    except Exception:
        scanning = None

    master = Path(master_path)
    master.parent.mkdir(parents=True, exist_ok=True)
    prior = _existing_labels(master, video_id)      # preservar labels humanos

    rows = []
    for i, ev in enumerate(events.to_dict("records")):
        eid = str(ev["event_id"])
        ws = (int(scanning.loc[eid, "window_start"]) if scanning is not None and eid in scanning.index
              else int(ev["frame_reception"]) - 75)
        ws = max(0, int(ws))                         # frame_start nunca negativo
        clip_rel = f"clips/{eid}_video.mp4"
        clip = clip_rel if (pack / clip_rel).exists() else ""
        row = {
            "sample_id": i, "event_id": eid, "video_id": video_id,
            "frame_start": ws, "frame_reception": int(ev["frame_reception"]),
            "frame_end": int(ev["frame_reception"]),
            "receiver_track_id": ev["receiver_track_id"],
            "receiver_role": ev.get("receiver_role"), "team_id": ev.get("receiver_team_id"),
            "event_source": ev.get("source"), "clip_path": clip,
        }
        # preservar etiquetas humanas previas (si existen) o vacio
        prev = prior.get(eid, {})
        for f in HUMAN_FIELDS:
            v = prev.get(f, "")
            row[f] = "" if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        rows.append(row)
    df = pd.DataFrame(rows, columns=GT_COLUMNS)
    df.to_csv(pack / "annotation_template.csv", index=False)
    (pack / "README_annotation_guidelines.md").write_text(GUIDELINES, encoding="utf-8")

    _sync_master(master, df, video_id)
    print(f"[annotation] pack -> {pack}  ({len(df)} ventanas; "
          f"labels preservados: {sum(1 for r in rows if str(r['scan_label_gt']) not in ('', 'nan'))})")
    return pack


def _sync_master(master: Path, df: pd.DataFrame, video_id: str):
    """Actualiza el master: reemplaza las filas de ESTE video por las actuales
    (preservando labels), conserva las de otros videos. Evita quedar desactualizado."""
    if master.exists():
        try:
            old = pd.read_csv(master)
        except Exception:
            old = pd.DataFrame(columns=GT_COLUMNS)
    else:
        old = pd.DataFrame(columns=GT_COLUMNS)
    for c in GT_COLUMNS:
        if c not in old.columns:
            old[c] = ""
    other = old[old["video_id"].astype(str) != str(video_id)] if "video_id" in old.columns else old
    merged = pd.concat([other[GT_COLUMNS], df[GT_COLUMNS]], ignore_index=True)
    # deduplicar por la clave logica (video_id, event_id), NO solo event_id
    merged = merged.drop_duplicates(["video_id", "event_id"], keep="last")
    merged.to_csv(master, index=False)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    a = get_args()
    build_annotation_pack(a.output_dir, a.video_id)


if __name__ == "__main__":
    main()
