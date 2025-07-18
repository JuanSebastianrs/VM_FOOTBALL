import os
import configparser
from pathlib import Path

def procesar_secuencia(sequence_dir):
    gt_path = sequence_dir / "gt" / "gt.txt"
    gameinfo_path = sequence_dir / "gameinfo.ini"
    seqinfo_path = sequence_dir / "seqinfo.ini"
    output_dir = sequence_dir / "labels"

    # Saltar si ya existen archivos de labels
    if output_dir.exists() and any(output_dir.glob("*.txt")):
        print(f"Saltado (ya procesado): {sequence_dir.name}")
        return

    if not gt_path.exists() or not gameinfo_path.exists() or not seqinfo_path.exists():
        print(f"Archivos faltantes en: {sequence_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # === Leer resolución del video ===
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    try:
        img_width = int(config["Sequence"]["imWidth"])
        img_height = int(config["Sequence"]["imHeight"])
    except KeyError as e:
        print(f"Error de resolución en {sequence_dir.name}: {e}")
        return

    # === Mapeo de tracklets a clase y equipo ===
    tracklet_class_map = {}
    tracklet_team_map = {}

    class_name_to_id = {
        "player team left": 0,
        "player team right": 1,
        "goalkeeper team left": 2,
        "goalkeeper team right": 3,
        "referee": 4,
        "ball": 5
    }

    with open(gameinfo_path, "r") as f:
        for line in f:
            if line.startswith("trackletID_"):
                parts = line.strip().split("=")
                track_id = int(parts[0].split("_")[1])
                class_info = parts[1].split(";")[0].strip()
                class_id = class_name_to_id.get(class_info, -1)
                if class_id != -1:
                    tracklet_class_map[track_id] = class_id
                    if "left" in class_info:
                        tracklet_team_map[track_id] = "left"
                    elif "right" in class_info:
                        tracklet_team_map[track_id] = "right"
                    else:
                        tracklet_team_map[track_id] = "other"

    # === Leer anotaciones y crear labels por frame ===
    frame_annotations = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id, track_id, x, y, w, h = map(float, parts[:6])
            frame_id = int(frame_id)
            track_id = int(track_id)

            if track_id not in tracklet_class_map:
                continue

            class_id = tracklet_class_map[track_id]
            team_label = tracklet_team_map[track_id]

            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {track_id} # team:{team_label}"
            frame_file = output_dir / f"{frame_id:06}.txt"
            frame_annotations.setdefault(frame_file, []).append(yolo_line)

    # === Guardar labels por frame ===
    for path, lines in frame_annotations.items():
        with open(path, "w") as f:
            f.write("\n".join(lines))

    print(f"{sequence_dir.name} procesado: {len(frame_annotations)} frames.")

# === MAIN ===
if __name__ == "__main__":
        
    root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "data" / "tracking" / "SoccerNet" / "tracking"
    for split in ["train", "test", "challenge"]:
        split_path = root_path / split / split
        print(f"\n Procesando split: {split.upper()} ({split_path})")
        for sequence in sorted(split_path.glob("SNMOT-*")):
            if sequence.is_dir():
                procesar_secuencia(sequence)
