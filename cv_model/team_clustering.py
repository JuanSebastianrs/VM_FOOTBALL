# team_clustering_optimized.py
from __future__ import annotations
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import json
import csv
import re
import os
import torch
from typing import Dict, List, Tuple, Optional

# =========================
# Utilidades de imagen/color
# =========================
def suppress_pitch_green(bgr: np.ndarray) -> np.ndarray:
    """Máscara para quitar el pasto (verde) y que el histograma de camiseta no se contamine."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, lower_green, upper_green)
    return cv2.bitwise_not(green)

def crop_torso(image: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
    """Recorta zona de torso dentro de la bbox para dar más peso a la camiseta."""
    x1, y1, x2, y2 = map(int, xyxy)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    top = y1 + int(0.15 * h)
    bottom = y1 + int(0.60 * h)
    left = x1 + int(0.20 * w)
    right = x1 + int(0.80 * w)
    top = max(y1, min(top, y2 - 1))
    bottom = max(top + 1, min(bottom, y2))
    left = max(x1, min(left, x2 - 1))
    right = max(left + 1, min(right, x2))
    return image[top:bottom, left:right]

def color_descriptor(bgr_crop: Optional[np.ndarray], bins: int = 12) -> Optional[np.ndarray]:
    """Histograma HSV (H y S) normalizado con máscara anti-verde."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    mask = suppress_pitch_green(bgr_crop)
    # Si la máscara quedó casi vacía, evita producir un vector ruidoso
    if cv2.countNonZero(mask) < 16:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
    feat = np.concatenate([h_hist.ravel(), s_hist.ravel()]).astype(np.float32)
    s = float(feat.sum())
    if s > 0.0:
        feat /= s
        return feat
    return None

# =========================
# Listado de frames por secuencia (prefijo SNMOT-###)
# =========================
def list_sequence_frames(images_dir: Path, seq_id: str) -> List[Path]:
    pat = re.compile(rf"^{re.escape(seq_id)}_(\d+)\.jpg$", re.IGNORECASE)
    frames: List[Tuple[int, Path]] = []
    for p in images_dir.glob(f"{seq_id}_*.jpg"):
        m = pat.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]

# =========================
# Pipeline de clustering por partido
# =========================
def process_sequence(
    seq_id: str,
    split: str,
    model_path: Path,
    out_dir: Path,
    dataset_root: Path,
    class_player: int = 0,     # YAML: ["player","goalkeeper","referee","ball"]
    tracker_cfg: str = "bytetrack.yaml",
    imgsz: int = 736,
    conf_thres: float = 0.35,
    iou_thres: float = 0.7,
    max_det: int = 60,
    device: str | int = "0",
    empty_cache_every: int = 50,   # menos presión al driver
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    images_path = dataset_root / "images" / split
    frame_paths = list_sequence_frames(images_path, seq_id)
    if not frame_paths:
        raise FileNotFoundError(f"No se encontraron frames {seq_id}_*.jpg en {images_path}")

    # ===== Modelo (no-grad + cudnn autotune) =====
    torch.backends.cudnn.benchmark = True
    use_cuda = (device != "cpu") and torch.cuda.is_available()

    model = YOLO(str(model_path))
    model.to(0 if use_cuda else "cpu")
    if use_cuda:
        # FP16 sólo si hay CUDA
        try:
            model.model.half()
        except Exception:
            pass

    # ===== Inference + Tracking en stream =====
    with torch.inference_mode():
        results = model.track(
            source=[str(p) for p in frame_paths],  # lista ordenada de paths
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            tracker=tracker_cfg,
            device=(0 if use_cuda else "cpu"),
            verbose=False,
            persist=True,      # mantiene IDs
            stream=True,       # generador
            max_det=max_det,
            workers=0          # Windows friendly
        )

        # Acumuladores por track (memoria O(#tracks))
        tracks_sum: Dict[int, np.ndarray] = {}      # sumatoria de features
        tracks_count: Dict[int, int] = defaultdict(int)
        frame_assignments: List[List[dict]] = []

        for fi, res in enumerate(results):
            # Usa la imagen ya cargada por Ultralytics (evita I/O extra)
            img = getattr(res, "orig_img", None)
            if img is None:
                # fallback ultra-defensivo
                try:
                    img = cv2.imread(res.path)
                except Exception:
                    img = None

            if img is None:
                frame_assignments.append([])
                if use_cuda and empty_cache_every and (fi % empty_cache_every == 0):
                    torch.cuda.empty_cache()
                continue

            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                frame_assignments.append([])
                if use_cuda and empty_cache_every and (fi % empty_cache_every == 0):
                    torch.cuda.empty_cache()
                continue

            xyxy = getattr(boxes, "xyxy", None)
            cls = getattr(boxes, "cls", None)
            ids = getattr(boxes, "id", None)

            if xyxy is None or cls is None:
                frame_assignments.append([])
                if use_cuda and empty_cache_every and (fi % empty_cache_every == 0):
                    torch.cuda.empty_cache()
                continue

            xyxy = xyxy.detach().cpu().numpy()
            cls = cls.detach().cpu().numpy().astype(int)

            if ids is not None:
                ids = ids.detach().cpu().numpy().astype(int)
            else:
                ids = np.full(len(cls), -1, dtype=int)

            frame_data: List[dict] = []
            for bb, c, tid in zip(xyxy, cls, ids):
                if c != class_player or tid < 0:
                    continue
                crop = crop_torso(img, bb)
                feat = color_descriptor(crop)
                if feat is None:
                    continue

                # Acumulación online: sum y count (evita guardar listas grandes)
                if tid in tracks_sum:
                    tracks_sum[tid] += feat
                else:
                    tracks_sum[tid] = feat.copy()
                tracks_count[tid] += 1

                x1, y1, x2, y2 = map(float, bb)
                frame_data.append({
                    "track_id": int(tid),
                    "cls": int(c),
                    "bbox": [x1, y1, x2, y2]
                })

            frame_assignments.append(frame_data)

            if use_cuda and empty_cache_every and (fi % empty_cache_every == 0):
                torch.cuda.empty_cache()

    # ===== Pooling (media) por track =====
    track_ids = sorted(tracks_sum.keys())
    if len(track_ids) < 2:
        raise RuntimeError("No hay suficientes jugadores para clusterizar en 2 equipos.")

    X = np.vstack([tracks_sum[tid] / max(1, tracks_count[tid]) for tid in track_ids])

    # ===== KMeans k=2 =====
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    track_to_team = {tid: int(lbl) for tid, lbl in zip(track_ids, labels)}

    # ===== Heurística left/right por frames iniciales =====
    N_init = min(150, len(frame_assignments))
    xs_by_team: Dict[int, List[float]] = defaultdict(list)
    for f in range(N_init):
        for obj in frame_assignments[f]:
            tid = obj["track_id"]
            team = track_to_team.get(tid, None)
            if team is not None:
                x1, y1, x2, y2 = obj["bbox"]
                xs_by_team[team].append(0.5 * (x1 + x2))

    if xs_by_team[0] and xs_by_team[1]:
        mean0, mean1 = float(np.mean(xs_by_team[0])), float(np.mean(xs_by_team[1]))
        if mean0 > mean1:
            # Normaliza para que team 0 quede a la izquierda
            track_to_team = {tid: (1 - team) for tid, team in track_to_team.items()}

    # ===== Guardar mapa track->team =====
    map_csv = out_dir / f"{seq_id}_track_to_team.csv"
    with open(map_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "team_id", "samples"])
        for tid in track_ids:
            w.writerow([tid, track_to_team[tid], tracks_count[tid]])

    # ===== Guardar por frame =====
    per_frame_json = out_dir / f"{seq_id}_frame_assignments.json"
    out_frames: List[dict] = []
    for frame_idx, objs in enumerate(frame_assignments):
        items = []
        for obj in objs:
            tid = obj["track_id"]
            team_id = track_to_team.get(tid, None)
            if team_id is not None:
                items.append({
                    "track_id": tid,
                    "cls": obj["cls"],
                    "team_id": int(team_id),
                    "bbox": obj["bbox"]
                })
        out_frames.append({"frame": frame_idx, "objects": items})
    per_frame_json.write_text(json.dumps(out_frames, indent=2, ensure_ascii=False))

    print(f"OK: {seq_id}")
    print(f"  -> {map_csv}")
    print(f"  -> {per_frame_json}")

# =========================
# MAIN
# =========================
def main() -> None:
    # === Configurables ===
    SECUENCIA = "SNMOT-116"
    SPLIT = "test"
    MODEL_WEIGHTS = Path("runs/train/modelo_yolo11vn_4class/weights/best.pt")
    DATASET_ROOT = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
    OUT_DIR = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "outputs" / "team_clustering"

    # Memoria (Windows)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    process_sequence(
        seq_id=SECUENCIA,
        split=SPLIT,
        model_path=MODEL_WEIGHTS,
        out_dir=OUT_DIR,
        dataset_root=DATASET_ROOT,
        class_player=0,           # "player"
        tracker_cfg="bytetrack.yaml",
        imgsz=736,                # baja a 704/640 si te da OOM
        conf_thres=0.35,
        iou_thres=0.7,
        max_det=60,
        device="0" if torch.cuda.is_available() else "cpu",
        empty_cache_every=75,     # menos presión en VRAM/driver sin penalizar FPS
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
