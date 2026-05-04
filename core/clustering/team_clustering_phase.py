"""
EvaluaciÃ³n end-to-end: RF-DETR + ByteTrack + TeamClassifier.

Pipeline:
    1. RF-DETR detecta jugadores por frame  â†’  sv.Detections sin IDs
    2. sv.ByteTrack asigna track_id persistentes por jugador
    3. Por cada track se acumula el descriptor HSV medio de la camiseta
    4. Al final del video, TeamClassifier.fit() agrupa en K clusters
    5. Se genera un video anotado con jugadores coloreados por equipo

Salida:
    outputs/visualizations/SNMOT-116_team_clustering.mp4

Uso:
    conda activate dl
    python eval_team_clustering.py [--mode hsv|dbscan] [--k 2] [--conf 0.40]
                                   [--no-use-gk-class]
                                   [--gk-assignment-mode legacy|fused]
                                   [--cluster-referee]
"""

import argparse
import csv
import os
import sys
import glob
import time
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import supervision as sv
from tqdm import tqdm

# Asegurar que el script puede importar 'core' y 'rfdetr' desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rfdetr import RFDETRBase
from core.clustering.team_classifier import TeamClassifier

# â”€â”€ ConfiguraciÃ³n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_PATH      = "models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth"
SEQUENCE_FOLDER = "data/test_sequences/SNMOT-116"
OUTPUT_VIDEO    = "outputs/visualizations/SNMOT-116_team_clustering.mp4"
RESOLUTION      = 448       # debe coincidir con el entrenamiento
FPS             = 25
MIN_TRACK_FRAMES = 8    # tracks con menos frames se ignoran en el clustering
MIN_GK_TRACK_FRAMES = 20
MIN_REF_TRACK_FRAMES = 12
MAX_REFEREE_TRACKS = 2
REF_HSV_OUTLIER_THRESHOLD = 0.50
REF_MIN_VOTE_RATIO = 0.05   # Un track con >=5% de votos como referee Y >=REF_MIN_ABS_VOTES es candidato
REF_MIN_ABS_VOTES = 15      # Mínimo de votos absolutos como referee para ser candidato soft

GK_SOFT_MIN_RATIO = 0.30
GK_SOFT_MIN_VOTES = 8
GK_EDGE_X_RATIO = 0.14
GK_MIN_EDGE_PERSISTENCE = 0.30
GK_NEIGHBOR_K = 4
GK_MIN_SCORE = 0.40
GK_MIN_VOTE_COUNT = 24
GK_MIN_VOTE_MARGIN = 0.12
GK_MAX_STD_X_RATIO = 0.08
GK_MAX_STD_Y_RATIO = 0.12
GK_COMBINED_VOTE_WEIGHT = 0.60
GK_SIDE_MIN_COMBINED_DELTA = 0.12
GK_SIDE_MIN_VOTE_MARGIN_DELTA = 0.10
GK_WINDOW_FRAMES = 220
GK_UPDATE_EVERY = 40
GK_SWITCH_CONFIRM = 2
GK_MIN_WINDOW_SEEN = 10
GK_MIN_WINDOW_VOTES = 12
TORSO_TOP_RATIO = 0.12
TORSO_BOTTOM_RATIO = 0.50
TORSO_LEFT_RATIO = 0.22
TORSO_RIGHT_RATIO = 0.78
DEBUG_DIR = "outputs/team_clustering_debug"
GK_ASSIGNMENT_MODES = ("legacy", "fused")


def _infer_gk_class_ids(observed_class_ids: set[int]) -> set[int]:
    """
    Inferir IDs de portero segÃºn taxonomÃ­a observada en inferencia.

    Casos soportados:
        - {0,1}          -> 2 clases RF-DETR (player, goalkeeper)
        - {0,1,2}        -> 3 clases RF-DETR (player, goalkeeper, referee)
        - incluye {2,3}  -> taxonomÃ­a legacy de 6 clases
        - {0}            -> detector unificado sin separaciÃ³n de GK
    """
    if not observed_class_ids:
        return set()
    if observed_class_ids.issubset({0, 1}) and 1 in observed_class_ids:
        return {1}
    if observed_class_ids.issubset({0, 1, 2}) and 1 in observed_class_ids:
        # TaxonomÃ­a moderna: 0=player, 1=goalkeeper, 2=referee.
        return {1}
    gk_ids = {cid for cid in observed_class_ids if cid in (2, 3)}
    return gk_ids


def _infer_ref_class_ids(observed_class_ids: set[int], gk_class_ids: set[int]) -> set[int]:
    """
    Inferir IDs de Ã¡rbitro segÃºn taxonomÃ­a observada.

    Casos soportados:
        - 3 clases modernas {0,1,2}: referee=2 (si 1 fue inferido como GK)
        - legacy 6 clases: referee=4
    """
    if 4 in observed_class_ids:
        return {4}
    if 2 in observed_class_ids and 1 in gk_class_ids:
        return {2}
    return set()


def _majority_role_by_track(
    track_class_counts: dict[int, dict[int, int]],
    gk_class_ids: set[int],
    ref_class_ids: set[int],
) -> dict[int, str]:
    """Inferir rol por track usando mayorÃ­a temporal de class_id."""
    role_by_tid: dict[int, str] = {}
    for tid, counts in track_class_counts.items():
        if not counts:
            continue
        ref_votes = sum(v for cid, v in counts.items() if cid in ref_class_ids)
        gk_votes = sum(v for cid, v in counts.items() if cid in gk_class_ids)
        player_votes = max(0, sum(counts.values()) - gk_votes - ref_votes)

        if ref_votes >= max(gk_votes, player_votes):
            role_by_tid[tid] = "referee"
        elif gk_votes >= max(ref_votes, player_votes):
            role_by_tid[tid] = "goalkeeper"
        else:
            role_by_tid[tid] = "player"
    return role_by_tid


def build_team_style(n_clusters: int) -> tuple[dict[int, tuple[int, int, int]], dict[int, str]]:
    """Construye paleta de colores y labels para K clusters."""
    base_colors = [
        (219, 152, 52),   # azul
        (37, 37, 213),    # rojo
        (70, 190, 80),    # verde
        (33, 180, 240),   # amarillo
        (180, 80, 180),   # magenta
    ]
    team_colors = {i: base_colors[i % len(base_colors)] for i in range(n_clusters)}
    team_labels = {i: f"Team {chr(65 + i)}" for i in range(n_clusters)}
    team_colors[-1] = (160, 160, 160)
    team_labels[-1] = "?"
    team_colors[-2] = (255, 255, 255)
    team_labels[-2] = "REF"
    return team_colors, team_labels

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_frames(folder: str) -> list[str]:
    # Intentar cargar directo desde la carpeta (legacy)
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    
    # Si no hay, intentar desde la subcarpeta img1 (pipeline estÃ¡ndar de MOT)
    if not paths:
        paths = sorted(glob.glob(os.path.join(folder, "img1", "*.jpg")))
        
    if not paths:
        raise FileNotFoundError(f"No se encontraron .jpg en {folder} ni en {folder}/img1")
        
    return paths


def draw_box(frame, x1, y1, x2, y2, color, label: str, conf: float):
    """Dibuja bbox + label con fondo de color sÃ³lido."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _weighted_center(hist: np.ndarray, low: float, high: float) -> float:
    if hist.size == 0:
        return 0.0
    centers = np.linspace(low, high, hist.size, endpoint=False) + (high - low) / (2.0 * hist.size)
    total = float(np.sum(hist))
    if total <= 1e-9:
        return 0.0
    return float(np.sum(hist * centers) / total)


def _hue_to_name(h: float) -> str:
    if h < 10 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 36:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    return "purple"


# â”€â”€ Pipeline principal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run(
    mode: str = "hsv",
    conf_threshold: float = 0.40,
    allow_unknown: bool = True,
    n_clusters: int = 2,
    use_gk_class: bool = True,
    gk_assignment_mode: str = "fused",
    cluster_referee: bool = False,
    model_path: str = MODEL_PATH,
    sequence_folder: str = SEQUENCE_FOLDER,
    output_video: str = OUTPUT_VIDEO,
    debug_dir: str = DEBUG_DIR,
    detections_json_path: str | None = None,
):
    if gk_assignment_mode not in GK_ASSIGNMENT_MODES:
        raise ValueError(
            f"gk_assignment_mode debe ser uno de {GK_ASSIGNMENT_MODES}, se recibiÃ³ '{gk_assignment_mode}'"
        )
    # --- Decidir modo de detección ---
    use_json_detections = detections_json_path is not None and os.path.exists(str(detections_json_path))

    if not use_json_detections:
        # Modo standalone: cargar modelo RF-DETR propio
        from rfdetr.detr import RFDETRBase  # type: ignore[import]
        print(f"Cargando RF-DETR ({RESOLUTION}px)...")
        model = RFDETRBase(pretrain_weights=model_path, resolution=RESOLUTION)
        tracker = sv.ByteTrack(
            track_activation_threshold=conf_threshold,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=FPS,
            minimum_consecutive_frames=2,
        )
    else:
        print(f"Modo pipeline: usando detecciones de {detections_json_path}")
        model = None
        tracker = None

    # Cargar frames
    frame_paths = load_frames(sequence_folder)
    print(f"Frames encontrados: {len(frame_paths)}")

    # Dimensiones del video de salida
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise FileNotFoundError(f"No se pudo leer el primer frame: {frame_paths[0]}")
    H, W = first.shape[:2]
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_video, fourcc, FPS, (W, H))
    team_colors, team_labels = build_team_style(n_clusters)
    os.makedirs(debug_dir, exist_ok=True)

    tracks_sum:   dict[int, np.ndarray] = {}
    tracks_count: dict[int, int]        = defaultdict(int)
    tracks_seen_count: dict[int, int]   = defaultdict(int)
    tracks_bbox:  dict[int, np.ndarray] = {}
    track_class_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    tracks_left_edge_count: dict[int, int] = defaultdict(int)
    tracks_right_edge_count: dict[int, int] = defaultdict(int)
    tracks_cx_sum: dict[int, float] = defaultdict(float)
    tracks_cx2_sum: dict[int, float] = defaultdict(float)
    tracks_cy_sum: dict[int, float] = defaultdict(float)
    tracks_cy2_sum: dict[int, float] = defaultdict(float)
    tracks_w_sum: dict[int, float] = defaultdict(float)
    tracks_h_sum: dict[int, float] = defaultdict(float)

    raw_frames:     list[np.ndarray]          = []
    frame_tracks:   list[list[dict]]          = []

    print(f"\n--- Fase 1: {'lectura JSON' if use_json_detections else 'deteccion + tracking'} + acumulacion HSV ---")
    t0_total = time.perf_counter()

    # Pre-cargar datos de Phase 1 si estamos en modo pipeline
    phase1_by_frame: dict[int, list[dict]] = {}
    if use_json_detections:
        import json as _json
        with open(detections_json_path, 'r') as _f:
            _phase1_data = _json.load(_f)
        for entry in _phase1_data:
            fid = entry["frame_id"]
            phase1_by_frame[fid] = entry.get("players", [])
        del _phase1_data
        print(f"  Phase 1 JSON cargado: {len(phase1_by_frame)} frames con detecciones.")

    for i, path in enumerate(frame_paths):
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"No se pudo leer el frame: {path}")

        frame_data: list[dict] = []

        if use_json_detections:
            # --- Modo pipeline: bboxes + track_ids de Phase 1 ---
            try:
                frame_id = int(os.path.splitext(os.path.basename(path))[0])
            except ValueError:
                frame_id = i
            players = phase1_by_frame.get(frame_id, [])
            for p in players:
                tid = int(p["track_id"])
                xyxy = np.array([p["x_min"], p["y_min"], p["x_max"], p["y_max"]], dtype=np.float32)
                conf = float(p.get("confidence", 1.0))
                cls_id = int(p.get("class_id", 0))

                track_class_counts[tid][cls_id] += 1
                tracks_seen_count[tid] += 1
                cx = (float(xyxy[0]) + float(xyxy[2])) * 0.5
                cy = (float(xyxy[1]) + float(xyxy[3])) * 0.5
                w_box = max(0.0, float(xyxy[2]) - float(xyxy[0]))
                h_box = max(0.0, float(xyxy[3]) - float(xyxy[1]))
                tracks_cx_sum[tid] += cx
                tracks_cx2_sum[tid] += cx * cx
                tracks_cy_sum[tid] += cy
                tracks_cy2_sum[tid] += cy * cy
                tracks_w_sum[tid] += w_box
                tracks_h_sum[tid] += h_box

                crop = TeamClassifier.crop_torso(
                    frame, xyxy,
                    top_ratio=TORSO_TOP_RATIO, bottom_ratio=TORSO_BOTTOM_RATIO,
                    left_ratio=TORSO_LEFT_RATIO, right_ratio=TORSO_RIGHT_RATIO,
                )
                if crop is not None and crop.size > 0:
                    feat = _hsv_descriptor(crop)
                    if feat is not None:
                        if tid in tracks_sum:
                            tracks_sum[tid] += feat
                        else:
                            tracks_sum[tid] = feat.copy()
                        tracks_count[tid] += 1
                        tracks_bbox[tid] = xyxy

                frame_data.append({"tid": tid, "xyxy": xyxy, "conf": conf, "class_id": cls_id})

                if cx < GK_EDGE_X_RATIO * W:
                    tracks_left_edge_count[tid] += 1
                elif cx > (1.0 - GK_EDGE_X_RATIO) * W:
                    tracks_right_edge_count[tid] += 1

        else:
            # --- Modo standalone: RF-DETR + ByteTrack ---
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = model.predict(pil, threshold=conf_threshold)
            if not isinstance(detections, sv.Detections):
                detections = detections[0] if detections else sv.Detections.empty()
            detections = sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id
            )
            tracked = tracker.update_with_detections(detections)

            if tracked.tracker_id is not None:
                confidences = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
                class_ids = tracked.class_id if tracked.class_id is not None else np.zeros(len(tracked), dtype=int)
                for j in range(len(tracked)):
                    tid  = int(tracked.tracker_id[j])
                    xyxy = tracked.xyxy[j]
                    conf = float(confidences[j])
                    cls_id = int(class_ids[j])
                    track_class_counts[tid][cls_id] += 1
                    tracks_seen_count[tid] += 1
                    cx = (float(xyxy[0]) + float(xyxy[2])) * 0.5
                    cy = (float(xyxy[1]) + float(xyxy[3])) * 0.5
                    w_box = max(0.0, float(xyxy[2]) - float(xyxy[0]))
                    h_box = max(0.0, float(xyxy[3]) - float(xyxy[1]))
                    tracks_cx_sum[tid] += cx
                    tracks_cx2_sum[tid] += cx * cx
                    tracks_cy_sum[tid] += cy
                    tracks_cy2_sum[tid] += cy * cy
                    tracks_w_sum[tid] += w_box
                    tracks_h_sum[tid] += h_box

                    crop = TeamClassifier.crop_torso(
                        frame, xyxy,
                        top_ratio=TORSO_TOP_RATIO, bottom_ratio=TORSO_BOTTOM_RATIO,
                        left_ratio=TORSO_LEFT_RATIO, right_ratio=TORSO_RIGHT_RATIO,
                    )
                    if crop is not None and crop.size > 0:
                        feat = _hsv_descriptor(crop)
                        if feat is not None:
                            if tid in tracks_sum:
                                tracks_sum[tid] += feat
                            else:
                                tracks_sum[tid] = feat.copy()
                            tracks_count[tid] += 1
                            tracks_bbox[tid] = xyxy

                    frame_data.append({"tid": tid, "xyxy": xyxy, "conf": conf, "class_id": cls_id})

                    if cx < GK_EDGE_X_RATIO * W:
                        tracks_left_edge_count[tid] += 1
                    elif cx > (1.0 - GK_EDGE_X_RATIO) * W:
                        tracks_right_edge_count[tid] += 1

        raw_frames.append(frame)
        frame_tracks.append(frame_data)

        if i % 100 == 0:
            elapsed = time.perf_counter() - t0_total
            print(f"  Frame {i+1}/{len(frame_paths)} | "
                  f"tracks activos: {len(frame_data)} | "
                  f"{elapsed:.1f}s transcurridos")

    print(f"Fase 1 completa. Tracks unicos acumulados: {len(tracks_sum)}")



    observed_class_ids: set[int] = set()
    for counts in track_class_counts.values():
        observed_class_ids.update(counts.keys())
    inferred_gk_class_ids = _infer_gk_class_ids(observed_class_ids)
    inferred_ref_class_ids = _infer_ref_class_ids(observed_class_ids, inferred_gk_class_ids)
    use_gk_class_effective = bool(use_gk_class and inferred_gk_class_ids)
    role_by_tid = _majority_role_by_track(track_class_counts, inferred_gk_class_ids, inferred_ref_class_ids)
    detected_referee_tracks = {
        tid for tid, role in role_by_tid.items()
        if role == "referee" and tracks_seen_count.get(tid, 0) >= MIN_REF_TRACK_FRAMES
    }

    # --- Capa 2: Soft referee detection ---
    # Tracks que no ganaron por mayoría pero tienen presencia consistente como referee
    soft_referee_candidates: dict[int, float] = {}
    for tid, counts in track_class_counts.items():
        if tid in detected_referee_tracks:
            continue  # ya está confirmado
        total = sum(counts.values())
        ref_votes = sum(v for cid, v in counts.items() if cid in inferred_ref_class_ids)
        if total < MIN_REF_TRACK_FRAMES or ref_votes < REF_MIN_ABS_VOTES:
            continue
        ratio = ref_votes / total
        if ratio >= REF_MIN_VOTE_RATIO:
            soft_referee_candidates[tid] = ratio
            
    if soft_referee_candidates:
        print(f"[info] Candidatos soft referee: {dict(sorted(soft_referee_candidates.items(), key=lambda x: -x[1]))}")
    class_goalie_tracks = {
        tid for tid, role in role_by_tid.items()
        if role == "goalkeeper" and tracks_seen_count.get(tid, 0) >= MIN_GK_TRACK_FRAMES
    }
    active_referee_tracks = set() if cluster_referee else set(detected_referee_tracks)

    # Fallback suave para clips congestionados (p.ej. tiros de esquina):
    # si la mayorÃ­a por track no alcanza, usar razÃ³n de votos GK por track.
    if inferred_gk_class_ids:
        for tid, counts in track_class_counts.items():
            seen = tracks_seen_count.get(tid, 0)
            if seen < MIN_GK_TRACK_FRAMES or tid in detected_referee_tracks:
                continue
            gk_votes = sum(v for cid, v in counts.items() if cid in inferred_gk_class_ids)
            gk_ratio = gk_votes / max(1, seen)
            if gk_votes >= GK_SOFT_MIN_VOTES and gk_ratio >= GK_SOFT_MIN_RATIO:
                class_goalie_tracks.add(tid)

    if use_gk_class and not use_gk_class_effective:
        print("[warn] --use-gk-class activo, pero no se observÃ³ separaciÃ³n de clases de portero. Se usarÃ¡ fallback legacy.")
    if use_gk_class_effective:
        print(
            f"[info] ClasificaciÃ³n por clase activada: gk_class_ids={sorted(inferred_gk_class_ids)} "
            f"| tracks_gk={len(class_goalie_tracks)}"
        )
    if detected_referee_tracks:
        print(
            f"[info] Seguimiento de Ã¡rbitro detectado: ref_class_ids={sorted(inferred_ref_class_ids)} "
            f"| tracks_ref={len(detected_referee_tracks)} | cluster_referee={int(cluster_referee)}"
        )

    # â”€â”€ Fase 2: Clustering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n--- Fase 2: clustering ({mode}, k={n_clusters}) ---")
    classifier = TeamClassifier(mode=mode, n_clusters=n_clusters)

    # Filtrar tracks cortos (muy ruidosos) y calcular descriptor MEDIO real
    track_ids_list = sorted(
        tid for tid in tracks_sum
        if tracks_count[tid] >= MIN_TRACK_FRAMES
        and (not use_gk_class_effective or tid not in class_goalie_tracks)
        and tid not in active_referee_tracks
    )
    if len(track_ids_list) < n_clusters:
        # Fallback de seguridad: si la exclusiÃ³n deja pocos tracks, usar todos.
        track_ids_list = sorted(
            tid for tid in tracks_sum
            if tracks_count[tid] >= MIN_TRACK_FRAMES and tid not in active_referee_tracks
        )
        print("[warn] Pocos outfield tracks tras exclusiÃ³n de porteros; se usa fallback con todos los tracks.")
    n_filtered = len(tracks_sum) - len(track_ids_list)
    print(f"  Tracks con >={MIN_TRACK_FRAMES} frames: {len(track_ids_list)} "
          f"(descartados {n_filtered} tracks cortos)")

    mean_descs:  list[np.ndarray] = []
    bboxes_list: list[np.ndarray] = []
    mean_desc_by_tid: dict[int, np.ndarray] = {}

    outfield_track_ids = set(track_ids_list)

    for tid in track_ids_list:
        # Descriptor MEDIO acumulado (no el Ãºltimo crop â€” ese era el bug)
        mean_desc = tracks_sum[tid] / tracks_count[tid]
        mean_descs.append(mean_desc)
        bboxes_list.append(tracks_bbox[tid])
        mean_desc_by_tid[tid] = mean_desc

    # fit_from_descriptors: recibe vectores ya computados, no crops crudos
    track_to_team = classifier.fit_from_descriptors(mean_descs, track_ids_list, bboxes_list)

    # Asignar tracks restantes (cortos) al centroide mÃ¡s cercano para evitar
    # que aparezca '?' solo por falta de frames en el fitting inicial.
    short_assigned = 0
    if classifier._cluster_centers is not None:
        for tid in sorted(tracks_sum.keys()):
            if tid in track_to_team or tid not in outfield_track_ids:
                continue
            mean_desc = tracks_sum[tid] / max(1, tracks_count[tid])
            mean_desc_by_tid[tid] = mean_desc
            dists = np.linalg.norm(classifier._cluster_centers - mean_desc, axis=1)
            track_to_team[tid] = int(np.argmin(dists))
            short_assigned += 1

    if short_assigned:
        print(f"  Tracks cortos asignados por centroide: {short_assigned}")

    # Si se desactiva "?", reasignar todo outlier (-1) al equipo mÃ¡s cercano.
    forced_reassignments = 0
    if not allow_unknown and classifier._cluster_centers is not None:
        for tid, team in list(track_to_team.items()):
            if team != -1:
                continue
            desc = mean_desc_by_tid.get(tid)
            if desc is None:
                continue
            dists = np.linalg.norm(classifier._cluster_centers - desc, axis=1)
            track_to_team[tid] = int(np.argmin(dists))
            forced_reassignments += 1

    if forced_reassignments:
        print(f"  Outliers reasignados por centroide (sin '?'): {forced_reassignments}")

    if active_referee_tracks:
        for tid in active_referee_tracks:
            track_to_team[tid] = -2

    # Rol de portero separado del team_id (solo para K=2), con modo legacy/fused.
    goalie_tracks: set[int] = set()
    referee_tracks: set[int] = set(active_referee_tracks)
    candidate_debug: dict[int, dict] = {}
    vote_debug: dict[int, tuple[int, int, int]] = {}
    team_p70: dict[int, float] = {}
    gk_decision_confidence: dict[int, float] = {}
    gk_reason_code: dict[int, str] = {}

    def _votes_for_tid(gk_tid: int, start_idx: int, end_idx: int, other_goalies: set[int]) -> tuple[int, int, int]:
        votes: list[int] = []
        for fidx in range(start_idx, end_idx):
            frame_data = frame_tracks[fidx]
            gk_obj = None
            others = []
            for obj in frame_data:
                tid = int(obj["tid"])
                if tid == gk_tid:
                    gk_obj = obj
                else:
                    others.append(obj)

            if gk_obj is None:
                continue

            gx1, gy1, gx2, gy2 = map(float, gk_obj["xyxy"])
            gcx = (gx1 + gx2) * 0.5
            gcy = (gy1 + gy2) * 0.5

            neighbor_dists: list[tuple[float, int]] = []
            for obj in others:
                tid = int(obj["tid"])
                if tid in other_goalies or tid in referee_tracks:
                    continue
                team = track_to_team.get(tid)
                if team not in (0, 1):
                    continue
                x1, y1, x2, y2 = map(float, obj["xyxy"])
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                dist = float(np.hypot(gcx - cx, gcy - cy))
                neighbor_dists.append((dist, team))

            if not neighbor_dists:
                continue

            neighbor_dists.sort(key=lambda x: x[0])
            for _, team in neighbor_dists[:GK_NEIGHBOR_K]:
                votes.append(team)

        vote_a = sum(1 for t in votes if t == 0)
        vote_b = sum(1 for t in votes if t == 1)
        return vote_a, vote_b, len(votes)

    if n_clusters == 2 and classifier._cluster_centers is not None:
        per_team_dist: dict[int, list[float]] = defaultdict(list)
        dist_by_tid: dict[int, float] = {}

        for tid, team in track_to_team.items():
            if team not in (0, 1):
                continue
            desc = mean_desc_by_tid.get(tid)
            if desc is None:
                continue
            d = float(np.linalg.norm(desc - classifier._cluster_centers[team]))
            dist_by_tid[tid] = d
            if tracks_count.get(tid, 0) >= MIN_TRACK_FRAMES:
                per_team_dist[team].append(d)

        for team in (0, 1):
            arr = np.asarray(per_team_dist.get(team, []), dtype=np.float32)
            team_p70[team] = float(np.percentile(arr, 70)) if arr.size >= 4 else 0.0

        # --- Referee Advanced Logic (Outlier + Soft + Limit) ---
        hsv_referee_candidates = []
        for tid, team in list(track_to_team.items()):
            if team not in (0, 1):
                continue
            desc = mean_desc_by_tid.get(tid)
            if desc is None:
                continue
            dist0 = float(np.linalg.norm(desc - classifier._cluster_centers[0]))
            dist1 = float(np.linalg.norm(desc - classifier._cluster_centers[1]))
            min_dist = min(dist0, dist1)
            
            if min_dist > REF_HSV_OUTLIER_THRESHOLD and tid not in class_goalie_tracks:
                hsv_referee_candidates.append((tid, min_dist))

        all_ref_scores = []
        # Prioridad 1: class_id confirmed (score = ratio + 10)
        for tid in detected_referee_tracks:
            counts = track_class_counts.get(tid, {})
            tot = sum(counts.values())
            ratio = counts.get(2, 0) / max(1, tot)
            all_ref_scores.append((tid, ratio + 10.0, "class_id"))

        # Prioridad 2: soft referee candidates (score = absolute votes, not ratio)
        # Un track con 46 votos absolutos en 423 frames es MÁS confiable
        # que uno con 22 votos en 77 frames, aunque el ratio sea menor.
        for tid, ratio in soft_referee_candidates.items():
            if tid not in detected_referee_tracks and tid not in class_goalie_tracks:
                ref_abs = sum(v for cid, v in track_class_counts.get(tid, {}).items() if cid in inferred_ref_class_ids)
                all_ref_scores.append((tid, 5.0 + ref_abs * 0.01, "soft_vote"))

        # Prioridad 3: HSV outliers (score = distance, typically 0.5-1.0)
        for tid, dist in hsv_referee_candidates:
            if tid not in detected_referee_tracks and tid not in soft_referee_candidates:
                all_ref_scores.append((tid, dist, "hsv_outlier"))

        all_ref_scores.sort(key=lambda x: x[1], reverse=True)
        final_referee_tracks = [x[0] for x in all_ref_scores[:MAX_REFEREE_TRACKS]]
        
        if all_ref_scores:
            print(f"[info] Referee scoring (top {MAX_REFEREE_TRACKS}):")
            for tid, score, reason in all_ref_scores:
                selected = "SELECTED" if tid in final_referee_tracks else "---"
                print(f"  #{tid}: score={score:.3f} reason={reason} {selected}")
        
        detected_referee_tracks = set(final_referee_tracks)
        
        # Ensure referees are cleanly removed from K-Means team/GK assignments
        for tid in detected_referee_tracks:
            if tid in track_to_team:
                del track_to_team[tid]
            if tid in class_goalie_tracks:
                class_goalie_tracks.remove(tid)
        # ------------------------------------------------

        if gk_assignment_mode == "legacy":
            if use_gk_class_effective and class_goalie_tracks:
                goalie_tracks = set(class_goalie_tracks)
                for tid in goalie_tracks:
                    gk_reason_code[tid] = "legacy_class_seed"
            else:
                side_best: dict[str, tuple[int, float]] = {}
                for tid, n_seen in tracks_seen_count.items():
                    if n_seen < MIN_GK_TRACK_FRAMES:
                        continue
                    left_hits = tracks_left_edge_count.get(tid, 0)
                    right_hits = tracks_right_edge_count.get(tid, 0)
                    edge_ratio = max(left_hits, right_hits) / max(1, n_seen)
                    if edge_ratio < GK_MIN_EDGE_PERSISTENCE:
                        continue
                    side = "left" if left_hits >= right_hits else "right"
                    rarity = max(0.0, dist_by_tid.get(tid, 0.0) - team_p70.get(track_to_team.get(tid, -1), 0.0))
                    candidate_score = edge_ratio + 0.6 * rarity
                    prev = side_best.get(side)
                    if prev is None or candidate_score > prev[1]:
                        side_best[side] = (tid, candidate_score)
                        candidate_debug[tid] = {
                            "side": side,
                            "edge_ratio": float(edge_ratio),
                            "rarity": float(rarity),
                            "score": float(candidate_score),
                        }

                goalie_tracks = {tid for tid, _ in side_best.values()}
                for tid in goalie_tracks:
                    gk_reason_code[tid] = "legacy_edge_candidate"

            for tid in sorted(goalie_tracks):
                va, vb, vt = _votes_for_tid(tid, 0, len(frame_tracks), goalie_tracks - {tid})
                vote_debug[tid] = (va, vb, vt)
                if vt > 0:
                    team = 0 if va >= vb else 1
                    track_to_team[tid] = team
                    margin = abs(va - vb) / max(1, vt)
                    gk_decision_confidence[tid] = float(margin)
                    if margin < GK_MIN_VOTE_MARGIN or vt < GK_MIN_VOTE_COUNT:
                        gk_reason_code[tid] = "legacy_low_confidence"
                    else:
                        gk_reason_code[tid] = gk_reason_code.get(tid, "legacy_neighbor_vote")

        else:
            if use_gk_class_effective and class_goalie_tracks:
                goalie_tracks = set(class_goalie_tracks)
                ws, wn, wa = 0.55, 0.35, 0.10
                for tid in sorted(goalie_tracks):
                    comp_sum = {0: {"spatial": 0.0, "neighbor": 0.0, "appearance": 0.0},
                                1: {"spatial": 0.0, "neighbor": 0.0, "appearance": 0.0}}
                    comp_count = 0
                    for frame_data in frame_tracks:
                        gk_obj = next((o for o in frame_data if int(o["tid"]) == tid), None)
                        if gk_obj is None:
                            continue

                        gx1, gy1, gx2, gy2 = map(float, gk_obj["xyxy"])
                        gcx = (gx1 + gx2) * 0.5
                        gcy = (gy1 + gy2) * 0.5

                        team_points = {0: [], 1: []}
                        for obj in frame_data:
                            oid = int(obj["tid"])
                            if oid == tid or oid in goalie_tracks or oid in referee_tracks:
                                continue
                            team = track_to_team.get(oid)
                            if team not in (0, 1):
                                continue
                            x1, y1, x2, y2 = map(float, obj["xyxy"])
                            cx = (x1 + x2) * 0.5
                            cy = (y1 + y2) * 0.5
                            team_points[team].append((cx, cy))

                        if not team_points[0] and not team_points[1]:
                            continue

                        for team in (0, 1):
                            pts = team_points[team]
                            if pts:
                                arr = np.asarray(pts, dtype=np.float32)
                                centroid = arr.mean(axis=0)
                                dist = float(np.hypot(gcx - centroid[0], gcy - centroid[1]))
                                spatial = 1.0 / (1.0 + dist / max(1.0, float(W)))
                            else:
                                spatial = 0.0

                            neigh_pairs: list[tuple[float, int]] = []
                            for other_team in (0, 1):
                                for px, py in team_points[other_team]:
                                    d = float(np.hypot(gcx - px, gcy - py))
                                    neigh_pairs.append((d, other_team))
                            neigh_pairs.sort(key=lambda x: x[0])
                            neigh = neigh_pairs[:GK_NEIGHBOR_K]
                            if neigh:
                                num = sum((1.0 / (d + 1.0)) for d, t in neigh if t == team)
                                den = sum((1.0 / (d + 1.0)) for d, _ in neigh)
                                neighbor_aff = float(num / max(1e-9, den))
                            else:
                                neighbor_aff = 0.0

                            desc = mean_desc_by_tid.get(tid)
                            if desc is not None and classifier._cluster_centers is not None:
                                d_app = float(np.linalg.norm(desc - classifier._cluster_centers[team]))
                                appearance = 1.0 / (1.0 + d_app)
                            else:
                                appearance = 0.0

                            comp_sum[team]["spatial"] += spatial
                            comp_sum[team]["neighbor"] += neighbor_aff
                            comp_sum[team]["appearance"] += appearance

                        comp_count += 1

                    if comp_count == 0:
                        va, vb, vt = _votes_for_tid(tid, 0, len(frame_tracks), goalie_tracks - {tid})
                        vote_debug[tid] = (va, vb, vt)
                        if vt > 0:
                            team = 0 if va >= vb else 1
                            track_to_team[tid] = team
                            gk_decision_confidence[tid] = float(abs(va - vb) / max(1, vt))
                            gk_reason_code[tid] = "fused_class_seed_neighbor_fallback"
                        continue

                    fused_score: dict[int, float] = {}
                    for team in (0, 1):
                        s = comp_sum[team]["spatial"] / comp_count
                        n = comp_sum[team]["neighbor"] / comp_count
                        a = comp_sum[team]["appearance"] / comp_count
                        fused_score[team] = ws * s + wn * n + wa * a

                    pred_team = 0 if fused_score[0] >= fused_score[1] else 1
                    track_to_team[tid] = pred_team
                    conf = abs(fused_score[0] - fused_score[1]) / max(1e-9, fused_score[0] + fused_score[1])
                    gk_decision_confidence[tid] = float(conf)
                    gk_reason_code[tid] = "fused_class_seeded"
                    candidate_debug[tid] = {
                        "score": float(fused_score[pred_team]),
                        "combined": float(fused_score[pred_team]),
                        "fused_spatial_a": float(comp_sum[0]["spatial"] / comp_count),
                        "fused_spatial_b": float(comp_sum[1]["spatial"] / comp_count),
                        "fused_neighbor_a": float(comp_sum[0]["neighbor"] / comp_count),
                        "fused_neighbor_b": float(comp_sum[1]["neighbor"] / comp_count),
                        "fused_appearance_a": float(comp_sum[0]["appearance"] / comp_count),
                        "fused_appearance_b": float(comp_sum[1]["appearance"] / comp_count),
                        "fused_score_a": float(fused_score[0]),
                        "fused_score_b": float(fused_score[1]),
                        "side": "left" if (tracks_left_edge_count.get(tid, 0) >= tracks_right_edge_count.get(tid, 0)) else "right",
                    }
            else:
                # Fused fallback sin clase de portero: conserva la lÃ³gica temporal legacy.
                best_seen: dict[int, float] = {}
                current_by_side: dict[str, int | None] = {"left": None, "right": None}
                pending_by_side: dict[str, int | None] = {"left": None, "right": None}
                pending_count: dict[str, int] = {"left": 0, "right": 0}

                n_frames = len(frame_tracks)
                update_points = list(range(GK_UPDATE_EVERY, n_frames + 1, GK_UPDATE_EVERY))
                if not update_points or update_points[-1] != n_frames:
                    update_points.append(n_frames)

                for end_idx in update_points:
                    start_idx = max(0, end_idx - GK_WINDOW_FRAMES)

                    window_seen: dict[int, int] = defaultdict(int)
                    window_left: dict[int, int] = defaultdict(int)
                    window_right: dict[int, int] = defaultdict(int)
                    window_cx_sum: dict[int, float] = defaultdict(float)
                    window_cx2_sum: dict[int, float] = defaultdict(float)
                    window_cy_sum: dict[int, float] = defaultdict(float)
                    window_cy2_sum: dict[int, float] = defaultdict(float)

                    for fidx in range(start_idx, end_idx):
                        for obj in frame_tracks[fidx]:
                            tid = int(obj["tid"])
                            if track_to_team.get(tid) not in (0, 1):
                                continue
                            x1, y1, x2, y2 = map(float, obj["xyxy"])
                            cx = (x1 + x2) * 0.5
                            cy = (y1 + y2) * 0.5
                            window_seen[tid] += 1
                            window_cx_sum[tid] += cx
                            window_cx2_sum[tid] += cx * cx
                            window_cy_sum[tid] += cy
                            window_cy2_sum[tid] += cy * cy
                            if cx < GK_EDGE_X_RATIO * W:
                                window_left[tid] += 1
                            elif cx > (1.0 - GK_EDGE_X_RATIO) * W:
                                window_right[tid] += 1

                    side_candidates: dict[str, list[int]] = {"left": [], "right": []}
                    temp_stats: dict[int, dict] = {}

                    for tid, n in window_seen.items():
                        if n < GK_MIN_WINDOW_SEEN or n < MIN_GK_TRACK_FRAMES:
                            continue

                        left_hits = window_left.get(tid, 0)
                        right_hits = window_right.get(tid, 0)
                        edge_ratio = max(left_hits, right_hits) / max(1, n)
                        if edge_ratio < GK_MIN_EDGE_PERSISTENCE:
                            continue

                        team = int(track_to_team.get(tid, -1))
                        d = dist_by_tid.get(tid, 0.0)
                        rarity = max(0.0, d - team_p70.get(team, 0.0))
                        candidate_score = edge_ratio + 0.6 * rarity

                        mean_cx = window_cx_sum[tid] / max(1, n)
                        mean_cy = window_cy_sum[tid] / max(1, n)
                        var_cx = max(0.0, window_cx2_sum[tid] / max(1, n) - mean_cx * mean_cx)
                        var_cy = max(0.0, window_cy2_sum[tid] / max(1, n) - mean_cy * mean_cy)
                        std_x_ratio = (var_cx ** 0.5) / max(1.0, float(W))
                        std_y_ratio = (var_cy ** 0.5) / max(1.0, float(H))
                        motion_ok = (std_x_ratio <= GK_MAX_STD_X_RATIO) and (std_y_ratio <= GK_MAX_STD_Y_RATIO)

                        side = "left" if left_hits >= right_hits else "right"
                        side_candidates[side].append(tid)

                        temp_stats[tid] = {
                            "team_before_vote": team,
                            "edge_ratio": float(edge_ratio),
                            "rarity": float(rarity),
                            "score": float(candidate_score),
                            "left_hits": int(left_hits),
                            "right_hits": int(right_hits),
                            "n_frames": int(n),
                            "std_x_ratio": float(std_x_ratio),
                            "std_y_ratio": float(std_y_ratio),
                            "motion_ok": bool(motion_ok),
                            "side": side,
                        }

                    proposed_by_side: dict[str, int | None] = {"left": None, "right": None}

                    for side in ("left", "right"):
                        tids = side_candidates.get(side, [])
                        if not tids:
                            continue

                        for tid in tids:
                            va, vb, vt = _votes_for_tid(tid, start_idx, end_idx, set())
                            vote_margin = abs(va - vb) / max(1, vt)
                            temp_stats[tid]["vote_margin"] = float(vote_margin)
                            temp_stats[tid]["vote_total"] = int(vt)
                            temp_stats[tid]["combined"] = float(
                                temp_stats[tid]["score"] + GK_COMBINED_VOTE_WEIGHT * vote_margin
                            )
                            if vt >= GK_MIN_WINDOW_VOTES:
                                vote_debug[tid] = (va, vb, vt)

                        tids_sorted = sorted(
                            tids,
                            key=lambda tid: float(temp_stats[tid].get("combined", 0.0)),
                            reverse=True,
                        )
                        best = tids_sorted[0]
                        b = temp_stats[best]
                        base_ok = (
                            float(b.get("score", 0.0)) >= GK_MIN_SCORE
                            and int(b.get("vote_total", 0)) >= GK_MIN_WINDOW_VOTES
                            and float(b.get("vote_margin", 0.0)) >= GK_MIN_VOTE_MARGIN
                            and bool(b.get("motion_ok", False))
                        )
                        if not base_ok:
                            continue

                        if len(tids_sorted) == 1:
                            proposed_by_side[side] = best
                        else:
                            second = tids_sorted[1]
                            delta_combined = float(b.get("combined", 0.0)) - float(temp_stats[second].get("combined", 0.0))
                            delta_margin = float(b.get("vote_margin", 0.0)) - float(temp_stats[second].get("vote_margin", 0.0))
                            if delta_combined >= GK_SIDE_MIN_COMBINED_DELTA and delta_margin >= GK_SIDE_MIN_VOTE_MARGIN_DELTA:
                                proposed_by_side[side] = best

                        for tid in tids:
                            combined = float(temp_stats[tid].get("combined", 0.0))
                            if combined > best_seen.get(tid, -1.0):
                                best_seen[tid] = combined
                                candidate_debug[tid] = temp_stats[tid].copy()

                    for side in ("left", "right"):
                        proposed = proposed_by_side[side]
                        current = current_by_side[side]

                        if proposed == current:
                            pending_by_side[side] = None
                            pending_count[side] = 0
                            continue

                        if proposed == pending_by_side[side]:
                            pending_count[side] += 1
                        else:
                            pending_by_side[side] = proposed
                            pending_count[side] = 1

                        if pending_count[side] >= GK_SWITCH_CONFIRM:
                            current_by_side[side] = proposed
                            pending_by_side[side] = None
                            pending_count[side] = 0

                goalie_tracks = {tid for tid in current_by_side.values() if tid is not None}
                for tid in goalie_tracks:
                    gk_reason_code[tid] = "fused_temporal_candidate"

                # Team del portero: voto global sobre todo el clip (mÃ¡s estable).
                for tid in sorted(goalie_tracks):
                    va, vb, vt = _votes_for_tid(tid, 0, len(frame_tracks), goalie_tracks - {tid})
                    vote_debug[tid] = (va, vb, vt)
                    if vt > 0:
                        track_to_team[tid] = 0 if va >= vb else 1
                        margin = abs(va - vb) / max(1, vt)
                        gk_decision_confidence[tid] = float(margin)
                        if margin < GK_MIN_VOTE_MARGIN or vt < GK_MIN_VOTE_COUNT:
                            gk_reason_code[tid] = "fused_low_confidence"

        if candidate_debug:
            print("Candidatos GK (track, side, score, votes, margin, combined, std_xy, reason, accepted):")
            for tid in sorted(candidate_debug):
                cdbg = candidate_debug.get(tid, {})
                score = float(cdbg.get("score", 0.0))
                va, vb, vt = vote_debug.get(tid, (0, 0, 0))
                margin = abs(va - vb) / max(1, vt)
                combined = float(cdbg.get("combined", 0.0))
                sx = float(cdbg.get("std_x_ratio", 0.0))
                sy = float(cdbg.get("std_y_ratio", 0.0))
                side = str(cdbg.get("side", "left"))
                reason = gk_reason_code.get(tid, "n/a")
                ok = "yes" if tid in goalie_tracks else "no"
                print(
                    f"  #{tid} | side={side} | score={score:.3f} | votes={va}/{vb}/{vt} | "
                    f"margin={margin:.2f} | combined={combined:.3f} | std=({sx:.3f},{sy:.3f}) | "
                    f"reason={reason} | accepted={ok}"
                )

        if goalie_tracks:
            print("Porteros detectados (team -> track_id):")
            for tid in sorted(goalie_tracks):
                team = track_to_team.get(tid)
                if team in (0, 1):
                    print(f"  Team {chr(65 + team)} -> #{tid}")
        else:
            print("Porteros detectados: ninguno (no hubo candidato estable).")

    if detected_referee_tracks:
        print(f"Arbitros detectados: {len(detected_referee_tracks)} track(s) -> {sorted(detected_referee_tracks)}")
    else:
        print("Arbitros detectados: ninguno.")

    # EstadÃ­sticas de clustering
    team_counts = defaultdict(int)
    for tid, team in track_to_team.items():
        team_counts[team] += 1
    print("DistribuciÃ³n de tracks por equipo:")
    for team_id in sorted(team_counts):
        label = team_labels.get(team_id, str(team_id))
        print(f"  {label} (id={team_id}): {team_counts[team_id]} tracks")

    seq_name = os.path.basename(os.path.normpath(sequence_folder))

    # Post report: perfil cromÃ¡tico por Team A/Team B + resumen de porteros.
    post_report_lines: list[str] = []
    post_report_lines.append("POST REPORT")
    post_report_lines.append(
        f"sequence={seq_name} mode={mode} k={n_clusters} "
        f"use_gk_class={int(use_gk_class_effective)} gk_assignment_mode={gk_assignment_mode}"
    )

    for team_id in (0, 1):
        team_descs = [
            mean_desc_by_tid[tid]
            for tid, t in track_to_team.items()
            if t == team_id and tid in mean_desc_by_tid
        ]

        if not team_descs:
            post_report_lines.append(f"{team_labels[team_id]}: no descriptors")
            continue

        team_mean = np.mean(np.vstack(team_descs), axis=0)
        h_hist = team_mean[:12]
        s_hist = team_mean[12:24] if team_mean.size >= 24 else np.array([], dtype=np.float32)
        v_hist = team_mean[24:36] if team_mean.size >= 36 else np.array([], dtype=np.float32)

        hue = _weighted_center(h_hist, 0.0, 180.0)
        sat = _weighted_center(s_hist, 0.0, 256.0) if s_hist.size else 0.0
        val = _weighted_center(v_hist, 0.0, 256.0) if v_hist.size else 0.0
        color_name = _hue_to_name(hue)

        post_report_lines.append(
            f"{team_labels[team_id]}: tracks={team_counts.get(team_id, 0)} "
            f"hue={hue:.1f}({color_name}) sat={sat:.1f} val={val:.1f}"
        )

    if goalie_tracks:
        for tid in sorted(goalie_tracks):
            team_id = int(track_to_team.get(tid, -1))
            cdbg = candidate_debug.get(tid, {})
            va, vb, vt = vote_debug.get(tid, (0, 0, 0))
            conf = float(gk_decision_confidence.get(tid, 0.0))
            reason = gk_reason_code.get(tid, "unknown")
            post_report_lines.append(
                f"goalie_track=#{tid} assigned={team_labels.get(team_id, str(team_id))} "
                f"votesA/B={va}/{vb}/{vt} edge={float(cdbg.get('edge_ratio', 0.0)):.2f} "
                f"confidence={conf:.3f} reason={reason}"
            )
    else:
        post_report_lines.append("goalie_track=none")

    if detected_referee_tracks:
        post_report_lines.append(
            f"referee_tracks={len(detected_referee_tracks)} ids={','.join(str(t) for t in sorted(detected_referee_tracks))}"
        )
    else:
        post_report_lines.append("referee_tracks=none")

    print("\n" + "\n".join(post_report_lines))

    post_report_path = os.path.join(debug_dir, f"{seq_name}_post_report.txt")
    with open(post_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(post_report_lines) + "\n")
    print(f"Post report guardado en: {post_report_path}")

    # Exportar diagnÃ³stico por track para anÃ¡lisis offline.
    diag_csv = os.path.join(debug_dir, f"{seq_name}_gk_diagnostics.csv")
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "sequence",
            "track_id",
            "team_id",
            "is_goalie",
            "is_referee",
            "is_candidate",
            "n_seen",
            "n_feat",
            "left_hits",
            "right_hits",
            "edge_ratio",
            "score",
            "combined",
            "rarity",
            "vote_a",
            "vote_b",
            "vote_total",
            "vote_margin",
            "side",
            "std_x_ratio",
            "std_y_ratio",
            "motion_ok",
            "decision_confidence",
            "reason_code",
            "mean_x_ratio",
            "mean_y_ratio",
            "mean_w_ratio",
            "mean_h_ratio",
            "team_p70",
            "fused_spatial_a",
            "fused_spatial_b",
            "fused_neighbor_a",
            "fused_neighbor_b",
            "fused_appearance_a",
            "fused_appearance_b",
            "fused_score_a",
            "fused_score_b",
        ])

        all_tids = sorted(tracks_seen_count.keys())
        for tid in all_tids:
            n_seen = int(tracks_seen_count.get(tid, 0))
            n_feat = int(tracks_count.get(tid, 0))
            team_id = int(track_to_team.get(tid, -1))

            left_hits = int(tracks_left_edge_count.get(tid, 0))
            right_hits = int(tracks_right_edge_count.get(tid, 0))
            edge_ratio = max(left_hits, right_hits) / max(1, n_seen)

            cdbg = candidate_debug.get(tid, {})
            score = float(cdbg.get("score", 0.0))
            combined = float(cdbg.get("combined", 0.0))
            rarity = float(cdbg.get("rarity", 0.0))
            std_x_ratio = float(cdbg.get("std_x_ratio", 0.0))
            std_y_ratio = float(cdbg.get("std_y_ratio", 0.0))
            motion_ok = bool(cdbg.get("motion_ok", False))
            side = "left" if left_hits >= right_hits else "right"
            decision_conf = float(gk_decision_confidence.get(tid, 0.0))
            reason_code = str(gk_reason_code.get(tid, ""))

            fused_spatial_a = float(cdbg.get("fused_spatial_a", 0.0))
            fused_spatial_b = float(cdbg.get("fused_spatial_b", 0.0))
            fused_neighbor_a = float(cdbg.get("fused_neighbor_a", 0.0))
            fused_neighbor_b = float(cdbg.get("fused_neighbor_b", 0.0))
            fused_appearance_a = float(cdbg.get("fused_appearance_a", 0.0))
            fused_appearance_b = float(cdbg.get("fused_appearance_b", 0.0))
            fused_score_a = float(cdbg.get("fused_score_a", 0.0))
            fused_score_b = float(cdbg.get("fused_score_b", 0.0))

            vote_a, vote_b, vote_total = vote_debug.get(tid, (0, 0, 0))
            vote_margin = abs(vote_a - vote_b) / max(1, vote_total)

            mean_x = tracks_cx_sum.get(tid, 0.0) / max(1, n_seen)
            mean_y = tracks_cy_sum.get(tid, 0.0) / max(1, n_seen)
            mean_w = tracks_w_sum.get(tid, 0.0) / max(1, n_seen)
            mean_h = tracks_h_sum.get(tid, 0.0) / max(1, n_seen)

            writer_csv.writerow([
                seq_name,
                tid,
                team_id,
                int(tid in goalie_tracks),
                int(tid in detected_referee_tracks),
                int(tid in candidate_debug),
                n_seen,
                n_feat,
                left_hits,
                right_hits,
                round(float(edge_ratio), 6),
                round(float(score), 6),
                round(float(combined), 6),
                round(float(rarity), 6),
                vote_a,
                vote_b,
                vote_total,
                round(float(vote_margin), 6),
                side,
                round(float(std_x_ratio), 6),
                round(float(std_y_ratio), 6),
                int(motion_ok),
                round(float(decision_conf), 6),
                reason_code,
                round(float(mean_x / max(1.0, float(W))), 6),
                round(float(mean_y / max(1.0, float(H))), 6),
                round(float(mean_w / max(1.0, float(W))), 6),
                round(float(mean_h / max(1.0, float(H))), 6),
                round(float(team_p70.get(team_id, 0.0)), 6),
                round(float(fused_spatial_a), 6),
                round(float(fused_spatial_b), 6),
                round(float(fused_neighbor_a), 6),
                round(float(fused_neighbor_b), 6),
                round(float(fused_appearance_a), 6),
                round(float(fused_appearance_b), 6),
                round(float(fused_score_a), 6),
                round(float(fused_score_b), 6),
            ])
    print(f"DiagnÃ³stico guardado en: {diag_csv}")

    # â”€â”€ Fase 3: Generar video anotado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n--- Fase 3: generando video ---")
    for i, (frame, frame_data) in enumerate(zip(raw_frames, frame_tracks)):
        annotated = frame.copy()

        for obj in frame_data:
            tid  = obj["tid"]
            xyxy = obj["xyxy"]
            conf = obj["conf"]
            team_id = track_to_team.get(tid, -1)
            color   = team_colors[team_id]
            if tid in detected_referee_tracks:
                label = f"REF #{tid}"
            else:
                role_suffix = " GK" if tid in goalie_tracks else ""
                label = f"{team_labels[team_id]}{role_suffix} #{tid}"

            x1, y1, x2, y2 = map(int, xyxy)
            draw_box(annotated, x1, y1, x2, y2, color, label, conf)

        # Overlay de info
        info = (f"RF-DETR + ByteTrack + {mode.upper()} | "
                f"Frame {i+1}/{len(raw_frames)} | "
                f"{len(frame_data)} tracks")
        cv2.rectangle(annotated, (0, 0), (W, 36), (0, 0, 0), -1)
        cv2.putText(annotated, info, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        # Leyenda de equipos (esquina inferior izquierda)
        legend_ids = sorted(team_counts.keys())
        for k, team_id in enumerate(legend_ids):
            color = team_colors[team_id]
            yl = H - 20 - k * 22
            cv2.rectangle(annotated, (10, yl - 14), (26, yl + 2), color, -1)
            cv2.putText(annotated, team_labels[team_id],
                        (32, yl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(annotated)

    writer.release()
    total = time.perf_counter() - t0_total
    print(f"\nâœ“ Video guardado en: {output_video}")
    print(f"  Frames: {len(raw_frames)} | Tiempo total: {total:.1f}s")
    print(f"  Tracks Ãºnicos: {len(track_to_team)} | "
          f"Outliers: {team_counts.get(-1, 0)}")
    return track_to_team, goalie_tracks, detected_referee_tracks

# â”€â”€ Descriptor HSV+V (debe coincidir con TeamClassifier.use_value=True)
def _hsv_descriptor(crop: np.ndarray, bins: int = 12) -> np.ndarray | None:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_not(cv2.inRange(hsv, lower_green, upper_green))
    if cv2.countNonZero(mask) < 16:
        return None
    h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])
    feat = np.concatenate([h_hist.ravel(), s_hist.ravel(), v_hist.ravel()]).astype(np.float32)
    s = float(feat.sum())
    return feat / s if s > 0 else None


# â”€â”€ Entrypoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9: Team Clustering (RF-DETR + HSV K-Means)")
    parser.add_argument("--sequence_dir", type=str, required=True,
                        help="Carpeta de la secuencia, ej: data/test_sequences/SNMOT-116")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Ruta donde se guardará el JSON de salida con las asignaciones")
    parser.add_argument("--output_video", type=str, required=True,
                        help="Ruta del video anotado de salida")
    parser.add_argument("--rfdetr_weights", type=str, default=MODEL_PATH,
                        help="Checkpoint RF-DETR a usar")
    parser.add_argument("--mode", choices=["hsv", "dbscan"], default="hsv",
                        help="Modo de clustering")
    parser.add_argument("--k", type=int, default=2,
                        help="Número de clusters")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="Umbral de confianza para RF-DETR (default: 0.40)")
    parser.add_argument("--detections_json", type=str, default=None,
                        help="Path to Phase 1 detections JSON (pipeline mode, ensures track_id consistency)")
    args = parser.parse_args()
    
    # We set debug_dir parallel to output_json
    debug_dir = os.path.join(os.path.dirname(args.output_json), "team_clustering_debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Call run, intercept the track assignments to save them to JSON
    track_to_team, goalie_tracks, referee_tracks = run(
        mode=args.mode,
        conf_threshold=args.conf,
        allow_unknown=True,
        n_clusters=args.k,
        use_gk_class=True,
        gk_assignment_mode="fused",
        cluster_referee=False,
        model_path=args.rfdetr_weights,
        sequence_folder=args.sequence_dir,
        output_video=args.output_video,
        debug_dir=debug_dir,
        detections_json_path=args.detections_json,
    )
    
    # Create final JSON
    assignments = []
    
    # Prepare lookup sets
    gk_set = set(goalie_tracks)
    ref_set = set(referee_tracks)
    
    for tid, team_id in track_to_team.items():
        role = "player"
        if tid in gk_set:
            role = "goalkeeper"
        elif tid in ref_set:
            role = "referee"
            
        assignments.append({
            "track_id": tid,
            "team_id": team_id,
            "role": role
        })
    
    for ref_id in ref_set:
        if ref_id not in track_to_team:
            assignments.append({"track_id": ref_id, "team_id": -2, "role": "referee"})
            
    with open(args.output_json, 'w') as f:
        import json
        json.dump(assignments, f, indent=4)
        
    print(f"Phase 9 complete! Clustering saved to {args.output_json}")
