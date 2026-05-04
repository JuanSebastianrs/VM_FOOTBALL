"""
Generate comparison videos for jersey models on clip frames.

Creates 4 videos per clip (full / processed / preprocessed_full / preprocessed_processed),
overlaying RF-DETR detections, tracking, team clustering, and jersey predictions.
"""

from __future__ import annotations

import argparse
import sys
import json
import os
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision
import torchvision.transforms as T
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.clustering import team_clustering_phase
from rfdetr import RFDETRBase


MODEL_DEFAULTS = {
    "full": "weights/jersey_full_best.pt",
    "processed": "weights/jersey_processed_best.pt",
    "preprocessed_full": "weights/jersey_full_preprocessed_v2_best.pt",
    "preprocessed_processed": "weights/jersey_processed_preprocessed_v2_best.pt",
}

TEAM_COLORS = [
    (219, 152, 52),
    (37, 37, 213),
    (70, 190, 80),
    (33, 180, 240),
    (180, 80, 180),
]


def run_cmd(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def ensure_local(path_or_uri: str, cache_dir: Path) -> Path:
    if not path_or_uri.startswith("gs://"):
        return Path(path_or_uri)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = path_or_uri.split("/")[-1]
    local_path = cache_dir / filename
    if not local_path.exists():
        env = os.environ.copy()
        env["PATH"] = env.get("PATH", "")
        run_cmd(["gcloud", "storage", "cp", path_or_uri, str(local_path)], env=env)
    return local_path


def load_frames(folder: Path) -> List[Path]:
    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        paths = sorted((folder / "img1").glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No se encontraron .jpg en {folder}")
    return paths


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


def crop_torso(image: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    top = max(y1, min(y1 + int(0.15 * h), y2 - 1))
    bottom = max(top + 1, min(y1 + int(0.60 * h), y2))
    left = max(x1, min(x1 + int(0.20 * w), x2 - 1))
    right = max(left + 1, min(x1 + int(0.80 * w), x2))
    return image[top:bottom, left:right]


def crop_dorsal(image: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    top = max(y1, min(y1 + int(0.20 * h), y2 - 1))
    bottom = max(top + 1, min(y1 + int(0.70 * h), y2))
    left = max(x1, min(x1 + int(0.25 * w), x2 - 1))
    right = max(left + 1, min(x1 + int(0.75 * w), x2))
    return image[top:bottom, left:right]


def build_resnet(model_name: str, num_classes: int = 100, head: str = "linear") -> torch.nn.Module:
    if model_name == "resnet34":
        model = torchvision.models.resnet34(weights=None)
    else:
        model = torchvision.models.resnet18(weights=None)
    num_features = model.fc.in_features
    if head == "dropout":
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(num_features, num_classes),
        )
    else:
        model.fc = torch.nn.Linear(num_features, num_classes)
    return model


def build_transforms(img_size: int = 96) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_jersey_models(model_paths: Dict[str, Path], device: torch.device) -> Dict[str, torch.nn.Module]:
    models: Dict[str, torch.nn.Module] = {}
    for name, path in model_paths.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        has_dropout_head = any(k.startswith("fc.1.") for k in state.keys())
        head = "dropout" if has_dropout_head else "linear"
        model = build_resnet("resnet18", num_classes=100, head=head)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        models[name] = model
    return models


def build_track_role_map(team_json: Path) -> Dict[int, str]:
    if not team_json.exists():
        return {}
    data = json.loads(team_json.read_text(encoding="utf-8"))
    return {int(row["track_id"]): str(row.get("role", "player")) for row in data}


def draw_label(frame: np.ndarray, x1: int, y1: int, text: str, color: Tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def team_color(team_id: int) -> Tuple[int, int, int]:
    if team_id == -2:
        return (255, 255, 255)
    if team_id < 0:
        return (160, 160, 160)
    return TEAM_COLORS[team_id % len(TEAM_COLORS)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate jersey comparison videos")
    parser.add_argument("--clips-root", type=Path, default=Path("clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jersey_compare"))
    parser.add_argument("--rfdetr-weights", type=Path,
                        default=Path("weights/rfdetr_player_gk_ref_best_total.pth"))
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/jersey_model_cache"))
    parser.add_argument("--clustering-mode", choices=["hsv", "dbscan"], default="hsv")
    parser.add_argument("--gk-assignment-mode", choices=["legacy", "fused"], default="legacy")
    parser.add_argument("--model", action="append", default=[],
                        help="Override model path. Format name=path_or_gs://")
    args = parser.parse_args()

    model_specs = MODEL_DEFAULTS.copy()
    for item in args.model:
        if "=" not in item:
            raise ValueError("Model override must be name=path")
        key, value = item.split("=", 1)
        model_specs[key.strip()] = value.strip()

    model_paths: Dict[str, Path] = {}
    for name, path in model_specs.items():
        model_paths[name] = ensure_local(path, args.cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jersey_models = load_jersey_models(model_paths, device)
    transform = build_transforms(img_size=96)

    rfdetr = RFDETRBase(pretrain_weights=str(args.rfdetr_weights), resolution=448)
    tracker = sv.ByteTrack(
        track_activation_threshold=args.conf,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=args.fps,
        minimum_consecutive_frames=2,
    )

    clips = [p for p in args.clips_root.iterdir() if p.is_dir()]
    if not clips:
        raise FileNotFoundError(f"No se encontraron subcarpetas en {args.clips_root}")

    for clip_dir in clips:
        clip_out_dir = args.output_dir / clip_dir.name
        clip_out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = load_frames(clip_dir)
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise FileNotFoundError(f"No se pudo leer {frame_paths[0]}")
        h, w = first.shape[:2]

        track_class_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        tracks_sum: Dict[int, np.ndarray] = {}
        tracks_count: Dict[int, int] = defaultdict(int)
        tracks_bbox: Dict[int, np.ndarray] = {}
        frame_tracks: List[List[Dict[str, object]]] = []

        for idx, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"No se pudo leer {path}")
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = rfdetr.predict(pil, threshold=args.conf)
            if not isinstance(detections, sv.Detections):
                detections = detections[0] if detections else sv.Detections.empty()
            detections = sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id,
            )
            tracked = tracker.update_with_detections(detections)

            frame_data: List[Dict[str, object]] = []
            if tracked.tracker_id is not None:
                confidences = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
                class_ids = tracked.class_id if tracked.class_id is not None else np.zeros(len(tracked), dtype=int)
                for j in range(len(tracked)):
                    tid = int(tracked.tracker_id[j])
                    xyxy = tracked.xyxy[j]
                    conf = float(confidences[j])
                    cls_id = int(class_ids[j])

                    track_class_counts[tid][cls_id] += 1
                    tracks_bbox[tid] = xyxy

                    crop = crop_torso(frame, xyxy)
                    if crop is not None and crop.size > 0:
                        feat = _hsv_descriptor(crop)
                        if feat is not None:
                            if tid in tracks_sum:
                                tracks_sum[tid] += feat
                            else:
                                tracks_sum[tid] = feat.copy()
                            tracks_count[tid] += 1

                    frame_data.append({
                        "tid": tid,
                        "xyxy": xyxy,
                        "conf": conf,
                        "class_id": cls_id,
                    })

            frame_tracks.append(frame_data)
            if idx % 100 == 0:
                print(f"[{clip_dir.name}] Frame {idx+1}/{len(frame_paths)}")

        team_json = clip_out_dir / f"{clip_dir.name}_team_assignments.json"
        team_video = clip_out_dir / f"{clip_dir.name}_team_clustering.mp4"
        # Write assignments JSON (track_id -> team_id/role)
        track_to_team, goalie_tracks, referee_tracks = team_clustering_phase.run(
            mode=args.clustering_mode,
            conf_threshold=args.conf,
            allow_unknown=True,
            n_clusters=2,
            use_gk_class=True,
            gk_assignment_mode=args.gk_assignment_mode,
            cluster_referee=False,
            model_path=str(args.rfdetr_weights),
            sequence_folder=str(clip_dir),
            output_video=str(team_video),
            debug_dir=str(clip_out_dir / "team_clustering_debug"),
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
        team_json.write_text(json.dumps(assignments, indent=2), encoding="utf-8")

        # Build team/role maps from clustering JSON
        track_to_team: Dict[int, int] = {}
        role_by_tid: Dict[int, str] = {}
        data = json.loads(team_json.read_text(encoding="utf-8"))
        for row in data:
            tid = int(row["track_id"])
            track_to_team[tid] = int(row.get("team_id", -1))
            role_by_tid[tid] = str(row.get("role", "player"))

        for model_name, model in jersey_models.items():
            output_path = clip_out_dir / f"{clip_dir.name}_{model_name}.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (w, h))

            history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))

            for idx, path in enumerate(frame_paths):
                frame = cv2.imread(str(path))
                if frame is None:
                    raise FileNotFoundError(f"No se pudo leer {path}")

                tracks = frame_tracks[idx]
                crops: List[torch.Tensor] = []
                crop_meta: List[Tuple[int, np.ndarray]] = []
                for t in tracks:
                    tid = int(t["tid"])
                    xyxy = t["xyxy"]
                    crop = crop_dorsal(frame, xyxy)
                    if crop is None or crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(crop_rgb)
                    tensor = transform(pil)
                    crops.append(tensor)
                    crop_meta.append((tid, xyxy))

                if crops:
                    batch = torch.stack(crops).to(device)
                    with torch.no_grad():
                        logits = model(batch)
                        probs = torch.softmax(logits, dim=1)
                        confs, preds = torch.max(probs, dim=1)
                    for (tid, _), pred, conf in zip(crop_meta, preds.cpu().numpy(), confs.cpu().numpy()):
                        history[tid].append((int(pred), float(conf)))

                for t in tracks:
                    tid = int(t["tid"])
                    xyxy = t["xyxy"]
                    conf = float(t["conf"])
                    cls_id = int(t["class_id"])

                    team_id = track_to_team.get(tid, -1)
                    role = role_by_tid.get(tid, "player")

                    votes: Dict[int, float] = defaultdict(float)
                    for num, c in history.get(tid, []):
                        votes[num] += c
                    if votes:
                        pred_num = max(votes.items(), key=lambda x: x[1])[0]
                    else:
                        pred_num = -1

                    color = team_color(team_id)
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    role_label = "P"
                    if cls_id == 1 or role == "gk":
                        role_label = "GK"
                    elif cls_id == 2 or role == "ref":
                        role_label = "REF"

                    num_text = "?" if pred_num < 0 else str(pred_num)
                    label = f"T{team_id} {role_label} J{num_text} {conf:.2f}"
                    draw_label(frame, x1, y1, label, color)

                cv2.putText(frame, f"Model: {model_name}", (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(frame)

            writer.release()
            print(f"[OK] Video generado: {output_path}")


if __name__ == "__main__":
    main()
