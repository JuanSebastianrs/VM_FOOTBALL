# render_match_visualization_opt.py
import re, json, os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

SECUENCIA = "SNMOT-116"
SPLIT = "test"
DEVICE = 0

ROOT = Path(__file__).resolve().parents[2] / "VM_FOOTBALL"
DATASET = ROOT / "datasets" / "reorganized_dataset"
IMAGES_DIR = DATASET / "images" / SPLIT
CLUSTER_OUT_DIR = ROOT / "outputs" / "team_clustering"
FRAME_ASSIGN_JSON = CLUSTER_OUT_DIR / f"{SECUENCIA}_frame_assignments.json"

DET_MODEL_WEIGHTS = ROOT / "runs" / "train" / "modelo_yolo11vn_4class" / "weights" / "best.pt"
BALL_MODEL_WEIGHTS = ROOT / "runs" / "train" / "ball_det_yolo11n" / "weights" / "best.pt"

OUT_DIR = ROOT / "outputs" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO = OUT_DIR / f"{SECUENCIA}_viz.mp4"

# colores
COLOR_TEAM0 = (0, 170, 255)
COLOR_TEAM1 = (255, 0, 170)
COLOR_GK    = (0, 255, 255)
COLOR_REF   = (200, 200, 200)
COLOR_BALL  = (0, 255, 0)
COLOR_NEUTRAL = (180, 180, 180)

CLS_PLAYER, CLS_GK, CLS_REF, CLS_BALL = 0, 1, 2, 3

# tunables
FPS = 25
IMGSZ_BALL = 960
CONF_BALL, IOU_BALL = 0.25, 0.5
IMGSZ_DET = 736
CONF_DET, IOU_DET = 0.35, 0.7
KEEP_GAP = 5                 # frames de persistencia de bbox cuando falta
MIN_PLAYERS_FOR_FALLBACK = 14
FALLBACK_EVERY = 3           # correr respaldo cada N frames si faltan jugadores
USE_FALLBACK_DET = True

def load_frame_list(images_dir: Path, seq_prefix: str):
    pat = re.compile(rf"^{re.escape(seq_prefix)}_(\d+)\.jpg$", re.IGNORECASE)
    frames = []
    for p in images_dir.glob(f"{seq_prefix}_*.jpg"):
        m = pat.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]

def load_frame_assignments(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frame_map = {}
    for item in data:
        frame_map[item["frame"]] = item.get("objects", [])
    return frame_map

def draw_box(img, xyxy, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)

def predict_ball_stream(ball_model, frames, device, imgsz, conf, iou):
    preds = []
    half = (device == 0) and torch.cuda.is_available()
    with torch.inference_mode():
        gen = ball_model.predict(
            source=[str(p) for p in frames],
            imgsz=imgsz, conf=conf, iou=iou,
            device=device, half=half, verbose=False,
            stream=True, workers=0
        )
        for res in gen:
            if res.boxes is None or len(res.boxes) == 0:
                preds.append(None)
                continue
            confs = res.boxes.conf.detach().cpu().numpy()
            idx = int(np.argmax(confs))
            xyxy = res.boxes.xyxy[idx].detach().cpu().numpy().tolist()
            preds.append(xyxy)
    return preds

def detect_players_once(det_model, img, device, imgsz, conf, iou):
    half = (device == 0) and torch.cuda.is_available()
    with torch.inference_mode():
        out = det_model.predict(
            source=img, imgsz=imgsz, conf=conf, iou=iou,
            device=device, half=half, verbose=False, stream=False, workers=0, classes=[CLS_PLAYER]
        )
    if not out: return []
    res = out[0]
    if res.boxes is None or len(res.boxes) == 0: return []
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    confs = res.boxes.conf.detach().cpu().numpy()
    return [(xyxy[i].tolist(), float(confs[i])) for i in range(len(confs))]

def main():
    frames = load_frame_list(IMAGES_DIR, SECUENCIA)
    if not frames:
        raise FileNotFoundError(f"No frames {SECUENCIA}_* en {IMAGES_DIR}")
    frame_assign = load_frame_assignments(FRAME_ASSIGN_JSON)

    if not BALL_MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"No existe: {BALL_MODEL_WEIGHTS}")
    ball_model = YOLO(str(BALL_MODEL_WEIGHTS))
    ball_model.to(DEVICE)

    det_model = None
    if USE_FALLBACK_DET and DET_MODEL_WEIGHTS.exists():
        det_model = YOLO(str(DET_MODEL_WEIGHTS))
        det_model.to(DEVICE)

    sample = cv2.imread(str(frames[0]))
    H, W = sample.shape[:2]
    vw = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    legend = [("Team 0", COLOR_TEAM0), ("Team 1", COLOR_TEAM1), ("GK", COLOR_GK), ("Ref", COLOR_REF), ("Ball", COLOR_BALL)]

    ball_preds = predict_ball_stream(ball_model, frames, DEVICE, IMGSZ_BALL, CONF_BALL, IOU_BALL)

    last_seen = {}  # tid -> {"bbox": [x1,y1,x2,y2], "team": int, "age": int}

    for fidx, img_path in enumerate(frames):
        img = cv2.imread(str(img_path))
        overlay = img.copy()

        objs = frame_assign.get(fidx, [])
        present_tids = set()

        for obj in objs:
            cls_id = obj["cls"]
            xyxy = obj["bbox"]
            if cls_id == CLS_PLAYER:
                team = int(obj.get("team_id", -1))
                tid = int(obj.get("track_id", -1))
                color = COLOR_TEAM0 if team == 0 else COLOR_TEAM1
                draw_box(overlay, xyxy, color, f"ID {tid} | T{team}")
                present_tids.add(tid)
                last_seen[tid] = {"bbox": xyxy, "team": team, "age": 0}
            elif cls_id == CLS_GK:
                draw_box(overlay, xyxy, COLOR_GK, "GK")
            elif cls_id == CLS_REF:
                draw_box(overlay, xyxy, COLOR_REF, "Ref")

        # persistencia de jugadores ausentes
        synthetic_count = 0
        to_delete = []
        for tid, st in last_seen.items():
            if tid in present_tids:
                continue
            if st["age"] < KEEP_GAP:
                draw_box(overlay, st["bbox"], COLOR_TEAM0 if st["team"] == 0 else COLOR_TEAM1, f"ID {tid}* | T{st['team']}")
                st["age"] += 1
                synthetic_count += 1
            else:
                to_delete.append(tid)
        for tid in to_delete:
            last_seen.pop(tid, None)

        # respaldo con detector si faltan muchos
        if det_model and ((len(objs) + synthetic_count) < MIN_PLAYERS_FOR_FALLBACK) and (fidx % FALLBACK_EVERY == 0):
            dets = detect_players_once(det_model, str(img_path), DEVICE, IMGSZ_DET, CONF_DET, IOU_DET)
            for det_xyxy, det_conf in dets:
                best_tid, best_iou = None, 0.0
                for tid, st in last_seen.items():
                    if tid in present_tids:  # ya dibujado real
                        continue
                    i = iou_xyxy(det_xyxy, st["bbox"])
                    if i > best_iou:
                        best_iou, best_tid = i, tid
                if best_tid is not None and best_iou >= 0.4:
                    team = last_seen[best_tid]["team"]
                    draw_box(overlay, det_xyxy, COLOR_TEAM0 if team == 0 else COLOR_TEAM1, f"ID {best_tid}~ | T{team}")
                    present_tids.add(best_tid)
                    last_seen[best_tid] = {"bbox": det_xyxy, "team": team, "age": 0}
                else:
                    draw_box(overlay, det_xyxy, COLOR_NEUTRAL, "player?")

        ball_xyxy = ball_preds[fidx]
        if ball_xyxy is not None:
            draw_box(overlay, ball_xyxy, COLOR_BALL, "Ball")

        y0 = 25
        for name, col in legend:
            cv2.putText(overlay, name, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            y0 += 22

        vw.write(overlay)

    vw.release()
    print(f"Video: {OUT_VIDEO}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
