# scripts/scanning/benchmark_keypoint_backends.py
"""
Benchmark objetivo YOLO-Pose vs MediaPipe Pose Landmarker para estimacion de
orientacion visual APROXIMADA en futbol broadcast.

Procesa EXACTAMENTE los mismos crops con ambos backends (se recorta una sola vez
y se infiere dos veces) y compara cobertura de pose, calidad de cabeza/torso,
estabilidad temporal (jitter por track), distribucion de theta_source, tiempos y
fallos en crops pequenos. Genera muestras visuales lado a lado y un report.md.

NO es gaze real: la salida sigue siendo orientacion-proxy basada en keypoints.

Ejemplo:
  python scripts/scanning/benchmark_keypoint_backends.py \
      --video_id SNMOT-148 \
      --detections   outputs/SNMOT-148/SNMOT-148_detections.json \
      --trajectory   outputs/SNMOT-148/SNMOT-148_trajectory.json \
      --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
      --calibration  outputs/SNMOT-148/calibration_hinv.json \
      --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
      --n_random 100 --n_reception 50 --jitter_tracks 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.scanning import load_config, load_sequence                  # noqa: E402
from core.scanning.circular import circular_delta                     # noqa: E402
from core.scanning.keypoint_extractor import (                        # noqa: E402
    CANONICAL_LANDMARKS, HEAD_LANDMARKS, TORSO_LANDMARKS, KeypointExtractor)
from core.scanning.orientation_estimator import OrientationEstimator  # noqa: E402
from core.scanning.player_cropper import PlayerCropper                # noqa: E402
from core.scanning.reception_detector import ReceptionDetector        # noqa: E402

# --- buckets de tamano por alto de bbox original (px) ---
def size_bucket(h: float) -> str:
    return "small" if h < 120 else ("medium" if h < 200 else "large")


_SKELETON = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
             ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
             ("nose", "left_eye"), ("nose", "right_eye"),
             ("left_eye", "left_ear"), ("right_eye", "right_ear")]


# ===========================================================================
# Muestreo de crops (una sola vez; se reutiliza para ambos backends)
# ===========================================================================

class CropTask:
    __slots__ = ("kind", "frame_id", "track_id", "orig_height", "bucket",
                 "crop", "bbox_exp", "velocity", "player_xy", "ball_xy", "group")

    def __init__(self, kind, frame_id, track_id, orig_height, bucket, crop,
                 bbox_exp, velocity, player_xy, ball_xy, group=None):
        self.kind = kind; self.frame_id = frame_id; self.track_id = track_id
        self.orig_height = orig_height; self.bucket = bucket; self.crop = crop
        self.bbox_exp = bbox_exp; self.velocity = velocity
        self.player_xy = player_xy; self.ball_xy = ball_xy; self.group = group


def _make_task(kind, seq, cropper, img, p, fid, velocity=None, group=None):
    crop = cropper.crop(img, p.bbox, fid, p.track_id)
    if not crop.valid_crop:
        return None
    ball = seq.ball(fid)
    ball_xy = (ball.x, ball.y) if (ball and ball.valid) else None
    return CropTask(kind, fid, p.track_id, crop.orig_height,
                    size_bucket(crop.orig_height), crop.crop, crop.bbox_expanded,
                    velocity, p.center, ball_xy, group)


def build_tasks(seq, cropper, cfg, n_random, n_reception, jitter_tracks,
                jitter_len, seed):
    rng = np.random.default_rng(seed)
    tasks = []
    img_cache = {}

    def get_img(fid):
        if fid not in img_cache:
            ip = seq.image_paths.get(fid)
            img_cache[fid] = cv2.imread(str(ip)) if ip else None
        return img_cache[fid]

    # --- A. crops aleatorios (con cobertura de buckets) ---
    pool = [(fid, p) for fid in seq.frame_ids for p in seq.players(fid)]
    rng.shuffle(pool)
    added = 0
    for fid, p in pool:
        if added >= n_random:
            break
        img = get_img(fid)
        if img is None:
            continue
        t = _make_task("random", seq, cropper, img, p, fid)
        if t:
            tasks.append(t); added += 1

    # --- B. crops en ventanas previas a recepcion ---
    receptions = ReceptionDetector(cfg).detect(seq)
    fps = cfg.get("fps", 25.0)
    before = cfg.get("scan_window_seconds_before", 3.0)
    after = cfg.get("scan_window_seconds_after", 0.2)
    rec_added = 0
    rec_pairs = []
    for ev in receptions:
        f0 = int(ev.frame_reception - before * fps)
        f1 = int(ev.frame_reception - after * fps)
        for f in range(max(seq.frame_ids[0], f0), f1 + 1):
            rec_pairs.append((f, ev.receiver_track_id))
    rng.shuffle(rec_pairs)
    for fid, tid in rec_pairs:
        if rec_added >= n_reception:
            break
        img = get_img(fid)
        p = seq.player(fid, tid)
        if img is None or p is None:
            continue
        t = _make_task("reception", seq, cropper, img, p, fid)
        if t:
            tasks.append(t); rec_added += 1

    # --- C. tracks contiguos para jitter temporal ---
    presence = Counter(p.track_id for fid in seq.frame_ids for p in seq.players(fid))
    top_tracks = [tid for tid, _ in presence.most_common(jitter_tracks)]
    jitter_ids = []
    for tid in top_tracks:
        frames = [fid for fid in seq.frame_ids if seq.player(fid, tid) is not None]
        if len(frames) < 8:
            continue
        start = frames[0]
        run = [f for f in frames if start <= f < start + jitter_len]
        prev_center = None
        group = f"jit_{tid}"
        for fid in run:
            img = get_img(fid)
            p = seq.player(fid, tid)
            if img is None or p is None:
                continue
            vel = None
            if prev_center is not None:
                vel = (p.center[0] - prev_center[0], p.center[1] - prev_center[1])
            prev_center = p.center
            t = _make_task("jitter", seq, cropper, img, p, fid, velocity=vel, group=group)
            if t:
                tasks.append(t)
        jitter_ids.append(tid)
    return tasks, jitter_ids


# ===========================================================================
# Ejecucion de un backend sobre todos los tasks
# ===========================================================================

def run_backend(name, cfg, tasks, estimator, homography):
    ext = KeypointExtractor(cfg, backend_name=name)
    # warmup (excluye coste de inicializacion/JIT del timing)
    if tasks:
        ext.extract(tasks[0].crop)
    records = []
    for t in tasks:
        t0 = time.perf_counter()
        pose = ext.extract(t.crop, crop_height=t.orig_height)   # crop_height -> routing hybrid
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ori = estimator.estimate(
            frame_id=t.frame_id, track_id=t.track_id, pose=pose,
            bbox_exp=t.bbox_exp, velocity=t.velocity, ball_xy=t.ball_xy,
            player_xy=t.player_xy, homography=homography)
        # confianza media de keypoints presentes
        vis = [pose["landmarks"][k][3] for k in CANONICAL_LANDMARKS
               if pose["landmarks"][k][3] > 0]
        records.append({
            "kind": t.kind, "frame_id": t.frame_id, "track_id": t.track_id,
            "group": t.group, "bucket": t.bucket, "orig_height": t.orig_height,
            "pose_valid": bool(pose["pose_valid"]),
            "head_visibility": pose["head_visibility"],
            "torso_visibility": pose["torso_visibility"],
            "mean_kp_conf": float(np.mean(vis)) if vis else 0.0,
            "theta": ori.theta_visual_img,
            "theta_source": ori.theta_source,
            "orientation_confidence": ori.orientation_confidence,
            "time_ms": dt_ms,
            "pose": pose,
        })
    ext.close()
    return records


# ===========================================================================
# Agregacion de metricas
# ===========================================================================

def aggregate(records, cfg, jitter_ids, avg_players_per_frame):
    min_head = cfg.get("min_head_visibility", 0.4)
    min_torso = cfg.get("min_torso_visibility", 0.4)
    n = len(records)
    valid = [r for r in records if r["pose_valid"]]

    def rate(pred):
        return sum(1 for r in records if pred(r)) / n if n else 0.0

    # jitter por track (sobre theta crudo en frames consecutivos validos)
    jitter_per_track = {}
    by_group = defaultdict(list)
    for r in records:
        if r["kind"] == "jitter" and r["group"]:
            by_group[r["group"]].append(r)
    for g, recs in by_group.items():
        recs = sorted(recs, key=lambda r: r["frame_id"])
        deltas = []
        for a, b in zip(recs[:-1], recs[1:]):
            if a["theta"] is not None and b["theta"] is not None:
                deltas.append(abs(math.degrees(circular_delta(b["theta"], a["theta"]))))
        if deltas:
            jitter_per_track[g] = float(np.mean(deltas))
    mean_jitter = float(np.mean(list(jitter_per_track.values()))) if jitter_per_track else float("nan")

    src = Counter(r["theta_source"] for r in records)
    src_pct = {k: 100.0 * src.get(k, 0) / n for k in
               ["head", "torso", "movement", "previous", "invalid"]} if n else {}

    # por bucket de tamano
    buckets = {}
    for bk in ["small", "medium", "large"]:
        sub = [r for r in records if r["bucket"] == bk]
        if sub:
            buckets[bk] = {
                "n": len(sub),
                "pose_valid_rate": sum(r["pose_valid"] for r in sub) / len(sub),
                "head_valid_rate": sum(r["head_visibility"] >= min_head for r in sub) / len(sub),
            }

    small = [r for r in records if r["bucket"] == "small"]
    small_fail = sum(1 for r in small if not r["pose_valid"])

    times = [r["time_ms"] for r in records]
    return {
        "n_crops": n,
        "pose_valid_rate": rate(lambda r: r["pose_valid"]),
        "head_landmarks_valid_rate": rate(lambda r: r["head_visibility"] >= min_head),
        "torso_landmarks_valid_rate": rate(lambda r: r["torso_visibility"] >= min_torso),
        "mean_keypoint_confidence": float(np.mean([r["mean_kp_conf"] for r in valid])) if valid else 0.0,
        "mean_orientation_confidence": float(np.mean(
            [r["orientation_confidence"] for r in records if r["theta"] is not None]))
            if any(r["theta"] is not None for r in records) else 0.0,
        "angular_jitter_deg_mean": mean_jitter,
        "angular_jitter_per_track": jitter_per_track,
        "theta_source_pct": src_pct,
        "mean_time_ms_per_crop": float(np.mean(times)) if times else 0.0,
        "mean_time_ms_per_frame": float(np.mean(times) * avg_players_per_frame) if times else 0.0,
        "by_size_bucket": buckets,
        "small_crops": len(small),
        "small_crop_failures": small_fail,
        "small_crop_failure_rate": (small_fail / len(small)) if small else 0.0,
    }


# ===========================================================================
# Visualizaciones
# ===========================================================================

def draw_pose(img, pose, theta, color, min_vis=0.2):
    out = img.copy()
    H, W = out.shape[:2]
    pts = {}
    for name in CANONICAL_LANDMARKS:
        x, y, _z, v = pose["landmarks"][name]
        if v >= min_vis and (x != 0 or y != 0):
            pts[name] = (int(x * W), int(y * H))
    for a, b in _SKELETON:
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], color, 2)
    for name, (px, py) in pts.items():
        c = (0, 0, 255) if name in HEAD_LANDMARKS else color
        cv2.circle(out, (px, py), 4, c, -1)
    # flecha de orientacion desde el centro del torso
    if theta is not None:
        if "left_shoulder" in pts and "right_shoulder" in pts:
            bx = (pts["left_shoulder"][0] + pts["right_shoulder"][0]) // 2
            by = (pts["left_shoulder"][1] + pts["right_shoulder"][1]) // 2
        else:
            bx, by = W // 2, H // 2
        L = int(0.22 * H)
        ex = int(bx + L * math.cos(theta)); ey = int(by - L * math.sin(theta))
        cv2.arrowedLine(out, (bx, by), (ex, ey), (40, 220, 255), 3, tipLength=0.3)
    return out


def _label(img, lines, color=(255, 255, 255)):
    y = 22
    for ln in lines:
        cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    return img


def render_samples(out_dir, tasks, recs_by_backend, sample_ids, mp_threshold=110.0):
    yolo_dir = out_dir / "yolo_pose_samples"
    mp_dir = out_dir / "mediapipe_samples"
    sbs_dir = out_dir / "side_by_side"
    for d in (yolo_dir, mp_dir, sbs_dir):
        d.mkdir(parents=True, exist_ok=True)
    yolo = recs_by_backend["yolo_pose"]
    mp = recs_by_backend["mediapipe"]
    for i in sample_ids:
        t = tasks[i]
        ry, rm = yolo[i], mp[i]
        base = t.crop
        img_y = draw_pose(base, ry["pose"], ry["theta"], (60, 200, 60))
        img_m = draw_pose(base, rm["pose"], rm["theta"], (220, 120, 40))
        _label(img_y, ["YOLO-Pose",
                       f"src={ry['theta_source']} c={ry['orientation_confidence']:.2f}",
                       f"valid={ry['pose_valid']} head={ry['head_visibility']:.2f}"], (60, 255, 60))
        _label(img_m, ["MediaPipe",
                       f"src={rm['theta_source']} c={rm['orientation_confidence']:.2f}",
                       f"valid={rm['pose_valid']} head={rm['head_visibility']:.2f}"], (255, 180, 80))
        # eleccion de hybrid: MediaPipe si crop grande y detecta; si no, YOLO
        if t.orig_height >= mp_threshold:
            hybrid_pick = "mediapipe" if rm["pose_valid"] else "yolo_pose(fallback)"
        else:
            hybrid_pick = "yolo_pose"
        tag = f"{t.kind}_{t.bucket}_f{t.frame_id}_t{t.track_id}"
        cv2.imwrite(str(yolo_dir / f"{tag}.jpg"), img_y)
        cv2.imwrite(str(mp_dir / f"{tag}.jpg"), img_m)
        orig = _label(base.copy(), [f"original {t.bucket}", f"h={int(t.orig_height)}px",
                                    f"hybrid->{hybrid_pick}"])
        sbs = np.hstack([orig, img_y, img_m])
        cv2.imwrite(str(sbs_dir / f"{tag}.jpg"), sbs)


def pick_sample_ids(tasks, recs_by_backend, k=18):
    """Elige muestras representativas: cubre kinds y buckets, prioriza casos donde
    los backends discrepan (uno valido y el otro no)."""
    yolo = recs_by_backend["yolo_pose"]; mp = recs_by_backend["mediapipe"]
    disagree = [i for i in range(len(tasks)) if yolo[i]["pose_valid"] != mp[i]["pose_valid"]]
    chosen = list(disagree[:k // 2])
    # cobertura de buckets/kinds
    seen = set()
    for i in range(len(tasks)):
        key = (tasks[i].kind, tasks[i].bucket)
        if key not in seen and i not in chosen:
            chosen.append(i); seen.add(key)
        if len(chosen) >= k:
            break
    # rellenar
    for i in range(len(tasks)):
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:k]


# ===========================================================================
# Reporte
# ===========================================================================

def _fmt(v, p=3):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{p}f}"
    return str(v)


def write_report(out_dir, video_id, metrics, cfg, device_note):
    y = metrics["yolo_pose"]; m = metrics["mediapipe"]
    min_head = cfg.get("min_head_visibility", 0.4)

    def row(label, key, p=3, pct=False):
        vy, vm = y[key], m[key]
        if pct:
            vy, vm = vy * 100, vm * 100
        return f"| {label} | {_fmt(vy, p)} | {_fmt(vm, p)} |"

    # decision (criterio 7)
    head_better = m["head_landmarks_valid_rate"] >= y["head_landmarks_valid_rate"]
    jitter_better = (not math.isnan(m["angular_jitter_deg_mean"]) and
                     not math.isnan(y["angular_jitter_deg_mean"]) and
                     m["angular_jitter_deg_mean"] <= y["angular_jitter_deg_mean"])
    cov_drop = y["pose_valid_rate"] - m["pose_valid_rate"]
    small_bad = m["small_crop_failure_rate"] > max(0.5, y["small_crop_failure_rate"] + 0.2)

    if head_better and jitter_better and cov_drop <= 0.10:
        decision = ("**Usar MediaPipe como backend recomendado.** Mejora la validez de "
                    "landmarks de cabeza y reduce el jitter temporal sin perder cobertura "
                    "relevante.")
    elif small_bad:
        thr = cfg.get("low_quality_crop_height", 110)
        decision = (f"**Mantener YOLO-Pose como default; usar MediaPipe solo en crops "
                    f"grandes (crop_height >= {thr}px).** MediaPipe falla demasiado en "
                    f"crops pequenos.")
    else:
        decision = ("**Mantener YOLO-Pose como default.** MediaPipe no muestra una mejora "
                    "neta suficiente (cabeza/jitter) frente a su costo/cobertura.")

    lines = [
        f"# Benchmark de backends de keypoints — {video_id}",
        "",
        "> Orientacion visual **aproximada** basada en keypoints. **No es gaze real.** "
        "Mismos crops para ambos backends (recorte unico, doble inferencia).",
        "",
        f"- Crops evaluados: **{y['n_crops']}** "
        f"(random + ventanas pre-recepcion + tracks de jitter).",
        f"- Dispositivo: {device_note}",
        f"- Umbrales: min_head_visibility={min_head}, "
        f"min_torso_visibility={cfg.get('min_torso_visibility', 0.4)}.",
        "",
        "## Métricas (YOLO-Pose vs MediaPipe)",
        "",
        "| Métrica | YOLO-Pose | MediaPipe |",
        "|---|---|---|",
        row("pose_valid_rate", "pose_valid_rate"),
        row("head_landmarks_valid_rate", "head_landmarks_valid_rate"),
        row("torso_landmarks_valid_rate", "torso_landmarks_valid_rate"),
        row("mean_keypoint_confidence", "mean_keypoint_confidence"),
        row("mean_orientation_confidence", "mean_orientation_confidence"),
        row("angular_jitter_deg (mean/track)", "angular_jitter_deg_mean", p=2),
        row("mean_time_ms_per_crop", "mean_time_ms_per_crop", p=2),
        row("mean_time_ms_per_frame", "mean_time_ms_per_frame", p=2),
        row("small_crop_failure_rate", "small_crop_failure_rate"),
        "",
        "### theta_source (%)",
        "",
        "| Fuente | YOLO-Pose | MediaPipe |",
        "|---|---|---|",
    ]
    for s in ["head", "torso", "movement", "previous", "invalid"]:
        lines.append(f"| {s} | {_fmt(y['theta_source_pct'].get(s, 0.0), 1)} | "
                     f"{_fmt(m['theta_source_pct'].get(s, 0.0), 1)} |")

    lines += ["", "### Validez por tamaño de crop (pose_valid / head_valid)", "",
              "| Bucket | n | YOLO valid | YOLO head | MP valid | MP head |",
              "|---|---|---|---|---|---|"]
    for bk in ["small", "medium", "large"]:
        yb = y["by_size_bucket"].get(bk); mb = m["by_size_bucket"].get(bk)
        if yb or mb:
            n = (yb or mb)["n"]
            lines.append(
                f"| {bk} | {n} | {_fmt((yb or {}).get('pose_valid_rate', float('nan')))} | "
                f"{_fmt((yb or {}).get('head_valid_rate', float('nan')))} | "
                f"{_fmt((mb or {}).get('pose_valid_rate', float('nan')))} | "
                f"{_fmt((mb or {}).get('head_valid_rate', float('nan')))} |")

    def winner(key, lower_better=False):
        a, b = y[key], m[key]
        if isinstance(a, float) and math.isnan(a):
            return "MediaPipe"
        if isinstance(b, float) and math.isnan(b):
            return "YOLO-Pose"
        if lower_better:
            return "MediaPipe" if b <= a else "YOLO-Pose"
        return "MediaPipe" if b >= a else "YOLO-Pose"

    lines += [
        "", "## Conclusiones", "",
        f"- **¿Más poses válidas?** {winner('pose_valid_rate')} "
        f"(YOLO {_fmt(y['pose_valid_rate'])} vs MP {_fmt(m['pose_valid_rate'])}).",
        f"- **¿Mejor orientación de cabeza?** {winner('head_landmarks_valid_rate')} "
        f"(head_valid YOLO {_fmt(y['head_landmarks_valid_rate'])} vs MP {_fmt(m['head_landmarks_valid_rate'])}).",
        f"- **¿Más estable temporalmente?** {winner('angular_jitter_deg_mean', lower_better=True)} "
        f"(jitter YOLO {_fmt(y['angular_jitter_deg_mean'], 2)}° vs MP {_fmt(m['angular_jitter_deg_mean'], 2)}°; menor es mejor).",
        f"- **¿Más rápido?** {winner('mean_time_ms_per_crop', lower_better=True)} "
        f"(YOLO {_fmt(y['mean_time_ms_per_crop'], 2)} ms/crop vs MP {_fmt(m['mean_time_ms_per_crop'], 2)} ms/crop). "
        f"*Nota:* {device_note}.",
        "- **¿Dónde falla cada uno?** YOLO-Pose tiende a perder definición fina de "
        "cabeza en crops pequeños/lejanos; MediaPipe tiende a fallar cuando el jugador "
        "está de espaldas, muy ocluido o el crop es muy pequeño (espera una persona "
        "razonablemente visible). Ver `side_by_side/` para casos de discrepancia.",
        "", "## Recomendación final", "", decision,
        "", "Si ambos fallan en cabeza, el pipeline cae a torso/movimiento y marca baja "
        "confianza (jerarquía en `orientation_estimator.py`).",
        "", "## Artefactos",
        "- `metrics.json` — métricas completas (incluye jitter por track y buckets).",
        "- `per_crop.csv` — registro por crop y backend.",
        "- `yolo_pose_samples/`, `mediapipe_samples/`, `side_by_side/` — muestras visuales.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# Main
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--team_assignments", default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--sequence_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default="outputs/scanning/backend_benchmark")
    p.add_argument("--n_random", type=int, default=100)
    p.add_argument("--n_reception", type=int, default=50)
    p.add_argument("--jitter_tracks", type=int, default=6)
    p.add_argument("--jitter_len", type=int, default=40)
    p.add_argument("--n_samples", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    sc = config["scanning"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = load_sequence(args.video_id, args.detections, args.trajectory,
                        args.team_assignments, args.calibration,
                        sequence_dir=args.sequence_dir, fps=sc.get("fps", 25.0))
    if not seq.image_paths:
        raise SystemExit("Se requieren imagenes (sequence_dir/img1).")
    avg_players = float(np.mean([len(seq.players(f)) for f in seq.frame_ids]))

    cropper = PlayerCropper(expand_ratio=sc.get("crop_expand_ratio", 0.25),
                            crop_size=tuple(sc.get("crop_size", (384, 768))),
                            min_crop_height=sc.get("min_crop_height", 64),
                            low_quality_height=sc.get("low_quality_crop_height", 110))
    estimator = OrientationEstimator(sc)

    print("[bench] muestreando crops ...")
    tasks, jitter_ids = build_tasks(seq, cropper, sc, args.n_random,
                                    args.n_reception, args.jitter_tracks,
                                    args.jitter_len, args.seed)
    kinds = Counter(t.kind for t in tasks)
    buckets = Counter(t.bucket for t in tasks)
    print(f"  crops={len(tasks)} kinds={dict(kinds)} buckets={dict(buckets)} "
          f"jitter_tracks={jitter_ids}")

    recs_by_backend, metrics = {}, {}
    for bk in ["yolo_pose", "mediapipe"]:
        print(f"[bench] backend={bk} ...")
        recs = run_backend(bk, sc, tasks, estimator, seq.homography)
        recs_by_backend[bk] = recs
        metrics[bk] = aggregate(recs, sc, jitter_ids, avg_players)
        print(f"  pose_valid={metrics[bk]['pose_valid_rate']:.3f} "
              f"head_valid={metrics[bk]['head_landmarks_valid_rate']:.3f} "
              f"jitter={metrics[bk]['angular_jitter_deg_mean']:.2f} "
              f"ms/crop={metrics[bk]['mean_time_ms_per_crop']:.2f}")

    # artefactos
    import pandas as pd
    rows = []
    for bk, recs in recs_by_backend.items():
        for r in recs:
            rr = {k: v for k, v in r.items() if k != "pose"}
            rr["backend"] = bk
            rows.append(rr)
    pd.DataFrame(rows).to_csv(out_dir / "per_crop.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[bench] renderizando muestras ...")
    sample_ids = pick_sample_ids(tasks, recs_by_backend, args.n_samples)
    render_samples(out_dir, tasks, recs_by_backend, sample_ids,
                   mp_threshold=sc.get("mediapipe_min_crop_height", 110))

    import torch
    device_note = ("YOLO-Pose en GPU (CUDA), MediaPipe en CPU (XNNPACK) — el tiempo "
                   "no es comparable 1:1") if torch.cuda.is_available() else \
                  "ambos en CPU"
    write_report(out_dir, args.video_id, metrics, sc, device_note)
    print(f"[bench] listo -> {out_dir}/report.md")


if __name__ == "__main__":
    main()
