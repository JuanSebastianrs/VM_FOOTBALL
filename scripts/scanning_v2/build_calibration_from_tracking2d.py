# scripts/scanning_v2/build_calibration_from_tracking2d.py
"""
Reconstruye `calibration_hinv.json` (homografia imagen->cancha por frame) a
partir de outputs ya existentes del pipeline tactico, SIN re-ejecutar PnLCalib.

Idea: `<SEQ>_tracking_2d.csv` ya contiene la posicion de cada track en metros
(origen esquina superior-izquierda, cancha 105x68) producida por el mapper 2D
con calibracion PnLCalib suavizada. `<SEQ>_detections.json` tiene el bbox de
cada track en imagen. Emparejando (punto-pie del bbox) <-> (x_m, y_m) por
(frame_id, track_id) se ajusta una homografia por frame con RANSAC.

Convencion de salida (identica a la de PnLCalib usada por HomographyLookup):
  H_inv @ [x_img, y_img, 1] -> coords de cancha CENTRADAS en metros
  (X in [-52.5, 52.5], Y in [-34, 34], Y hacia abajo).

Gates de calidad por frame (si no pasan, el frame se omite y HomographyLookup
usara el frame valido mas cercano <=5 frames):
  - >= min_points correspondencias no colineales
  - >= min_inliers inliers RANSAC
  - error mediano de reproyeccion (m) <= max_median_err_m

Uso:
  python scripts/scanning_v2/build_calibration_from_tracking2d.py \
      --video_id SNMOT-116 --outputs_root outputs \
      [--validate_against outputs/SNMOT-148/calibration_hinv.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PITCH_L, PITCH_W = 105.0, 68.0


def foot_point(p: dict) -> tuple:
    return (0.5 * (p["x_min"] + p["x_max"]), p["y_max"])


def fit_frame_homography(img_pts, world_pts, ransac_thr_m=1.5,
                         min_points=6, min_inliers=5, max_median_err_m=2.0):
    """Ajusta H imagen->cancha centrada para un frame. None si no es fiable."""
    if len(img_pts) < min_points:
        return None, "few_points"
    src = np.asarray(img_pts, dtype=np.float64)
    dst = np.asarray(world_pts, dtype=np.float64)
    # colinealidad: el area del hull de los puntos imagen debe ser razonable
    hull = cv2.convexHull(src.astype(np.float32))
    if cv2.contourArea(hull) < 5000.0:      # px^2; puntos casi colineales
        return None, "degenerate"
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr_m)
    if H is None or mask is None or int(mask.sum()) < min_inliers:
        return None, "ransac_failed"
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)
    med = float(np.median(err[mask.ravel().astype(bool)]))
    if med > max_median_err_m:
        return None, f"median_err {med:.2f}m"
    return H, None


def build_for_video(video_id: str, outputs_root: str, fps: float = 25.0,
                    overwrite: bool = False) -> dict:
    vdir = os.path.join(outputs_root, video_id)
    out_path = os.path.join(vdir, "calibration_hinv.json")
    if os.path.exists(out_path) and not overwrite:
        return {"video_id": video_id, "status": "exists", "path": out_path}

    det_path = os.path.join(vdir, f"{video_id}_detections.json")
    t2d_path = os.path.join(vdir, f"{video_id}_tracking_2d.csv")
    if not (os.path.exists(det_path) and os.path.exists(t2d_path)):
        return {"video_id": video_id, "status": "missing_inputs"}

    with open(det_path, "r", encoding="utf-8") as f:
        detections = json.load(f)
    t2d = pd.read_csv(t2d_path)
    t2d = t2d[(t2d["entity_type"] == "player") & (t2d["visible"] == 1)]
    pos = {(int(r.frame_id), int(r.track_id)): (float(r.x_m), float(r.y_m))
           for r in t2d.itertuples()}

    calib, reasons = {}, {}
    for fr in detections:
        fid = int(fr["frame_id"])
        img_pts, world_pts = [], []
        for p in fr.get("players", []):
            key = (fid, int(p["track_id"]))
            if key not in pos:
                continue
            x_m, y_m = pos[key]
            if not (0.0 <= x_m <= PITCH_L and 0.0 <= y_m <= PITCH_W):
                continue
            img_pts.append(foot_point(p))
            world_pts.append((x_m - PITCH_L / 2.0, y_m - PITCH_W / 2.0))
        H, why = fit_frame_homography(img_pts, world_pts)
        if H is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        calib[str(fid)] = {"H_inv": H.tolist(), "time_s": round((fid - 1) / fps, 4)}

    status = "ok" if calib else "no_frames"
    if calib:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(calib, f)
    return {"video_id": video_id, "status": status, "path": out_path,
            "frames_total": len(detections), "frames_calibrated": len(calib),
            "skipped": reasons}


def validate(fitted_path: str, reference_path: str, n_grid: int = 25) -> dict:
    """Compara proyecciones de una rejilla imagen fija: H ajustada vs referencia."""
    with open(fitted_path, "r", encoding="utf-8") as f:
        fit = json.load(f)
    with open(reference_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    common = sorted(set(fit) & set(ref), key=int)
    xs = np.linspace(200, 1700, 5)
    ys = np.linspace(400, 1000, 5)
    grid = np.array([[x, y, 1.0] for x in xs for y in ys]).T
    errs = []
    for k in common:
        Hf = np.asarray(fit[k]["H_inv"])
        Hr = np.asarray(ref[k]["H_inv"] if isinstance(ref[k], dict) else ref[k])
        pf, pr = Hf @ grid, Hr @ grid
        pf, pr = pf[:2] / pf[2], pr[:2] / pr[2]
        # solo puntos que caen dentro de la cancha segun la referencia
        inside = (np.abs(pr[0]) <= 55) & (np.abs(pr[1]) <= 37)
        if inside.sum() < 5:
            continue
        errs.append(float(np.median(np.linalg.norm(pf[:, inside] - pr[:, inside],
                                                   axis=0))))
    return {"frames_compared": len(errs),
            "median_err_m": float(np.median(errs)) if errs else None,
            "p90_err_m": float(np.percentile(errs, 90)) if errs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", action="append", required=True,
                    help="repetible; o 'ALL' para todo outputs_root")
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate_against", default=None,
                    help="calibration_hinv.json de referencia (mismo video)")
    args = ap.parse_args()

    vids = args.video_id
    if vids == ["ALL"]:
        vids = sorted(d for d in os.listdir(args.outputs_root)
                      if os.path.isfile(os.path.join(
                          args.outputs_root, d, f"{d}_tracking_2d.csv")))
    results = []
    for v in vids:
        r = build_for_video(v, args.outputs_root, args.fps, args.overwrite)
        results.append(r)
        print(json.dumps(r))
    if args.validate_against and len(vids) == 1:
        fitted = os.path.join(args.outputs_root, vids[0], "calibration_hinv.json")
        print("validation:", json.dumps(validate(fitted, args.validate_against)))
    ok = sum(1 for r in results if r["status"] in ("ok", "exists"))
    print(f"\n{ok}/{len(results)} videos calibrados")


if __name__ == "__main__":
    main()
