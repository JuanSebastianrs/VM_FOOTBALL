# scripts/analyze_lie_smoothing.py
"""
Analisis cuantitativo del suavizado de calibracion del mapa 2D.

Consume el dump de mediciones crudas de PnLCalib
(`tactical_vision_2d_mapper.py --dump_raw_calib`) y compara variantes:

  raw        sin suavizar (lo que mide PnLCalib frame a frame)
  ema_fwd    EMA puro forward sin prediccion (comportamiento del viejo
             CameraParamsSmoother, misma parametrizacion de la clase actual)
  lie_fwd    pasada forward predictiva (ESKF-Lite) sola
  lie_bidir  PRODUCCION: forward predictivo + backward EMA + merge geodesico
  gauss      patron oro offline: rechazo por rep_err, interpolacion y
             filtro gaussiano de fase cero sobre rotvec/focal/pos

Metricas (sobre puntos de imagen proyectados al plano del campo):
  jitter  RMS de la segunda diferencia (m/frame^2) — ruido de alta frecuencia
  lag     RMS de distancia a la trayectoria cruda (m) — retardo/deriva
  peak    desviacion maxima vs cruda (m)

  python scripts/analyze_lie_smoothing.py \
      --raw_calib outputs/_verify/lie_review/raw_calib_SNMOT-148.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mapping.tactical_vision_2d_mapper import (  # noqa: E402
    BidirectionalLieSmoother)

# Puntos de imagen a proyectar (1920x1080, mitad inferior = campo casi seguro)
IMAGE_POINTS = np.array([
    [960.0, 980.0, 1.0],
    [480.0, 820.0, 1.0],
    [1440.0, 820.0, 1.0],
], dtype=np.float64).T  # (3, K)


def measurements_from_dump(dump):
    """Reconstruye la lista de mediciones como las consume el smoother."""
    ms = []
    for m in dump["measurements"]:
        if m is None:
            ms.append(None)
            continue
        ms.append({
            "R": Rotation.from_matrix(np.array(m["rotation_matrix"])),
            "fx": m["x_focal_length"], "fy": m["y_focal_length"],
            "cx": m["principal_point"][0], "cy": m["principal_point"][1],
            "pos": np.array(m["position_meters"], dtype=np.float64),
            "rep_err": m["rep_err"], "source": m["source"],
        })
    return ms


def params_to_hinv(params):
    out = []
    for p in params:
        if p is None:
            out.append(None)
            continue
        out.append(BidirectionalLieSmoother._build_H_inv_static(
            p["R"], p["fx"], p["fy"], p["cx"], p["cy"], p["pos"]))
    return out


def project_series(h_invs):
    """(T, K, 2) coordenadas de campo de los IMAGE_POINTS; NaN si no hay H."""
    T = len(h_invs)
    K = IMAGE_POINTS.shape[1]
    out = np.full((T, K, 2), np.nan)
    for t, H in enumerate(h_invs):
        if H is None:
            continue
        w = H @ IMAGE_POINTS
        out[t] = (w[:2] / w[2:3]).T
    return out


def run_variant_raw(ms):
    return params_to_hinv([m if m is None else {
        "R": m["R"], "fx": m["fx"], "fy": m["fy"],
        "cx": m["cx"], "cy": m["cy"], "pos": m["pos"]} for m in ms])


def run_variant_pass(ms, use_prediction):
    sm = BidirectionalLieSmoother()
    params = sm._run_pass(ms, direction="forward", use_prediction=use_prediction)
    return params_to_hinv(params)


def run_variant_bidir(ms):
    sm = BidirectionalLieSmoother()
    for m in ms:
        if m is None:
            sm._measurements.append(None)
        else:
            sm._measurements.append(dict(m))
    sm.smooth_all()
    return [sm.get_H_inv(i) for i in range(len(ms))]


def run_variant_gauss(ms, sigma=3.0, max_rep_err=20.0):
    """Oro offline: outliers fuera, interpolacion lineal, gauss fase cero."""
    N = len(ms)

    def _sane(m):
        H = BidirectionalLieSmoother._build_H_inv_static(
            m["R"], m["fx"], m["fy"], m["cx"], m["cy"], m["pos"])
        if H is None:
            return False
        w = H @ IMAGE_POINTS
        p = (w[:2] / w[2:3]).T
        return bool((np.abs(p[:, 0]) <= 120).all()
                    and (np.abs(p[:, 1]) <= 80).all())

    ok = [i for i, m in enumerate(ms)
          if m is not None and m["rep_err"] <= max_rep_err and _sane(m)]
    if len(ok) < 5:
        return [None] * N
    ref = ms[ok[0]]["R"]
    rotvecs = np.full((N, 3), np.nan)
    scal = np.full((N, 7), np.nan)  # fx fy cx cy pos(3)
    for i in ok:
        m = ms[i]
        rotvecs[i] = (ref.inv() * m["R"]).as_rotvec()
        scal[i] = [m["fx"], m["fy"], m["cx"], m["cy"], *m["pos"]]
    idx = np.arange(N)
    for arr in (rotvecs, scal):
        for c in range(arr.shape[1]):
            col = arr[:, c]
            good = ~np.isnan(col)
            arr[:, c] = np.interp(idx, idx[good], col[good])
    rotvecs = gaussian_filter1d(rotvecs, sigma, axis=0, mode="nearest")
    scal = gaussian_filter1d(scal, sigma, axis=0, mode="nearest")
    params = []
    for i in range(N):
        params.append({
            "R": ref * Rotation.from_rotvec(rotvecs[i]),
            "fx": scal[i, 0], "fy": scal[i, 1],
            "cx": scal[i, 2], "cy": scal[i, 3], "pos": scal[i, 4:7]})
    return params_to_hinv(params)


def sane_mask(series, max_x=120.0, max_y=80.0):
    """Frames cuya proyeccion cae en un entorno razonable del campo."""
    ok = ~np.isnan(series[:, :, 0]).any(axis=1)
    with np.errstate(invalid="ignore"):
        inb = ((np.abs(series[:, :, 0]) <= max_x)
               & (np.abs(series[:, :, 1]) <= max_y)).all(axis=1)
    return ok & inb


def run_variant_gauss_robust(ms, sigma=3.0, max_rep_err=20.0,
                             med_win=11, med_reject_deg=4.0,
                             max_gap_fill=25):
    """
    Candidata a produccion: gauss fase-cero con rechazo ROBUSTO.
      1. rechazo por rep_err y por proyeccion insana
      2. rechazo por residuo vs mediana deslizante del rotvec (outliers
         espejados de PnLCalib con rep_err aceptable)
      3. interpolacion solo dentro de huecos <= max_gap_fill
      4. gauss fase-cero sobre rotvec/focal/pos
      5. huecos largos -> None (igual que MAX_GAP_FILL_FRAMES en prod)
    """
    from scipy.ndimage import median_filter
    N = len(ms)

    def _sane(m):
        H = BidirectionalLieSmoother._build_H_inv_static(
            m["R"], m["fx"], m["fy"], m["cx"], m["cy"], m["pos"])
        if H is None:
            return False
        w = H @ IMAGE_POINTS
        p = (w[:2] / w[2:3]).T
        return bool((np.abs(p[:, 0]) <= 120).all()
                    and (np.abs(p[:, 1]) <= 80).all())

    ok = [i for i, m in enumerate(ms)
          if m is not None and m["rep_err"] <= max_rep_err and _sane(m)]
    if len(ok) < 5:
        return [None] * N
    ref = ms[ok[0]]["R"]
    rotvecs = np.full((N, 3), np.nan)
    scal = np.full((N, 7), np.nan)
    for i in ok:
        m = ms[i]
        rotvecs[i] = (ref.inv() * m["R"]).as_rotvec()
        scal[i] = [m["fx"], m["fy"], m["cx"], m["cy"], *m["pos"]]

    # 2. mediana deslizante sobre las mediciones aceptadas (solo indices ok)
    rv_ok = rotvecs[ok]
    med = np.stack([median_filter(rv_ok[:, c], size=med_win, mode="nearest")
                    for c in range(3)], axis=1)
    resid_deg = np.rad2deg(np.linalg.norm(rv_ok - med, axis=1))
    keep = resid_deg <= med_reject_deg
    ok = [i for i, k in zip(ok, keep) if k]
    if len(ok) < 5:
        return [None] * N
    drop = ~np.isin(np.arange(N), ok)
    rotvecs[drop] = np.nan
    scal[drop] = np.nan

    # 3-4. interpolar + suavizar
    idx = np.arange(N)
    good = ~np.isnan(rotvecs[:, 0])
    for arr in (rotvecs, scal):
        for c in range(arr.shape[1]):
            col = arr[:, c]
            g = ~np.isnan(col)
            arr[:, c] = np.interp(idx, idx[g], col[g])
    rotvecs = gaussian_filter1d(rotvecs, sigma, axis=0, mode="nearest")
    scal = gaussian_filter1d(scal, sigma, axis=0, mode="nearest")

    # 5. invalidar huecos largos entre mediciones ACEPTADAS (regla de prod)
    params = [{
        "R": ref * Rotation.from_rotvec(rotvecs[i]),
        "fx": scal[i, 0], "fy": scal[i, 1],
        "cx": scal[i, 2], "cy": scal[i, 3], "pos": scal[i, 4:7]}
        for i in range(N)]
    i = 0
    while i < N:
        if not good[i]:
            j = i
            while j < N and not good[j]:
                j += 1
            if (j - i) > max_gap_fill:
                for k in range(i, j):
                    params[k] = None
            i = j
        else:
            i += 1
    return params_to_hinv(params)


def metrics(series, raw_series, raw_sane):
    """Jitter (mediana de |2a diferencia|) y desviacion vs crudo sano."""
    valid = sane_mask(series)
    acc = series[2:] - 2 * series[1:-1] + series[:-2]
    vmask = valid[2:] & valid[1:-1] & valid[:-2]
    if vmask.any():
        mag = np.sqrt(np.sum(acc[vmask] ** 2, axis=-1)).mean(axis=-1)
        jitter_med = float(np.median(mag))
        jitter_p95 = float(np.percentile(mag, 95))
    else:
        jitter_med = jitter_p95 = np.nan
    both = valid & raw_sane
    if both.any():
        d = np.sqrt(np.sum((series[both] - raw_series[both]) ** 2,
                           axis=-1)).mean(axis=-1)
        dev_med = float(np.median(d))
        dev_p95 = float(np.percentile(d, 95))
    else:
        dev_med = dev_p95 = np.nan
    return jitter_med, jitter_p95, dev_med, dev_p95, int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_calib", required=True)
    ap.add_argument("--gauss_sigma", type=float, default=3.0)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    dump = json.loads(Path(args.raw_calib).read_text())
    ms = measurements_from_dump(dump)
    n_ok = sum(1 for m in ms if m is not None)
    reps = [m["rep_err"] for m in ms if m is not None]
    print(f"frames: {len(ms)} | con calibracion: {n_ok} "
          f"| rep_err mediana {np.median(reps):.2f}px p90 {np.percentile(reps, 90):.2f}px")
    srcs = {}
    for m in ms:
        if m:
            srcs[m["source"]] = srcs.get(m["source"], 0) + 1
    print(f"fuentes: {srcs}")

    variants = {
        "raw": run_variant_raw(ms),
        "ema_fwd (viejo)": run_variant_pass(ms, use_prediction=False),
        "lie_fwd": run_variant_pass(ms, use_prediction=True),
        "lie_bidir (PROD)": run_variant_bidir(ms),
        f"gauss s={args.gauss_sigma:g} (oro)": run_variant_gauss(
            ms, sigma=args.gauss_sigma),
        "gauss ROBUSTO (cand)": run_variant_gauss_robust(
            ms, sigma=args.gauss_sigma),
    }
    raw_series = project_series(variants["raw"])
    raw_sane = sane_mask(raw_series)
    n_insane = int((~np.isnan(raw_series[:, 0, 0]) & ~raw_sane).sum())
    print(f"mediciones crudas INSANAS (proyectan fuera de +-120x80 m): "
          f"{n_insane} de {int((~np.isnan(raw_series[:, 0, 0])).sum())}")

    print(f"\n{'variante':<22} {'jit_med':>8} {'jit_p95':>8} "
          f"{'dev_med':>8} {'dev_p95':>8} {'frames':>7}   (m)")
    results = {}
    for name, h in variants.items():
        s = project_series(h)
        jm, jp, dm, dp, nv = metrics(s, raw_series, raw_sane)
        results[name] = {"jitter_med": jm, "jitter_p95": jp,
                         "dev_med": dm, "dev_p95": dp, "frames": nv}
        print(f"{name:<22} {jm:>8.4f} {jp:>8.3f} {dm:>8.3f} {dp:>8.2f} {nv:>7}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=1))
        print(f"\nGuardado: {args.output_json}")


if __name__ == "__main__":
    main()
