# src/tactical_vision_pipeline.py
"""
TacticalVision AI — pipeline unificado end-to-end por secuencia.

Fases (cada una es un modulo/CLI independiente; ver AGENTS.md):

  detect     YOLO26 (balon) + RF-DETR (jugadores/porteros/arbitros)
  cmc        compensacion de movimiento de camara
  track      grafo HMM + Viterbi (trayectoria del balon)
  eval       metricas de tracking vs GT (grafico)
  team       clustering de equipos (HSV role-aware)
  audit      auditoria del clustering vs gameinfo.ini      (--run_team_audit)
  jersey     identificacion de dorsales                    (--jersey_model)
  map2d      calibracion PnLCalib + suavizado SO(3) -> tracking_2d.csv
             + calibration_hinv.json (video solo con --render)
  analytics  metricas fisicas y de forma de equipo
  gt2d       GT proyectado a 2D + comparacion Pred vs GT   (--with_gt)
  sam2       segmentacion SAM2 + render                    (--render + --sam2_weights)
  scanning   scanning V2 (orientacion aprox. + head-turn antes de recepcion)
  scan_pred  prediccion del clasificador de scanning entrenado

OPTIMIZACION CLAVE — cache incremental: cada fase declara sus archivos de
salida; si ya existen se SALTA (usa `--force all` o `--force fase1,fase2`
para recomputar). Los renders de video estan APAGADOS por defecto
(`--render` los activa): el pipeline de datos corre sin costo de video.

Ejemplos:
  # una secuencia, solo datos (rapido; salta lo ya computado)
  python src/tactical_vision_pipeline.py \
      --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148

  # varias secuencias
  python src/tactical_vision_pipeline.py \
      --sequences SNMOT-116 SNMOT-117 SNMOT-148 \
      --data_root data/tracking/SoccerNet/tracking/test/test

  # con videos y dorsales
  python src/tactical_vision_pipeline.py --sequence_dir ... --render \
      --jersey_model runs/jersey_perframe_v3_224/best.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.scanning_v2.paths import scanning_dir  # noqa: E402

DEFAULT_DATA_ROOT = os.path.join("data", "tracking", "SoccerNet", "tracking",
                                 "test", "test")
DEFAULT_SCAN_MODEL = os.path.join("outputs", "scanning_training", "models",
                                  "best", "scanning_classifier.pkl")


@dataclass
class Phase:
    name: str
    desc: str
    cmd: List[str]
    outputs: List[str]              # si TODOS existen -> se salta
    enabled: bool = True
    skip_reason: str = ""           # cuando enabled=False


@dataclass
class PhaseResult:
    name: str
    status: str                     # ran | cached | disabled | failed
    seconds: float = 0.0
    detail: str = ""


def _exists(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return os.path.isdir(path)


def _done(phase: Phase) -> bool:
    return bool(phase.outputs) and all(_exists(p) for p in phase.outputs)


def run_phase(phase: Phase, force: bool) -> PhaseResult:
    if not phase.enabled:
        return PhaseResult(phase.name, "disabled", detail=phase.skip_reason)
    if not force and _done(phase):
        return PhaseResult(phase.name, "cached",
                           detail=os.path.basename(phase.outputs[0]))
    print(f"\n{'=' * 60}\n[{phase.name}] {phase.desc}\n"
          f"  $ {' '.join(phase.cmd)}\n{'=' * 60}")
    t0 = time.time()
    subprocess.run(phase.cmd, check=True)
    return PhaseResult(phase.name, "ran", seconds=time.time() - t0)


# ---------------------------------------------------------------------------
def build_phases(args, sequence_dir: str) -> List[Phase]:
    py = sys.executable
    seq = os.path.basename(os.path.normpath(sequence_dir))
    out_dir = args.output_dir or os.path.join("outputs", seq)
    os.makedirs(out_dir, exist_ok=True)

    def out(*parts):
        return os.path.join(out_dir, *parts)

    def mod(*parts):
        return os.path.join(BASE_DIR, *parts)

    det = out(f"{seq}_detections.json")
    cmc = out(f"{seq}_cmc.json")
    traj = out(f"{seq}_trajectory.json")
    team = out(f"{seq}_team_assignments.json")
    team_audit = out(f"{seq}_team_audit.json")
    jersey = out(f"{seq}_jersey_identity.json")
    track2d = out(f"{seq}_tracking_2d.csv")
    calib = out("calibration_hinv.json")
    scan_dir = str(scanning_dir("outputs", seq)) if args.output_dir is None \
        else os.path.join(out_dir, "scanning")

    phases: List[Phase] = []

    phases.append(Phase(
        "detect", "Deteccion YOLO26 + RF-DETR",
        [py, mod("core", "detection", "tactical_vision_extractor.py"),
         "--sequence_dir", sequence_dir, "--yolo_weights", args.yolo_weights,
         "--rfdetr_weights", args.rfdetr_weights, "--output_json", det],
        [det]))

    phases.append(Phase(
        "cmc", "Compensacion de movimiento de camara",
        [py, mod("core", "tracking", "tactical_vision_cmc.py"),
         "--sequence_dir", sequence_dir, "--output_json", cmc, "--fast"],
        [cmc]))

    phases.append(Phase(
        "track", "HMM + Viterbi (trayectoria del balon)",
        [py, mod("core", "tracking", "tactical_vision_hmm.py"),
         "--detections_json", det, "--cmc_json", cmc, "--output_json", traj],
        [traj]))

    plot = args.output_plot or out(f"{seq}_tracking_metrics.png")
    phases.append(Phase(
        "eval", "Metricas de tracking vs GT",
        [py, mod("core", "tracking", "tactical_vision_eval.py"),
         "--sequence_dir", sequence_dir, "--trajectory_json", traj,
         "--output_plot", plot],
        [plot]))

    phases.append(Phase(
        "team", "Clustering de equipos",
        [py, mod("core", "clustering", "team_clustering_phase.py"),
         "--sequence_dir", sequence_dir, "--rfdetr_weights", args.rfdetr_weights,
         "--detections_json", det, "--output_json", team,
         "--output_video", out(f"{seq}_team_clustering.mp4")],
        [team]))

    gi = os.path.join(sequence_dir, "gameinfo.ini")
    phases.append(Phase(
        "audit", "Auditoria del clustering",
        [py, mod("scripts", "audit_team_clustering.py"),
         "--sequence_dir", sequence_dir, "--team_json", team,
         "--detections_json", det, "--output_json", team_audit],
        [team_audit],
        enabled=bool(args.run_team_audit and os.path.exists(gi)),
        skip_reason="usa --run_team_audit (requiere gameinfo.ini)"))

    jersey_cmd = [py, mod("core", "identity", "jersey_identity_phase.py"),
                  "--sequence_dir", sequence_dir, "--detections_json", det,
                  "--team_assignments_json", team,
                  "--model_path", args.jersey_model or "",
                  "--output_json", jersey, "--device", args.device,
                  "--p1_threshold", str(args.p1_threshold),
                  "--margin_threshold", str(args.margin_threshold),
                  "--seed", str(args.seed),
                  "--inference_mode", args.inference_mode,
                  "--min_legible_frames", str(args.min_legible_frames),
                  "--min_peak_quality", str(args.min_peak_quality),
                  "--fusion_mode", args.fusion_mode,
                  "--temperature", str(args.temperature)]
    if args.roster_json:
        jersey_cmd += ["--roster_json", args.roster_json]
    if args.parseq_model and not args.no_parseq:
        jersey_cmd += ["--parseq_model", args.parseq_model,
                       "--parseq_weight", str(args.parseq_weight)]
        if args.parseq_checkpoint:
            jersey_cmd += ["--parseq_checkpoint", args.parseq_checkpoint]
    if args.legibility_model:
        jersey_cmd += ["--legibility_model", args.legibility_model,
                       "--legibility_threshold", str(args.legibility_threshold)]
    for flag in ("multi_crop", "link_fragments", "split_on_switch",
                 "reassign_conflicts", "infer_unknowns"):
        if getattr(args, flag):
            jersey_cmd.append(f"--{flag}")
    tm = args.team_mapping or (team_audit if args.run_team_audit else None)
    if tm:
        jersey_cmd += ["--team_mapping", tm]
    phases.append(Phase(
        "jersey", "Identificacion de dorsales",
        jersey_cmd, [jersey],
        enabled=bool(args.jersey_model),
        skip_reason="usa --jersey_model"))

    map_cmd = [py, mod("core", "mapping", "tactical_vision_2d_mapper.py"),
               "--sequence_dir", sequence_dir, "--detections", det,
               "--trajectory", traj, "--team_assignments", team,
               "--pnlcalib_kp_weights", args.pnlcalib_kp_weights,
               "--pnlcalib_line_weights", args.pnlcalib_line_weights,
               "--output_csv", track2d, "--output_calibration", calib]
    map_outs = [track2d, calib]
    if args.render:
        map_cmd += ["--output", out(f"{seq}_2d_map.mp4")]
        map_outs.append(out(f"{seq}_2d_map.mp4"))
        if args.jersey_model:
            map_cmd += ["--jersey_json", jersey]
    else:
        map_cmd.append("--no_video")
    phases.append(Phase(
        "map2d", "Mapeo 2D metrico (PnLCalib + suavizado SO(3))",
        map_cmd, map_outs))

    phases.append(Phase(
        "analytics", "Metricas fisicas y de forma",
        [py, mod("core", "analytics", "compute_metrics.py"),
         "--input_csv", track2d, "--output_dir", out_dir,
         "--fps", str(args.fps)],
        [out(f"{seq}_player_physical_metrics.csv"),
         out(f"{seq}_team_shape_metrics.csv")]))

    gt2d = out(f"{seq}_gt_tracking_2d.csv")
    phases.append(Phase(
        "gt2d", "GT proyectado a 2D + comparacion Pred vs GT",
        [py, mod("core", "analytics", "project_gt_to_2d.py"),
         "--sequence_dir", sequence_dir, "--calibration_json", calib,
         "--output_csv", gt2d, "--fps", str(args.fps)],
        [gt2d],
        enabled=bool(args.with_gt and os.path.exists(
            os.path.join(sequence_dir, "gt", "gt.txt"))),
        skip_reason="usa --with_gt (requiere gt/gt.txt)"))
    phases.append(Phase(
        "gt_compare", "Comparacion Pred vs GT (Hungarian)",
        [py, mod("core", "analytics", "compare_pred_gt.py"),
         "--pred_csv", track2d, "--gt_csv", gt2d,
         "--output_dir", out("comparison"), "--max_dist_m", "3.0"],
        [out("comparison", "comparison_summary.json")],
        enabled=bool(args.with_gt and os.path.exists(
            os.path.join(sequence_dir, "gt", "gt.txt"))),
        skip_reason="usa --with_gt"))

    phases.append(Phase(
        "dashboard", "Dashboard HTML interactivo",
        [py, mod("dashboard", "generate_dashboard.py"),
         "--sequence_name", seq,
         "--output_html", out("dashboard", "index.html")],
        [out("dashboard", "index.html")]))

    phases.append(Phase(
        "sam2", "Segmentacion SAM2 + render",
        [py, mod("core", "segmentation", "tactical_vision_sam2.py"),
         "--sequence_dir", sequence_dir, "--detections_json", det,
         "--trajectory_json", traj, "--sam2_weights", args.sam2_weights or "",
         "--output_dir", out_dir],
        [out(f"{seq}_tacticalvision_refined.mp4")],
        enabled=bool(args.render and args.sam2_weights),
        skip_reason="usa --render y --sam2_weights (solo visualizacion)"))

    scan_cmd = [py, mod("scripts", "scanning_v2", "run_scanning_v2.py"),
                "--config", args.scanning_config, "--video_id", seq,
                "--detections", det, "--trajectory", traj,
                "--team_assignments", team, "--calibration", calib,
                "--sequence_dir", sequence_dir, "--output_dir", scan_dir]
    if args.render:
        scan_cmd.append("--render")
    phases.append(Phase(
        "scanning", "Scanning V2 (orientacion aprox. + head-turn)",
        scan_cmd,
        [os.path.join(scan_dir, "scanning_events.parquet")],
        enabled=not args.no_scanning,
        skip_reason="--no_scanning"))

    final_cmd = [py, mod("scripts", "compose_final_video.py"), "--video_id", seq,
                 "--outputs_root", os.path.dirname(out_dir) or "outputs",
                 "--overwrite", "--archive_debug"]
    if args.tidy_outputs:
        final_cmd.append("--tidy")
    phases.append(Phase(
        "final_video", "VIDEO FINAL (todos los modelos) en la carpeta de la secuencia",
        final_cmd,
        [out(f"{seq}_FINAL.mp4")],
        enabled=bool(args.render),
        skip_reason="usa --render (necesita el video del mapper)"))

    phases.append(Phase(
        "scan_pred", "Prediccion del clasificador de scanning",
        [py, mod("scripts", "scanning_v2", "predict_scanning_for_video.py"),
         "--config", args.scanning_train_config, "--video_id", seq,
         "--outputs_root", os.path.dirname(out_dir) or "outputs",
         "--model", args.scan_model],
        [os.path.join(scan_dir, "model_predictions.parquet")],
        enabled=bool(not args.no_scanning and os.path.exists(args.scan_model)),
        skip_reason=f"modelo no encontrado: {args.scan_model}"))

    return phases


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="TacticalVision AI: pipeline end-to-end con cache incremental")
    p.add_argument("--sequence_dir", type=str, default=None,
                   help="directorio de UNA secuencia (con img1/)")
    p.add_argument("--sequences", nargs="+", default=None,
                   help="varias secuencias por nombre (usa --data_root)")
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output_dir", type=str, default=None,
                   help="default: outputs/<secuencia>")
    p.add_argument("--fps", type=float, default=25.0)

    # seleccion / cache de fases
    p.add_argument("--only", nargs="+", default=None,
                   help="ejecutar solo estas fases (p.ej. --only scanning scan_pred)")
    p.add_argument("--skip", nargs="+", default=[],
                   help="fases a omitir")
    p.add_argument("--force", nargs="+", default=[],
                   help="'all' o nombres de fase a recomputar aunque exista output")
    p.add_argument("--render", action="store_true",
                   help="genera videos (mapper, clips de scanning, SAM2). "
                        "Apagado por defecto: el pipeline de datos no lo necesita")
    p.add_argument("--no_scanning", action="store_true")
    p.add_argument("--tidy_outputs", action="store_true",
                   help="al final BORRA archivos de ruido de la secuencia "
                        "(crops, mascaras, debug); quedan los mp4 importantes "
                        "y los datos del cache")

    # pesos / modelos
    p.add_argument("--yolo_weights", type=str, default="models/yolo26.pt")
    p.add_argument("--rfdetr_weights", type=str,
                   default="models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth")
    p.add_argument("--sam2_weights", type=str, default=None)
    p.add_argument("--pnlcalib_kp_weights", type=str, default="models/SV_kp")
    p.add_argument("--pnlcalib_line_weights", type=str, default="models/SV_lines")
    p.add_argument("--scanning_config", type=str, default="configs/scanning_v2.yaml")
    p.add_argument("--scanning_train_config", type=str,
                   default="configs/scanning_v2_supervised_weak.yaml")
    p.add_argument("--scan_model", type=str, default=DEFAULT_SCAN_MODEL)
    p.add_argument("--output_plot", type=str, default="")

    # GT / auditoria
    p.add_argument("--with_gt", action="store_true",
                   help="proyecta GT a 2D y compara Pred vs GT")
    p.add_argument("--run_team_audit", action="store_true")

    # dorsales (Phase jersey) — igual que antes
    p.add_argument("--jersey_model", type=str, default=None)
    p.add_argument("--roster_json", type=str,
                   default="datasets/jersey_tracking_v1/rosters.json",
                   help="dorsales validos por equipo (postproceso); si la "
                        "secuencia no esta en el archivo, no restringe")
    p.add_argument("--team_mapping", type=str, default=None)
    # v2.0: confidence_topk (los frames donde el numero se ve claro mandan)
    # + ROSTER. Test 49 seqs: 270 locks @ 88.9% (p1 .95 -> 213 @ 93.0%)
    p.add_argument("--p1_threshold", type=float, default=0.90)
    p.add_argument("--margin_threshold", type=float, default=0.30)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inference_mode", type=str, default="temporal",
                   choices=["mil", "temporal", "hybrid"])
    p.add_argument("--legibility_model", type=str, default=None)
    p.add_argument("--legibility_threshold", type=float, default=0.5)
    p.add_argument("--multi_crop", action="store_true")
    p.add_argument("--no_parseq", action="store_true",
                   help="desactiva el lector PARSeq en la fase de dorsales (mas rapido)")
    p.add_argument("--parseq_model", type=str, default="parseq",
                   help="segundo lector PARSeq en el ensamble ('' lo desactiva)")
    p.add_argument("--parseq_weight", type=float, default=0.25)
    p.add_argument("--parseq_checkpoint", type=str,
                   default="runs/parseq_jersey_ft/best.pt",
                   help="pesos PARSeq fine-tuneados en dorsales (v2.2; si el "
                        "archivo no existe la fase usa los pesos genericos)")
    p.add_argument("--infer_unknowns", action="store_true",
                   help="eliminacion con roster para tracklets sin lock "
                        "(recomendado solo en partido completo; en clips "
                        "cortos las restricciones son debiles)")
    p.add_argument("--min_legible_frames", type=int, default=4)
    p.add_argument("--min_peak_quality", type=float, default=0.3)
    p.add_argument("--fusion_mode", type=str, default="confidence_topk",
                   choices=["geometric", "arithmetic", "topk_geometric", "confidence_topk"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--link_fragments", action="store_true")
    p.add_argument("--split_on_switch", action="store_true")
    p.add_argument("--reassign_conflicts", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    if not args.sequence_dir and not args.sequences:
        sys.exit("Indica --sequence_dir o --sequences (ver --help).")
    seq_dirs = ([args.sequence_dir] if args.sequence_dir else []) + [
        os.path.join(args.data_root, s) for s in (args.sequences or [])]
    for d in seq_dirs:
        if not os.path.isdir(d):
            sys.exit(f"Secuencia no encontrada: {d}")

    force_all = "all" in args.force
    grand: List[PhaseResult] = []
    t0 = time.time()
    for sdir in seq_dirs:
        seq = os.path.basename(os.path.normpath(sdir))
        print(f"\n############ {seq} ############")
        results: List[PhaseResult] = []
        for phase in build_phases(args, sdir):
            if args.only and phase.name not in args.only:
                continue
            if phase.name in args.skip:
                results.append(PhaseResult(phase.name, "disabled", detail="--skip"))
                continue
            try:
                r = run_phase(phase, force_all or phase.name in args.force)
            except subprocess.CalledProcessError as e:
                results.append(PhaseResult(phase.name, "failed",
                                           detail=f"exit {e.returncode}"))
                print(f"[{seq}] FASE '{phase.name}' FALLO (exit {e.returncode}); "
                      f"se detiene esta secuencia.")
                break
            results.append(r)
        grand += results

        print(f"\n--- resumen {seq} ---")
        for r in results:
            mark = {"ran": "OK ", "cached": "CACHE", "disabled": "OFF ",
                    "failed": "FAIL"}[r.status]
            extra = f" ({r.seconds:.1f}s)" if r.status == "ran" else \
                    (f" [{r.detail}]" if r.detail else "")
            print(f"  {mark:6s} {r.name:10s}{extra}")

    n_ran = sum(r.status == "ran" for r in grand)
    n_cache = sum(r.status == "cached" for r in grand)
    n_fail = sum(r.status == "failed" for r in grand)
    print(f"\n=== TOTAL {time.time() - t0:.1f}s | fases ejecutadas {n_ran}, "
          f"en cache {n_cache}, fallidas {n_fail} ===")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
