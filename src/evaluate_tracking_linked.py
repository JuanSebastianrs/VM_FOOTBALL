"""
TrackEval comparison: online tracker vs. offline tracklet linking.

Builds the same MOTChallenge workspace as evaluate_tracking_official.py,
adds a second tracker ("TacticalVision_Linked") produced by
core.tracking.offline_tracklet_linker, evaluates both with TrackEval and
prints a side-by-side comparison.

Usage:
    $env:PYTHONPATH="."
    python src/evaluate_tracking_linked.py
"""

import os
import sys
import json
import shutil
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_tracking_official import (  # noqa: E402
    GT_DIR, PREDS_DIR, OUTPUT_DIR, WORK_DIR,
    convert_gt_to_mot, convert_preds_to_mot,
)
import core.tracking.offline_tracklet_linker as linker  # noqa: E402
from core.tracking.offline_tracklet_linker import postprocess_sequence  # noqa: E402

LINKED_TRACKER = "TacticalVision_Linked"
BASELINE_METRICS_JSON = os.path.join(OUTPUT_DIR, "tracking_global_metrics.json")


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def convert_linked_preds(preds_dir, work_dir, seq_names, use_app=True, use_jersey=True):
    """Post-process each sequence's tracklets and write linked tracker files."""
    tracker_dir = os.path.join(work_dir, "trackers", LINKED_TRACKER)
    os.makedirs(tracker_dir, exist_ok=True)

    for seq_name in seq_names:
        seq_dir = os.path.join(preds_dir, seq_name)
        preds = _load_json(os.path.join(seq_dir, f"{seq_name}_detections.json"))
        if preds is None:
            print(f"  [LINK] {seq_name}: SKIPPED (no detections.json)")
            continue
        cmc = _load_json(os.path.join(seq_dir, f"{seq_name}_cmc.json"))
        teams = _load_json(os.path.join(seq_dir, f"{seq_name}_team_assignments.json"))

        emb_path = os.path.join(seq_dir, f"{seq_name}_tracklet_embeddings.npz")
        emb_path = emb_path if use_app and os.path.exists(emb_path) else None
        jersey = None
        if use_jersey:
            jersey = _load_json(os.path.join(seq_dir, f"{seq_name}_jersey_identity_v2_all.json"))

        rows = postprocess_sequence(preds, cmc=cmc, team_assignments=teams,
                                    embeddings_npz=emb_path, jersey_data=jersey)
        out_file = os.path.join(tracker_dir, f"{seq_name}.txt")
        with open(out_file, "w") as f:
            for frame, tid, x, y, w, h in rows:
                f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        print(f"  [LINK] {seq_name}: {len(rows)} predictions "
              f"(cmc={'y' if cmc else 'n'} app={'y' if emb_path else 'n'} "
              f"jersey={'y' if jersey else 'n'})")


def run_trackeval(work_dir, tracker_names):
    import trackeval

    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': os.path.join(OUTPUT_DIR, 'trackeval_error_log.txt'),
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': True,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    }
    dataset_config = {
        'GT_FOLDER': os.path.join(work_dir, 'gt'),
        'TRACKERS_FOLDER': os.path.join(work_dir, 'trackers'),
        'OUTPUT_FOLDER': os.path.join(work_dir, 'output'),
        'TRACKERS_TO_EVAL': tracker_names,
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'SoccerNet',
        'SPLIT_TO_EVAL': 'test',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': True,
        'TRACKER_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FILE': os.path.join(work_dir, 'gt', 'seqmaps', 'SoccerNet-test.txt'),
        'SKIP_SPLIT_FOL': True,
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]
    raw_results, _ = evaluator.evaluate([dataset], metrics_list)
    return raw_results


def extract_metrics(raw_results, tracker_name):
    dataset_key = list(raw_results.keys())[0]
    ped = raw_results[dataset_key][tracker_name]['COMBINED_SEQ']['pedestrian']
    hota, clear, ident = ped['HOTA'], ped['CLEAR'], ped['Identity']
    return {
        "HOTA": round(float(np.mean(hota['HOTA'])) * 100, 2),
        "DetA": round(float(np.mean(hota['DetA'])) * 100, 2),
        "AssA": round(float(np.mean(hota['AssA'])) * 100, 2),
        "MOTA": round(float(clear['MOTA']) * 100, 2),
        "IDF1": round(float(ident['IDF1']) * 100, 2),
        "IDSW": int(clear['IDSW']),
        "CLR_FP": int(clear['CLR_FP']),
        "CLR_FN": int(clear['CLR_FN']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link-gap", type=int, default=linker.MAX_LINK_GAP)
    parser.add_argument("--interp-gap", type=int, default=linker.MAX_INTERP_GAP)
    parser.add_argument("--max-dist", type=float, default=linker.MAX_DIST_THRESH)
    parser.add_argument("--only-linked", action="store_true",
                        help="Skip baseline evaluation (reuse saved metrics)")
    parser.add_argument("--no-app", action="store_true",
                        help="Ignore appearance embeddings")
    parser.add_argument("--no-jersey", action="store_true",
                        help="Ignore jersey identities")
    parser.add_argument("--app-max", type=float, default=linker.APP_MAX)
    parser.add_argument("--app-strong", type=float, default=linker.APP_STRONG)
    args = parser.parse_args()

    linker.MAX_LINK_GAP = args.link_gap
    linker.MAX_INTERP_GAP = args.interp_gap
    linker.MAX_DIST_THRESH = args.max_dist
    linker.APP_MAX = args.app_max
    linker.APP_STRONG = args.app_strong
    print(f"[Config] link_gap={args.link_gap} interp_gap={args.interp_gap} "
          f"max_dist={args.max_dist} app_max={args.app_max} "
          f"app_strong={args.app_strong} app={not args.no_app} "
          f"jersey={not args.no_jersey}")

    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    print("[Step 1] Converting GT...")
    seq_names = convert_gt_to_mot(GT_DIR, WORK_DIR)
    print(f"  Total: {len(seq_names)} sequences")

    baseline = "TacticalVision"
    if not args.only_linked:
        print("[Step 2] Converting baseline predictions...")
        baseline = convert_preds_to_mot(PREDS_DIR, WORK_DIR, seq_names)

    print("[Step 3] Offline tracklet linking + interpolation...")
    convert_linked_preds(PREDS_DIR, WORK_DIR, seq_names,
                         use_app=not args.no_app, use_jersey=not args.no_jersey)

    trackers = [LINKED_TRACKER] if args.only_linked else [baseline, LINKED_TRACKER]
    print(f"[Step 4] Running TrackEval on: {trackers}")
    raw = run_trackeval(WORK_DIR, trackers)

    results = {name: extract_metrics(raw, name) for name in trackers}
    if args.only_linked:
        saved = _load_json(BASELINE_METRICS_JSON) or {}
        results[baseline] = {k: saved.get(k, float("nan"))
                             for k in ["HOTA", "DetA", "AssA", "MOTA", "IDF1",
                                       "IDSW", "CLR_FP", "CLR_FN"]}

    print("\n" + "=" * 72)
    print(f"  {'Metric':<8}{baseline:>22}{LINKED_TRACKER:>28}{'Delta':>12}")
    print("=" * 72)
    for metric in ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN"]:
        b, l = results[baseline][metric], results[LINKED_TRACKER][metric]
        print(f"  {metric:<8}{b:>22}{l:>28}{l - b:>+12.2f}")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "tracking_linked_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Comparison saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
