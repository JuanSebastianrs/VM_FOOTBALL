"""
Official TrackEval-based Tracking Evaluation.

Converts pipeline outputs and SoccerNet GT to MOTChallenge format,
then runs TrackEval for official HOTA, DetA, AssA, CLEAR (MOTA/MOTP),
and Identity (IDF1) metrics.

Usage:
    python src/evaluate_tracking_official.py
"""

import os
import sys
import glob
import json
import shutil
import numpy as np

# ── NumPy 2.x compatibility shim for TrackEval ──────────────────────────────
# TrackEval uses deprecated np.float, np.int, np.bool removed in NumPy 2.0
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'bool'):
    np.bool = np.bool_

# ── Configuration ────────────────────────────────────────────────────────────
GT_DIR      = r"data\tracking\SoccerNet\tracking\test\test"
PREDS_DIR   = r"outputs"
OUTPUT_DIR  = r"results_final\evaluation"
WORK_DIR    = r"results_final\evaluation\trackeval_workspace"
IMG_W, IMG_H = 1920, 1080


def convert_gt_to_mot(gt_dir, work_dir):
    """Convert YOLO-format GT labels to MOTChallenge gt.txt format."""
    seqs = sorted(glob.glob(os.path.join(gt_dir, "SNMOT-*")))
    seq_names = []
    
    for seq_dir in seqs:
        seq_name = os.path.basename(seq_dir)
        lbl_dir = os.path.join(seq_dir, "labels")
        if not os.path.exists(lbl_dir):
            continue
            
        # Create MOTChallenge GT directory structure
        gt_out_dir = os.path.join(work_dir, "gt", seq_name, "gt")
        os.makedirs(gt_out_dir, exist_ok=True)
        
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        lines = []
        
        for lbl_file in lbl_files:
            frame_id = int(os.path.splitext(os.path.basename(lbl_file))[0])
            
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    if cls_id == 5:  # Skip ball
                        continue
                    
                    x_n, y_n, w_n, h_n = map(float, parts[1:5])
                    track_id = int(parts[5])
                    
                    # Convert YOLO normalized → pixel (top-left, w, h)
                    w_px = w_n * IMG_W
                    h_px = h_n * IMG_H
                    x_tl = (x_n * IMG_W) - w_px / 2
                    y_tl = (y_n * IMG_H) - h_px / 2
                    
                    # MOTChallenge format: frame,id,x,y,w,h,conf,class,visibility
                    # class=1 for pedestrian, visibility=1.0
                    lines.append(f"{frame_id},{track_id},{x_tl:.2f},{y_tl:.2f},{w_px:.2f},{h_px:.2f},1,1,1.0\n")
        
        gt_file = os.path.join(gt_out_dir, "gt.txt")
        with open(gt_file, 'w') as f:
            f.writelines(lines)
        
        # Create seqinfo.ini
        num_frames = len(lbl_files)
        seqinfo = (
            f"[Sequence]\n"
            f"name={seq_name}\n"
            f"imDir=img1\n"
            f"frameRate=25\n"
            f"seqLength={num_frames}\n"
            f"imWidth={IMG_W}\n"
            f"imHeight={IMG_H}\n"
            f"imExt=.jpg\n"
        )
        seqinfo_path = os.path.join(work_dir, "gt", seq_name, "seqinfo.ini")
        with open(seqinfo_path, 'w') as f:
            f.write(seqinfo)
        
        seq_names.append(seq_name)
        print(f"  [GT] {seq_name}: {len(lines)} annotations, {num_frames} frames")
    
    # Create seqmap file
    seqmap_dir = os.path.join(work_dir, "gt", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    seqmap_file = os.path.join(seqmap_dir, "SoccerNet-test.txt")
    with open(seqmap_file, 'w') as f:
        f.write("name\n")
        for s in seq_names:
            f.write(f"{s}\n")
    
    return seq_names


def convert_preds_to_mot(preds_dir, work_dir, seq_names):
    """Convert pipeline detections JSON to MOTChallenge tracker format."""
    tracker_name = "TacticalVision"
    
    tracker_dir = os.path.join(work_dir, "trackers", tracker_name)
    os.makedirs(tracker_dir, exist_ok=True)
    
    for seq_name in seq_names:
        pred_json = os.path.join(preds_dir, seq_name, f"{seq_name}_detections.json")
        if not os.path.exists(pred_json):
            print(f"  [PRED] {seq_name}: SKIPPED (no detections.json)")
            continue
        
        with open(pred_json, 'r') as f:
            preds = json.load(f)
        
        lines = []
        for p_data in preds:
            frame_id = p_data['frame_id']
            # Merge all tracked objects
            for role in ['players', 'referees', 'goalkeepers']:
                for obj in p_data.get(role, []):
                    tid = obj['track_id']
                    w = obj['x_max'] - obj['x_min']
                    h = obj['y_max'] - obj['y_min']
                    x = obj['x_min']
                    y = obj['y_min']
                    lines.append(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        
        # TrackEval expects: trackers/<name>/data/<seq_name>.txt (flat, no subdirs)
        out_file = os.path.join(tracker_dir, f"{seq_name}.txt")
        with open(out_file, 'w') as f:
            f.writelines(lines)
        
        print(f"  [PRED] {seq_name}: {len(lines)} predictions")
    
    return tracker_name


def run_trackeval(work_dir, tracker_name, seq_names):
    """Run TrackEval with HOTA, CLEAR, and Identity metrics."""
    import trackeval
    
    # Build evaluator config
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': os.path.join(OUTPUT_DIR, 'trackeval_error_log.txt'),
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': True,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': True,
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
        'TRACKERS_TO_EVAL': [tracker_name],
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
    
    print("\n" + "=" * 60)
    print("  Running TrackEval (Official HOTA, CLEAR, Identity)")
    print("=" * 60 + "\n")
    
    raw_results, messages = evaluator.evaluate([dataset], metrics_list)
    
    return raw_results


def extract_and_save_results(raw_results, tracker_name):
    """Extract key metrics from TrackEval raw results and save to JSON."""
    
    # Navigate the nested result dict
    # Structure: raw_results[dataset_name][tracker_name][seq_or_COMBINED][class_name][metric_name]
    dataset_key = list(raw_results.keys())[0]
    tracker_results = raw_results[dataset_key][tracker_name]
    
    # Get COMBINED results
    combined = tracker_results.get('COMBINED_SEQ', {})
    ped = combined.get('pedestrian', {})
    
    # HOTA metrics (averaged across IoU thresholds 0.05 to 0.95)
    hota_data = ped.get('HOTA', {})
    clear_data = ped.get('CLEAR', {})
    identity_data = ped.get('Identity', {})
    
    # HOTA is computed at multiple IoU thresholds; the "official" HOTA is the mean
    hota_array = hota_data.get('HOTA', np.array([0]))
    deta_array = hota_data.get('DetA', np.array([0]))
    assa_array = hota_data.get('AssA', np.array([0]))
    
    results = {
        # HOTA family (mean across IoU thresholds)
        "HOTA": round(float(np.mean(hota_array)) * 100, 2),
        "DetA": round(float(np.mean(deta_array)) * 100, 2),
        "AssA": round(float(np.mean(assa_array)) * 100, 2),
        # HOTA at specific IoU thresholds
        "HOTA@50": round(float(hota_array[0]) * 100, 2) if len(hota_array) > 0 else 0,
        "DetA@50": round(float(deta_array[0]) * 100, 2) if len(deta_array) > 0 else 0,
        "AssA@50": round(float(assa_array[0]) * 100, 2) if len(assa_array) > 0 else 0,
        # CLEAR metrics
        "MOTA": round(float(clear_data.get('MOTA', 0)) * 100, 2),
        "MOTP": round(float(clear_data.get('MOTP', 0)) * 100, 2),
        "CLR_FP": int(clear_data.get('CLR_FP', 0)),
        "CLR_FN": int(clear_data.get('CLR_FN', 0)),
        "IDSW": int(clear_data.get('IDSW', 0)),
        "CLR_TP": int(clear_data.get('CLR_TP', 0)),
        # Identity metrics
        "IDF1": round(float(identity_data.get('IDF1', 0)) * 100, 2),
        "IDP": round(float(identity_data.get('IDP', 0)) * 100, 2),
        "IDR": round(float(identity_data.get('IDR', 0)) * 100, 2),
    }
    
    # Also extract per-sequence results for detailed analysis
    per_seq = {}
    for seq_name, seq_data in tracker_results.items():
        if seq_name == 'COMBINED_SEQ':
            continue
        ped_seq = seq_data.get('pedestrian', {})
        hota_seq = ped_seq.get('HOTA', {})
        clear_seq = ped_seq.get('CLEAR', {})
        identity_seq = ped_seq.get('Identity', {})
        
        hota_arr_seq = hota_seq.get('HOTA', np.array([0]))
        deta_arr_seq = hota_seq.get('DetA', np.array([0]))
        assa_arr_seq = hota_seq.get('AssA', np.array([0]))
        
        per_seq[seq_name] = {
            "HOTA": round(float(np.mean(hota_arr_seq)) * 100, 2),
            "DetA": round(float(np.mean(deta_arr_seq)) * 100, 2),
            "AssA": round(float(np.mean(assa_arr_seq)) * 100, 2),
            "MOTA": round(float(clear_seq.get('MOTA', 0)) * 100, 2),
            "IDF1": round(float(identity_seq.get('IDF1', 0)) * 100, 2),
            "IDSW": int(clear_seq.get('IDSW', 0)),
        }
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save global results
    json_path = os.path.join(OUTPUT_DIR, "tracking_global_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Global metrics saved to {json_path}")
    
    # Save per-sequence results  
    per_seq_path = os.path.join(OUTPUT_DIR, "tracking_per_sequence_metrics.json")
    with open(per_seq_path, 'w') as f:
        json.dump(per_seq, f, indent=2)
    print(f"[INFO] Per-sequence metrics saved to {per_seq_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  OFFICIAL TRACKEVAL RESULTS")
    print("=" * 60)
    print(f"  HOTA  (mean)     : {results['HOTA']:.2f}%")
    print(f"  DetA  (mean)     : {results['DetA']:.2f}%")
    print(f"  AssA  (mean)     : {results['AssA']:.2f}%")
    print(f"  HOTA@50          : {results['HOTA@50']:.2f}%")
    print(f"  DetA@50          : {results['DetA@50']:.2f}%")
    print(f"  AssA@50          : {results['AssA@50']:.2f}%")
    print(f"  MOTA             : {results['MOTA']:.2f}%")
    print(f"  IDF1             : {results['IDF1']:.2f}%")
    print(f"  ID Switches      : {results['IDSW']}")
    print("=" * 60)
    
    return results, per_seq


def main():
    print("=" * 60)
    print("  OFFICIAL TRACKING EVALUATION (TrackEval)")
    print("  Metrics: HOTA, CLEAR (MOTA), Identity (IDF1)")
    print("=" * 60)
    
    # Clean workspace
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)
    
    # Step 1: Convert GT
    print("\n[Step 1] Converting GT labels to MOTChallenge format...")
    seq_names = convert_gt_to_mot(GT_DIR, WORK_DIR)
    print(f"  Total: {len(seq_names)} sequences\n")
    
    # Step 2: Convert Predictions
    print("[Step 2] Converting pipeline predictions to MOTChallenge format...")
    tracker_name = convert_preds_to_mot(PREDS_DIR, WORK_DIR, seq_names)
    
    # Step 3: Run TrackEval
    raw_results = run_trackeval(WORK_DIR, tracker_name, seq_names)
    
    # Step 4: Extract and save
    results, per_seq = extract_and_save_results(raw_results, tracker_name)
    
    return results, per_seq


if __name__ == "__main__":
    main()
