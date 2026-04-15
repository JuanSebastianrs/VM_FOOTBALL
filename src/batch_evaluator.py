"""
Batch Evaluator: TacticalVision AI Pipeline
Runs phases P1-P8 (no video) across all test sequences and generates
a master CSV with Ball Tracking, MOT, and Team Clustering metrics.

Usage:
    python src/batch_evaluator.py                          # All 49 sequences
    python src/batch_evaluator.py --sequences SNMOT-197    # Single
    python src/batch_evaluator.py --skip-existing          # Reuse existing JSONs
"""

import os
import argparse
import subprocess
import json
import glob
import csv
import math
import cv2
from pathlib import Path

try:
    import motmetrics as mm
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    HAS_MOT = True
except ImportError:
    HAS_MOT = False
    import numpy as np
    print("Warning: motmetrics/scipy not installed. MOT metrics will be skipped. Run: pip install motmetrics scipy")

# ── GT Taxonomy (SoccerNet 6-class) ──────────────────────────────────────────
# The GT labels use the original SoccerNet 6-class taxonomy:
#   0 = player_left      (team:left)
#   1 = player_right     (team:right)
#   2 = goalkeeper_left  (team:left)
#   3 = goalkeeper_right (team:right)
#   4 = referee          (team:other)
#   5 = ball             (team:other)
GT_PLAYER_CLS   = {0, 1}
GT_GK_CLS       = {2, 3}
GT_REFEREE_CLS  = {4}
GT_BALL_CLS     = {5}
GT_PERSON_CLS   = GT_PLAYER_CLS | GT_GK_CLS | GT_REFEREE_CLS


def run_command(cmd, desc):
    """Run a shell command, capturing output. Raises on failure."""
    print(f"\n{'='*50}")
    print(f"-> {desc}")
    print(f"{'='*50}\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in {desc}")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout


# ── Ball Tracking Evaluation ─────────────────────────────────────────────────
def _bb_iou(b1, b2):
    """IoU between two (cx,cy,w,h) normalized boxes."""
    x1 = max(b1[0] - b1[2]/2, b2[0] - b2[2]/2)
    y1 = max(b1[1] - b1[3]/2, b2[1] - b2[3]/2)
    x2 = min(b1[0] + b1[2]/2, b2[0] + b2[2]/2)
    y2 = min(b1[1] + b1[3]/2, b2[1] + b2[3]/2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0


def evaluate_ball_tracking(seq_dir, trajectory_json):
    """
    Evaluates ball tracking against GT labels (class_id=5).
    Returns dict with F1, Precision@IoU50, RMSE, MAE, FPPI, Fragmentations.
    """
    lbl_folder = os.path.join(seq_dir, "labels")
    if not os.path.exists(lbl_folder):
        return {}

    img_dir = os.path.join(seq_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        return {}

    sample = cv2.imread(images[0])
    h, w = sample.shape[:2]

    with open(trajectory_json, 'r') as f:
        preds = json.load(f)
    # Only consider non-dummy predictions
    pred_frames = {int(p['frame_id']): p for p in preds if p['x'] != -1}

    TP, FP, FN = 0, 0, 0
    TP_iou50, FP_iou50, FN_iou50 = 0, 0, 0
    sq_errs, abs_errs = [], []
    frags = 0
    prev_dummy = False

    for img_path in images:
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # Load GT ball
        gt_ball = None
        lbl_file = os.path.join(lbl_folder, f"{frame_id:06d}.txt")
        if not os.path.exists(lbl_file):
            lbl_file = os.path.join(lbl_folder, f"{frame_id}.txt")

        if os.path.exists(lbl_file):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "5":
                        gt_ball = (float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4]))
                        break

        p = pred_frames.get(frame_id)
        pred_ball = (p['x']/w, p['y']/h, p['w']/w, p['h']/h) if p else None

        if gt_ball and pred_ball:
            dist = math.hypot(gt_ball[0] - pred_ball[0], gt_ball[1] - pred_ball[1])
            sq = ((gt_ball[0]*w - pred_ball[0]*w)**2 +
                  (gt_ball[1]*h - pred_ball[1]*h)**2)
            sq_errs.append(sq)
            abs_errs.append(math.sqrt(sq))

            if prev_dummy:
                frags += 1
            prev_dummy = False

            iou_val = _bb_iou(gt_ball, pred_ball)
            if iou_val >= 0.5:
                TP_iou50 += 1
            else:
                FP_iou50 += 1
                FN_iou50 += 1  # GT exists but not matched -> also a miss

            if dist <= 0.02:
                TP += 1
            else:
                FP += 1
                FN += 1

        elif gt_ball and not pred_ball:
            FN += 1
            FN_iou50 += 1
            prev_dummy = True
        elif not gt_ball and pred_ball:
            FP += 1
            FP_iou50 += 1
            prev_dummy = False

    eps = 1e-6
    prec = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    # Precision and Recall at IoU >= 0.5
    prec_iou50 = TP_iou50 / (TP_iou50 + FP_iou50 + eps)
    rec_iou50 = TP_iou50 / (TP_iou50 + FN_iou50 + eps)

    rmse = math.sqrt(sum(sq_errs) / len(sq_errs)) if sq_errs else 0
    mae = sum(abs_errs) / len(abs_errs) if abs_errs else 0

    return {
        "Ball_F1": round(f1, 4),
        "Ball_Prec_IoU50": round(prec_iou50, 4),
        "Ball_Rec_IoU50": round(rec_iou50, 4),
        "Ball_Prec": round(prec, 4),
        "Ball_Rec": round(rec, 4),
        "Ball_RMSE": round(rmse, 2),
        "Ball_MAE": round(mae, 2),
        "Ball_FPPI": round(FP / len(images), 4),
        "Ball_Frags": frags
    }


# ── IoU Matrix (vectorized) ─────────────────────────────────────────────────
def iou_matrix(gt_boxes, pred_boxes):
    """IoU matrix between lists of [x, y, w, h] boxes (absolute coords)."""
    if not gt_boxes or not pred_boxes:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    gt = np.array(gt_boxes)
    pr = np.array(pred_boxes)

    gt_x1, gt_y1 = gt[:, 0], gt[:, 1]
    gt_x2, gt_y2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    pr_x1, pr_y1 = pr[:, 0], pr[:, 1]
    pr_x2, pr_y2 = pr[:, 0] + pr[:, 2], pr[:, 1] + pr[:, 3]

    lt = np.maximum(gt_x1[:, None], pr_x1)
    rb = np.minimum(gt_x2[:, None], pr_x2)
    tp = np.maximum(gt_y1[:, None], pr_y1)
    bd = np.minimum(gt_y2[:, None], pr_y2)

    inter = np.maximum(0, rb - lt) * np.maximum(0, bd - tp)
    gt_area = gt[:, 2] * gt[:, 3]
    pr_area = pr[:, 2] * pr[:, 3]
    union = gt_area[:, None] + pr_area - inter
    return inter / np.maximum(union, 1e-6)


# ── MOT + Clustering Evaluation ─────────────────────────────────────────────
def evaluate_mot_and_clustering(seq_dir, det_json, team_json):
    """
    Evaluates:
    - MOT metrics (MOTA, IDF1, MOTP, ID Switches) using motmetrics
    - Team clustering accuracy via Hungarian matching against GT labels
    - Referee and Goalkeeper detection precision/recall

    Uses the SoccerNet 6-class GT taxonomy:
      0,1 = player (left,right), 2,3 = goalkeeper (left,right), 4 = referee, 5 = ball
    """
    if not HAS_MOT:
        return {}

    lbl_folder = os.path.join(seq_dir, "labels")
    if not os.path.exists(lbl_folder):
        return {}

    img_dir = os.path.join(seq_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        return {}

    sample = cv2.imread(images[0])
    h, w = sample.shape[:2]

    # Load detections
    with open(det_json, 'r') as f:
        dets = json.load(f)
    pred_frames = {d['frame_id']: d for d in dets}

    # Load team assignments
    team_map = {}
    if os.path.exists(team_json):
        with open(team_json, 'r') as f:
            for item in json.load(f):
                team_map[item['track_id']] = item

    acc = mm.MOTAccumulator(auto_id=True)

    # Per-track voting: pred_track_id -> {gt_track_id: frame_count}
    track_id_votes = {}

    # GT role info: gt_track_id -> {"team": "left"/"right"/"other", "is_gk": bool, "is_ref": bool}
    gt_role_map = {}

    for img_path in images:
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # ── Load GT ──
        gt_boxes, gt_ids = [], []
        lbl_file = os.path.join(lbl_folder, f"{frame_id:06d}.txt")
        if not os.path.exists(lbl_file):
            lbl_file = os.path.join(lbl_folder, f"{frame_id}.txt")

        if os.path.exists(lbl_file):
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cls_id = int(parts[0])
                        if cls_id in GT_BALL_CLS:
                            continue  # Skip ball for MOT

                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                        gt_id = int(parts[5])

                        # Parse team from comment
                        team_tag = "other"
                        if "# team:left" in line:
                            team_tag = "left"
                        elif "# team:right" in line:
                            team_tag = "right"

                        gt_role_map[gt_id] = {
                            "team": team_tag,
                            "is_gk": cls_id in GT_GK_CLS,
                            "is_ref": cls_id in GT_REFEREE_CLS,
                            "cls_id": cls_id
                        }

                        box_abs = [
                            (cx - bw / 2) * w,
                            (cy - bh / 2) * h,
                            bw * w,
                            bh * h
                        ]
                        gt_boxes.append(box_abs)
                        gt_ids.append(gt_id)

        # ── Load predictions ──
        pred_boxes, pred_ids = [], []
        frame_preds = pred_frames.get(frame_id, {}).get("players", [])
        for p in frame_preds:
            tid = p.get('track_id', -1)
            if tid == -1:
                continue
            bw = p['x_max'] - p['x_min']
            bh = p['y_max'] - p['y_min']
            pred_boxes.append([p['x_min'], p['y_min'], bw, bh])
            pred_ids.append(tid)

        # ── IoU + MOT update ──
        custom_iou = iou_matrix(gt_boxes, pred_boxes)

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            dists = 1.0 - custom_iou
            dists[dists > 0.5] = np.nan  # motmetrics uses NaN for "no match"
        else:
            dists = np.empty((len(gt_boxes), len(pred_boxes)))

        acc.update(gt_ids, pred_ids, dists)

        # ── Track-level voting for clustering eval ──
        for i, g_id in enumerate(gt_ids):
            for j, p_id in enumerate(pred_ids):
                if len(gt_boxes) > 0 and len(pred_boxes) > 0 and custom_iou[i, j] >= 0.5:
                    if p_id not in track_id_votes:
                        track_id_votes[p_id] = {}
                    track_id_votes[p_id][g_id] = track_id_votes[p_id].get(g_id, 0) + 1

    # ── Compute MOT metrics ──
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'motp', 'num_switches'], name='acc')

    # ── Resolve pred → gt mapping (majority vote per track) ──
    pred_to_gt = {}
    for pid, votes in track_id_votes.items():
        if votes:
            best_gt = max(votes.items(), key=lambda x: x[1])[0]
            pred_to_gt[pid] = best_gt

    # ── Evaluate Team Clustering ──
    # Count matches for both possible permutations:
    #   Perm A: Team0=left, Team1=right
    #   Perm B: Team0=right, Team1=left
    perm_a_correct = 0  # matches if Team0=left, Team1=right
    perm_b_correct = 0  # matches if Team0=right, Team1=left
    total_team_players = 0

    # ── Role evaluation ──
    tp_ref, fp_ref = 0, 0
    tp_gk, fp_gk = 0, 0
    gt_refs_found = set()
    gt_gks_found = set()

    for pid, gt_id in pred_to_gt.items():
        gt_info = gt_role_map.get(gt_id)
        if not gt_info:
            continue

        pred_info = team_map.get(pid, {"team_id": -1, "role": "player"})
        p_team = pred_info.get("team_id", -1)
        p_role = pred_info.get("role", "player")

        # ── Referee evaluation ──
        if p_role == "referee":
            if gt_info["is_ref"]:
                tp_ref += 1
                gt_refs_found.add(gt_id)
            else:
                fp_ref += 1

        # ── Goalkeeper evaluation ──
        if p_role == "goalkeeper":
            if gt_info["is_gk"]:
                tp_gk += 1
                gt_gks_found.add(gt_id)
            else:
                fp_gk += 1

        # ── Team clustering (only field players + GKs with a real team) ──
        g_team = gt_info["team"]
        if p_team in [0, 1] and g_team in ["left", "right"]:
            total_team_players += 1
            if (p_team == 0 and g_team == "left") or (p_team == 1 and g_team == "right"):
                perm_a_correct += 1
            if (p_team == 0 and g_team == "right") or (p_team == 1 and g_team == "left"):
                perm_b_correct += 1

    best_team_accuracy = max(perm_a_correct, perm_b_correct) / max(1, total_team_players)

    # FN for roles (unique GT tracks not found)
    total_gt_refs = len([g for g in gt_role_map.values() if g["is_ref"]])
    total_gt_gks = len([g for g in gt_role_map.values() if g["is_gk"]])
    fn_ref = total_gt_refs - len(gt_refs_found)
    fn_gk = total_gt_gks - len(gt_gks_found)

    eps = 1e-6
    ref_prec = tp_ref / (tp_ref + fp_ref + eps)
    ref_rec = tp_ref / (tp_ref + fn_ref + eps) if (tp_ref + fn_ref) > 0 else 0
    gk_prec = tp_gk / (tp_gk + fp_gk + eps)
    gk_rec = tp_gk / (tp_gk + fn_gk + eps) if (tp_gk + fn_gk) > 0 else 0

    def safe_float(val):
        return float(val) if not np.isnan(val) else 0.0

    return {
        "MOTA": round(safe_float(summary['mota'].iloc[0]), 4),
        "IDF1": round(safe_float(summary['idf1'].iloc[0]), 4),
        "MOTP": round(safe_float(summary['motp'].iloc[0]), 4),
        "ID_Switches": int(summary['num_switches'].iloc[0]),
        "Team_Clustering_Acc": round(best_team_accuracy, 4),
        "Ref_Prec": round(ref_prec, 4),
        "Ref_Rec": round(ref_rec, 4),
        "GK_Prec": round(gk_prec, 4),
        "GK_Rec": round(gk_rec, 4),
        "Tracks_Eval": len(pred_to_gt),
        "GT_Refs": total_gt_refs,
        "GT_GKs": total_gt_gks,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(
        description="Batch Evaluator: TacticalVision Pipeline")
    parser.add_argument("--sequences", nargs="+",
                        help="Specific sequences to evaluate (e.g. SNMOT-197). If omitted, runs all 49.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pipeline phases if output JSONs already exist")
    return parser.parse_args()


def main():
    args = get_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "tracking", "SoccerNet",
                            "tracking", "test", "test")

    if args.sequences:
        seqs = [os.path.join(data_dir, s) for s in args.sequences]
    else:
        seqs = sorted([
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

    out_dir = os.path.join(base_dir, "results_final", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "batch_evaluation.csv")

    fieldnames = [
        "Sequence",
        "Ball_F1", "Ball_Prec_IoU50", "Ball_Rec_IoU50",
        "Ball_Prec", "Ball_Rec", "Ball_RMSE", "Ball_MAE", "Ball_FPPI", "Ball_Frags",
        "MOTA", "IDF1", "MOTP", "ID_Switches",
        "Team_Clustering_Acc",
        "Ref_Prec", "Ref_Rec", "GK_Prec", "GK_Rec",
        "Tracks_Eval", "GT_Refs", "GT_GKs"
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    yolo_w = os.path.join(base_dir, "models", "yolo26.pt")
    rfd_w = os.path.join(base_dir, "models",
                         "models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth")

    for seq in seqs:
        seq_name = os.path.basename(seq)
        seq_out = os.path.join(base_dir, "outputs", seq_name)
        os.makedirs(seq_out, exist_ok=True)

        det_json  = os.path.join(seq_out, f"{seq_name}_detections.json")
        cmc_json  = os.path.join(seq_out, f"{seq_name}_cmc.json")
        traj_json = os.path.join(seq_out, f"{seq_name}_trajectory.json")
        team_json = os.path.join(seq_out, f"{seq_name}_team_assignments.json")
        team_video = os.path.join(seq_out, f"{seq_name}_team_clustering.mp4")

        print(f"\n{'='*60}")
        print(f"PROCESSING: {seq_name}")
        print(f"{'='*60}")

        try:
            # P1: Feature Extraction
            if args.skip_existing and os.path.exists(det_json):
                print(f"[skip] P1: {det_json} exists.")
            else:
                cmd = (f"python {os.path.join(base_dir, 'core', 'detection', 'tactical_vision_extractor.py')}"
                       f" --sequence_dir {seq} --yolo_weights {yolo_w}"
                       f" --rfdetr_weights {rfd_w} --output_json {det_json}")
                run_command(cmd, f"P1: Feature Extraction ({seq_name})")

            # P2: CMC
            if args.skip_existing and os.path.exists(cmc_json):
                print(f"[skip] P2: {cmc_json} exists.")
            else:
                cmd = (f"python {os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_cmc.py')}"
                       f" --sequence_dir {seq} --output_json {cmc_json} --fast")
                run_command(cmd, f"P2: CMC ({seq_name})")

            # P3-P5: HMM Viterbi
            if args.skip_existing and os.path.exists(traj_json):
                print(f"[skip] P3-P5: {traj_json} exists.")
            else:
                cmd = (f"python {os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_hmm.py')}"
                       f" --detections_json {det_json} --cmc_json {cmc_json}"
                       f" --output_json {traj_json}")
                run_command(cmd, f"P3-P5: HMM Viterbi ({seq_name})")

            # P8: Team Clustering
            if args.skip_existing and os.path.exists(team_json):
                print(f"[skip] P8: {team_json} exists.")
            else:
                cmd = (f"python {os.path.join(base_dir, 'core', 'clustering', 'team_clustering_phase.py')}"
                       f" --sequence_dir {seq} --rfdetr_weights {rfd_w}"
                       f" --detections_json {det_json} --output_json {team_json}"
                       f" --output_video {team_video}")
                run_command(cmd, f"P8: Team Clustering ({seq_name})")

            # ── Metrics calculation ──
            print(f"Calculating metrics for {seq_name}...")
            metrics = {"Sequence": seq_name}

            ball_m = evaluate_ball_tracking(seq, traj_json)
            metrics.update(ball_m)

            mot_m = evaluate_mot_and_clustering(seq, det_json, team_json)
            metrics.update(mot_m)

            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(metrics)

            print(f"[OK] {seq_name} | F1={metrics.get('Ball_F1',0):.3f}"
                  f" | MOTA={metrics.get('MOTA',0):.3f}"
                  f" | Clust={metrics.get('Team_Clustering_Acc',0):.3f}"
                  f" | Ref={metrics.get('Ref_Rec',0):.2f}"
                  f" | GK={metrics.get('GK_Rec',0):.2f}")

        except Exception as e:
            print(f"[FAIL] {seq_name}: {e}")
            metrics = {"Sequence": seq_name}
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(metrics)

    print(f"\nALL BATCH EVALUATIONS COMPLETE. Results: {csv_file}")


if __name__ == "__main__":
    main()
