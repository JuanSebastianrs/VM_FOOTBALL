"""
tactical_vision_optimize.py
──────────────────────────────────────────────────────────────────────────────
Phase 10: Bayesian Hyperparameter Optimization for the Viterbi HMM Tracker.

Methodology:
  - Optimizes on a REPRESENTATIVE SUBSET of TRAIN sequences (~10 videos
    with diverse conditions: occlusions, fast pans, clean visibility).
  - Uses a corrected composite loss that does NOT penalize legitimate
    Dummy Node usage during real occlusions:
        L = (1 - F1) + CLE / 1000
  - Exports best_hmm_params.yaml for production pipeline consumption.

Output:  best_hmm_params.yaml
──────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
import json
import math
import numpy as np
import optuna

# ═══════════════════════════════════════════════════════════════════════════
#  REPRESENTATIVE TRAIN SUBSET (stratified: easy + occlusions + pan/blur)
#  These are train sequences with cached Phase 1+2 JSONs and GT labels.
#  Chosen to cover diverse match conditions without overfitting.
# ═══════════════════════════════════════════════════════════════════════════

TRAIN_SUBSET = [
    "SNMOT-060",  # Standard match conditions
    "SNMOT-062",  # Different camera angle
    "SNMOT-064",  # Varied player density
    "SNMOT-066",  # Fast camera panning
    "SNMOT-068",  # Multiple occlusion events
    "SNMOT-070",  # Wide-angle view
    "SNMOT-072",  # Close-up segments
    "SNMOT-074",  # Mixed conditions
    "SNMOT-164",  # Reference sequence (train benchmark)
]

BATCH_DIR  = r"d:\sebastian\Tesis\VM_FOOTBALL\tactical_results\batch_eval"
TRAIN_ROOT = r"d:\sebastian\Tesis\VM_FOOTBALL\data\tracking\SoccerNet\tracking\train\train"


# ═══════════════════════════════════════════════════════════════════════════
#                          PURE VITERBI ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class Node:
    __slots__ = ['idx', 'frame_id', 'x', 'y', 'w', 'h',
                 'area', 'score', 'is_dummy', 'attached_player_id']

    def __init__(self, idx, frame_id, x, y, w, h, score=1.0, is_dummy=False):
        self.idx = idx
        self.frame_id = frame_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h if (w and h) else 0.0
        self.score = score
        self.is_dummy = is_dummy
        self.attached_player_id = None


def _kinematic_cost(u, v, M, delta_P):
    if u.x is None or v.x is None:
        return 0.0
    p_proj = np.dot(M, np.array([u.x, u.y, 1.0]))
    dist = np.hypot(v.x - p_proj[0], v.y - p_proj[1])
    if dist > delta_P:
        return float('inf')
    return (dist / delta_P) ** 2


def _appearance_cost(u, v):
    if u.is_dummy or v.is_dummy or u.area <= 0 or v.area <= 0:
        return 0.0
    return abs(np.log(u.area / v.area))


def _intersects_feet(bx, by, player, foot_pct):
    xmin, ymin = player['x_min'], player['y_min']
    xmax, ymax = player['x_max'], player['y_max']
    if bx < xmin or bx > xmax:
        return False
    threshold_y = ymin + (1.0 - foot_pct) * (ymax - ymin)
    return by >= threshold_y and by <= ymax


def run_viterbi_pipeline(detections, cmc_matrices, params):
    """
    Execute full Viterbi decoding with arbitrary hyperparameters.
    Returns list[dict] with {frame_id, x, y, w, h, is_dummy} per frame.
    """
    w1       = params['w1']
    w2       = params['w2']
    w3       = params['w3']
    delta_P  = params['delta_P']
    lam      = params['lambda_cost']
    alpha    = params['alpha']
    score_rwd = params['score_reward']
    foot_pct = params.get('foot_pct', 0.30)

    T = len(detections)
    V = [{} for _ in range(T)]

    # Frame 0
    f0 = detections[0]
    for i, c in enumerate(f0['ball_candidates']):
        V[0][i] = {
            'cost': -score_rwd * c['score'],
            'prev': None,
            'state': Node(i, 0, c['x_center'], c['y_center'],
                          c['w'], c['h'], c['score'], False),
        }
    d0 = len(f0['ball_candidates'])
    V[0][d0] = {
        'cost': lam,
        'prev': None,
        'state': Node(d0, 0, None, None, None, None, 0.0, True),
    }

    # Forward pass
    for t in range(1, T):
        prev, curr = detections[t - 1], detections[t]
        key = f"{prev['frame_id']}->{curr['frame_id']}"
        M = np.array(cmc_matrices.get(key, np.eye(2, 3, dtype=np.float32)))
        players_dict = {p['track_id']: p for p in curr['players']}

        # Normal targets
        for i, c in enumerate(curr['ball_candidates']):
            v_nd = Node(i, t, c['x_center'], c['y_center'],
                        c['w'], c['h'], c['score'], False)
            best_c, best_p = float('inf'), None
            for u_idx, u_info in V[t - 1].items():
                u_nd = u_info['state']
                Ck = _kinematic_cost(u_nd, v_nd, M, delta_P)
                if Ck == float('inf'):
                    continue
                Ca = _appearance_cost(u_nd, v_nd)
                Cp = lam if u_nd.is_dummy else 0.0
                tc = u_info['cost'] + w1*Ck + w2*Ca + w3*Cp - score_rwd*v_nd.score
                if tc < best_c:
                    best_c, best_p = tc, u_idx
            V[t][i] = {'cost': best_c, 'prev': best_p, 'state': v_nd}

        # Dummy target
        d_idx = len(curr['ball_candidates'])
        best_dc, best_dp, best_ds = float('inf'), None, None
        for u_idx, u_info in V[t - 1].items():
            u_nd = u_info['state']
            Cp = lam
            ds = Node(d_idx, t, None, None, None, None, 0.0, True)

            if not u_nd.is_dummy:
                attached = None
                for pl in prev['players']:
                    if _intersects_feet(u_nd.x, u_nd.y, pl, foot_pct):
                        attached = pl; break
                if attached:
                    Cp = lam / alpha
                    ds.attached_player_id = attached['track_id']
                    if attached['track_id'] in players_dict:
                        cp = players_dict[attached['track_id']]
                        ds.x = (cp['x_min'] + cp['x_max']) / 2.0
                        ds.y = (cp['y_min'] + cp['y_max']) / 2.0
                        ds.w, ds.h = u_nd.w, u_nd.h
                else:
                    if u_nd.x is not None:
                        p_proj = np.dot(M, np.array([u_nd.x, u_nd.y, 1.0]))
                        ds.x, ds.y = p_proj[0], p_proj[1]
                    ds.w, ds.h = u_nd.w, u_nd.h
            else:
                Cp = 0.0
                ds.attached_player_id = u_nd.attached_player_id
                if ds.attached_player_id and ds.attached_player_id in players_dict:
                    cp = players_dict[ds.attached_player_id]
                    ds.x = (cp['x_min'] + cp['x_max']) / 2.0
                    ds.y = (cp['y_min'] + cp['y_max']) / 2.0
                elif u_nd.x is not None:
                    p_proj = np.dot(M, np.array([u_nd.x, u_nd.y, 1.0]))
                    ds.x, ds.y = p_proj[0], p_proj[1]
                ds.w, ds.h = u_nd.w, u_nd.h

            tc = u_info['cost'] + w1*0.0 + w2*0.0 + w3*Cp
            if tc < best_dc:
                best_dc, best_dp, best_ds = tc, u_idx, ds

        V[t][d_idx] = {'cost': best_dc, 'prev': best_dp, 'state': best_ds}

    # Backtrack
    last = T - 1
    best_end = min(V[last], key=lambda k: V[last][k]['cost'])
    path, cur = [], best_end
    for t in range(last, -1, -1):
        info = V[t][cur]
        path.append(info['state'])
        cur = info['prev']
    path.reverse()

    # Interpolate unattached dummies
    for i in range(len(path)):
        nd = path[i]
        if nd.is_dummy and nd.x is None:
            pv, nv = None, None
            for j in range(i - 1, -1, -1):
                if path[j].x is not None and path[j].w is not None:
                    pv = path[j]; break
            for j in range(i + 1, len(path)):
                if path[j].x is not None and path[j].w is not None:
                    nv = path[j]; break
            if pv and nv and nv.frame_id != pv.frame_id:
                a = (i - pv.frame_id) / (nv.frame_id - pv.frame_id)
                nd.x = pv.x + a * (nv.x - pv.x)
                nd.y = pv.y + a * (nv.y - pv.y)
                nd.w = pv.w + a * ((nv.w or pv.w) - pv.w)
                nd.h = pv.h + a * ((nv.h or pv.h) - pv.h)
            elif pv:
                nd.x, nd.y, nd.w, nd.h = pv.x, pv.y, pv.w, pv.h
            elif nv:
                nd.x, nd.y, nd.w, nd.h = nv.x, nv.y, nv.w, nv.h

    # Build output
    trajectory = []
    for nd in path:
        trajectory.append({
            'frame_id': detections[nd.frame_id]['frame_id'],
            'x': float(nd.x) if nd.x is not None else -1,
            'y': float(nd.y) if nd.y is not None else -1,
            'w': float(nd.w) if nd.w is not None else -1,
            'h': float(nd.h) if nd.h is not None else -1,
            'is_dummy': nd.is_dummy,
        })
    return trajectory


# ═══════════════════════════════════════════════════════════════════════════
#                        EVALUATION (Corrected)
# ═══════════════════════════════════════════════════════════════════════════

def _box_iou_norm(a, b):
    ax1, ay1 = a[0] - a[2]/2, a[1] - a[3]/2
    ax2, ay2 = a[0] + a[2]/2, a[1] + a[3]/2
    bx1, by1 = b[0] - b[2]/2, b[1] - b[3]/2
    bx2, by2 = b[0] + b[2]/2, b[1] + b[3]/2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / (union + 1e-6)


def evaluate_trajectory(trajectory, gt_labels, img_w=1920, img_h=1080):
    """
    Returns (mean_cle, f1) — NO dummy_ratio penalty.
    CLE naturally penalizes degenerate straight-line solutions because
    a straight line diverges from the real parabolic ball trajectory.
    """
    pred_map = {}
    for p in trajectory:
        fid = int(p['frame_id'])
        if p['x'] != -1 and p['y'] != -1:
            pred_map[fid] = (p['x'] / img_w, p['y'] / img_h,
                             max(p['w'], 1) / img_w, max(p['h'], 1) / img_h)

    cle_list = []
    tp50, fp50, fn = 0, 0, 0

    for fid, gt in gt_labels.items():
        pred = pred_map.get(fid)
        if pred is None:
            fn += 1
            continue
        dx = (gt[0] - pred[0]) * img_w
        dy = (gt[1] - pred[1]) * img_h
        cle_list.append(math.hypot(dx, dy))

        if _box_iou_norm(gt, pred) >= 0.5:
            tp50 += 1
        else:
            fp50 += 1

    eps = 1e-6
    mean_cle = np.mean(cle_list) if cle_list else 500.0
    prec = tp50 / (tp50 + fp50 + eps)
    rec  = tp50 / (tp50 + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)

    return mean_cle, f1


# ═══════════════════════════════════════════════════════════════════════════
#                       DATA LOADING (Cached)
# ═══════════════════════════════════════════════════════════════════════════

def load_gt_labels(seq_dir):
    lbl_dir = os.path.join(seq_dir, "labels")
    if not os.path.exists(lbl_dir):
        return {}
    gt = {}
    for lbl_path in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
        fid = int(os.path.splitext(os.path.basename(lbl_path))[0])
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == "5":
                    gt[fid] = tuple(float(p) for p in parts[1:5])
                    break
    return gt


_CACHED_DATA = None

def _get_train_data():
    """Load the representative subset into RAM (once)."""
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    print("[INIT] Loading REPRESENTATIVE TRAIN SUBSET into RAM...")
    loaded = []
    for seq_name in TRAIN_SUBSET:
        det_path = os.path.join(BATCH_DIR, f"{seq_name}_detections.json")
        cmc_path = os.path.join(BATCH_DIR, f"{seq_name}_cmc.json")

        if not os.path.exists(det_path) or not os.path.exists(cmc_path):
            print(f"   ✗ {seq_name} — missing cached JSONs, skipping")
            continue

        seq_dir = os.path.join(TRAIN_ROOT, seq_name)
        gt = load_gt_labels(seq_dir)
        if len(gt) < 10:
            print(f"   ✗ {seq_name} — insufficient GT labels ({len(gt)}), skipping")
            continue

        with open(det_path, 'r') as f:
            dets = json.load(f)
        with open(cmc_path, 'r') as f:
            cmc = json.load(f)

        loaded.append({
            'name': seq_name,
            'detections': dets,
            'cmc': cmc,
            'gt': gt,
        })
        print(f"   ✓ {seq_name}  ({len(dets)} frames, {len(gt)} GT labels)")

    print(f"[INIT] Loaded {len(loaded)} sequences ({sum(len(s['detections']) for s in loaded)} total frames)\n")
    _CACHED_DATA = loaded
    return _CACHED_DATA


# ═══════════════════════════════════════════════════════════════════════════
#                        OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════

def objective(trial):
    """
    Corrected composite loss (no dummy_ratio penalty):
        L = (1 - F1) + CLE / 1000

    Why no dummy_ratio?
    - Penalizing dummy usage suppresses the Dummy Node, destroying ORR.
    - If Viterbi degenerates to a straight line, CLE naturally spikes
      because a line can't follow the ball's real parabolic trajectory.
    """
    # Search Space
    w1 = trial.suggest_float("w1", 0.3, 0.8)
    w2 = trial.suggest_float("w2", 0.05, 0.4)
    w3_raw = 1.0 - w1 - w2
    if w3_raw < 0.01:
        raise optuna.TrialPruned()

    params = {
        'w1': w1,
        'w2': w2,
        'w3': w3_raw,
        'delta_P':      trial.suggest_float("delta_P", 50.0, 800.0, log=True),
        'lambda_cost':  trial.suggest_float("lambda_cost", 1.0, 50.0, log=True),
        'alpha':        trial.suggest_float("alpha", 1.5, 15.0),
        'score_reward': trial.suggest_float("score_reward", 10.0, 100.0),
        'foot_pct':     trial.suggest_float("foot_pct", 0.15, 0.50),
    }

    data = _get_train_data()
    if not data:
        return 999.0

    losses = []
    for seq in data:
        traj = run_viterbi_pipeline(seq['detections'], seq['cmc'], params)
        cle, f1 = evaluate_trajectory(traj, seq['gt'])

        # Corrected loss: does NOT penalize legitimate Dummy Node usage
        loss = (1.0 - f1) + (cle / 1000.0)
        losses.append(loss)

    return np.mean(losses)


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization of Viterbi HMM (Optuna)")
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--output_yaml", type=str,
                        default=r"d:\sebastian\Tesis\VM_FOOTBALL\best_hmm_params.yaml")
    args = parser.parse_args()

    _get_train_data()

    print("=" * 70)
    print("  OPTUNA BAYESIAN OPTIMIZATION  –  TacticalVision Viterbi HMM")
    print(f"  Trials: {args.n_trials}  |  Sampler: TPE (seed=42)")
    print(f"  Train Subset: {len(TRAIN_SUBSET)} representative sequences")
    print("  Loss:  L = (1 - F1) + CLE/1000  (no dummy penalty)")
    print("=" * 70 + "\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="viterbi_hmm_optimization",
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   show_progress_bar=True)

    # Results
    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\n  Best Loss: {study.best_value:.6f}")
    print(f"  Best Trial: #{study.best_trial.number}\n")
    print("  Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k:>15s} = {v:.6f}")

    best = dict(study.best_params)
    best['w3'] = round(1.0 - best['w1'] - best['w2'], 6)
    print(f"    {'w3':>15s} = {best['w3']:.6f}  (derived)")

    # Export YAML
    import yaml
    yaml_out = {
        'viterbi_hyperparameters': {
            'w1': round(best['w1'], 6),
            'w2': round(best['w2'], 6),
            'w3': round(best['w3'], 6),
            'delta_P': round(best['delta_P'], 2),
            'lambda_cost': round(best['lambda_cost'], 4),
            'alpha': round(best['alpha'], 4),
            'score_reward': round(best['score_reward'], 4),
            'foot_pct': round(best['foot_pct'], 4),
        },
        'optimization_metadata': {
            'best_loss': round(study.best_value, 6),
            'loss_formula': 'L = (1 - F1) + CLE/1000',
            'n_trials': args.n_trials,
            'sampler': 'TPESampler(seed=42)',
            'train_subset': TRAIN_SUBSET,
        },
    }

    with open(args.output_yaml, 'w') as f:
        yaml.dump(yaml_out, f, default_flow_style=False, sort_keys=False)

    print(f"\n  ✓ Exported to: {args.output_yaml}")
    print("  → Apply these params to batch_eval_test.py on TEST sequences")
    print("=" * 70)


if __name__ == "__main__":
    main()
