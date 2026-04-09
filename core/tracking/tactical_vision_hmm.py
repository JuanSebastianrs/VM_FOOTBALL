import os
import json
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Phase 3-4-5: HMM Graph Construction and Viterbi")
    parser.add_argument("--detections_json", type=str, default="secuencia_197_detections.json")
    parser.add_argument("--cmc_json", type=str, default="secuencia_197_cmc.json")
    parser.add_argument("--output_json", type=str, default="secuencia_197_trajectory.json")
    parser.add_argument("--params_yaml", type=str, default="", help="Path to best_hmm_params.yaml")
    return parser.parse_args()

class Node:
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

def compute_kinematic_cost(node_u, node_v, M_t_t1, delta_P=100.0):
    """
    node_u (frame t), node_v (frame t+1)
    M_t_t1: 2x3 affine matrix from CMC
    """
    if node_u.x is None or node_v.x is None:
        return 0.0 # Ignore distance if dummy has no position
    
    # Project u to t+1
    p_u = np.array([node_u.x, node_u.y, 1.0])
    p_u_proj = np.dot(M_t_t1, p_u)
    
    p_v = np.array([node_v.x, node_v.y])
    
    # Euclidean distance
    dist = np.linalg.norm(p_v - p_u_proj)
    
    # Gating
    if dist > delta_P:
        return float('inf')
        
    # Using squared scaled distance as Mahalanobis approximation
    return (dist / delta_P)**2

def compute_appearance_cost(node_u, node_v):
    if node_u.is_dummy or node_v.is_dummy:
        return 0.0
    if node_u.area <= 0 or node_v.area <= 0:
        return 0.0
    ratio = node_u.area / node_v.area
    return abs(np.log(ratio))

def is_intersecting_bottom_30(bx, by, player, foot_pct=0.30):
    xmin, ymin, xmax, ymax = player['x_min'], player['y_min'], player['x_max'], player['y_max']
    
    # Check if point is inside player horizontally
    if bx >= xmin and bx <= xmax:
        # Check if point is in bottom % vertically
        height = ymax - ymin
        threshold_y = ymin + (1.0 - foot_pct) * height
        if by >= threshold_y and by <= ymax:
            return True
    return False

def build_graph_and_viterbi():
    args = get_args()
    
    with open(args.detections_json, 'r') as f:
        detections = json.load(f)
        
    with open(args.cmc_json, 'r') as f:
        cmc_matrices = json.load(f)
        
    # Default Hyperparameters
    w1, w2, w3 = 0.6, 0.2, 0.2
    delta_P = 500.0
    lambda_cost = 10.0
    alpha = 5.0
    score_reward = 50.0 # Default from original code
    foot_pct = 0.30

    if args.params_yaml and os.path.exists(args.params_yaml):
        import yaml
        with open(args.params_yaml, 'r') as f:
            y_data = yaml.safe_load(f)
            if 'viterbi_hyperparameters' in y_data:
                p = y_data['viterbi_hyperparameters']
                w1, w2, w3 = p['w1'], p['w2'], p['w3']
                delta_P = p['delta_P']
                lambda_cost = p['lambda_cost']
                alpha = p['alpha']
                score_reward = p.get('score_reward', score_reward)
                foot_pct = p.get('foot_pct', foot_pct)
        print(f"Loaded optimized parameters from {args.params_yaml}")
    
    # Viterbi tables
    # DP State: V[t][node_idx] = (min_cost, prev_node_idx, (x, y, w, h, attached_player_id))
    V = [{} for _ in range(len(detections))]
    
    # Initialize frame 0
    f0_data = detections[0]
    for idx, c in enumerate(f0_data['ball_candidates']):
        V[0][idx] = {
            'cost': -score_reward * c['score'], # Use optimized reward
            'prev': None,
            'state': Node(idx, 0, c['x_center'], c['y_center'], c['w'], c['h'], c['score'], False)
        }
    # Dummy node for frame 0
    dummy_idx = len(f0_data['ball_candidates'])
    V[0][dummy_idx] = {
        'cost': lambda_cost,
        'prev': None,
        'state': Node(dummy_idx, 0, None, None, None, None, 0.0, True)
    }

    print(f"Running Viterbi Decoding over {len(detections)} frames...")
    
    for t in range(1, len(detections)):
        prev_data = detections[t-1]
        curr_data = detections[t]
        
        # Get matrix M_{t-1 -> t}
        key = f"{prev_data['frame_id']}->{curr_data['frame_id']}"
        M = np.array(cmc_matrices.get(key, np.eye(2, 3, dtype=np.float32)))
        
        curr_candidates = curr_data['ball_candidates']
        curr_players_dict = {p['track_id']: p for p in curr_data['players']}
        
        # Generate target nodes
        target_nodes = []
        for i, c in enumerate(curr_candidates):
            target_nodes.append(Node(i, t, c['x_center'], c['y_center'], c['w'], c['h'], c['score'], False))
        
        # Dummy node
        d_idx = len(curr_candidates)
        # We will dynamically set dummy node state based on the best path to it
        
        # Compute transitions from t-1 to t
        # For normal node targets
        for v_node in target_nodes:
            best_cost = float('inf')
            best_prev = None
            best_state = v_node
            
            for u_idx, u_info in V[t-1].items():
                u_node = u_info['state']
                base_cost = u_info['cost']
                
                # C_k
                C_k = compute_kinematic_cost(u_node, v_node, M, delta_P)
                if C_k == float('inf'):
                    continue
                    
                # C_a
                C_a = compute_appearance_cost(u_node, v_node)
                
                # C_p (Dummy to Normal, or Normal to Normal)
                if u_node.is_dummy:
                    C_p = lambda_cost # Penalize reappearing from occlusion to avoid flickering
                else:
                    C_p = 0.0
                    
                transition_cost = w1 * C_k + w2 * C_a + w3 * C_p - score_reward * v_node.score
                total_cost = base_cost + transition_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_prev = u_idx
            
            V[t][v_node.idx] = {
                'cost': best_cost,
                'prev': best_prev,
                'state': best_state
            }
            
        # For Dummy node target
        best_dummy_cost = float('inf')
        best_dummy_prev = None
        best_dummy_state = None
        
        for u_idx, u_info in V[t-1].items():
            u_node = u_info['state']
            base_cost = u_info['cost']
            
            C_k = 0.0
            C_a = 0.0
            C_p = lambda_cost # default penalty
            
            next_dummy_state = Node(d_idx, t, None, None, None, None, 0.0, True)
            
            if not u_node.is_dummy:
                # Normal -> Dummy
                # Check for intersection with players in frame t-1
                # To be precise, we check intersection in frame t-1, and attach to player's track
                intersected_player = None
                for player in prev_data['players']:
                    if is_intersecting_bottom_30(u_node.x, u_node.y, player, foot_pct):
                        intersected_player = player
                        break
                        
                if intersected_player:
                    C_p = lambda_cost / alpha
                    next_dummy_state.attached_player_id = intersected_player['track_id']
                    # Attach position to player centroid in frame t
                    if intersected_player['track_id'] in curr_players_dict:
                        cp = curr_players_dict[intersected_player['track_id']]
                        next_dummy_state.x = (cp['x_min'] + cp['x_max']) / 2.0
                        next_dummy_state.y = (cp['y_min'] + cp['y_max']) / 2.0
                        next_dummy_state.w = u_node.w
                        next_dummy_state.h = u_node.h
                else:
                    # Carried over position using CMC
                    p_u = np.array([u_node.x, u_node.y, 1.0])
                    p_proj = np.dot(M, p_u)
                    next_dummy_state.x = p_proj[0]
                    next_dummy_state.y = p_proj[1]
                    next_dummy_state.w = u_node.w
                    next_dummy_state.h = u_node.h
            else:
                # Dummy -> Dummy
                C_p = 0.0
                next_dummy_state.attached_player_id = u_node.attached_player_id
                if next_dummy_state.attached_player_id is not None and next_dummy_state.attached_player_id in curr_players_dict:
                    cp = curr_players_dict[next_dummy_state.attached_player_id]
                    next_dummy_state.x = (cp['x_min'] + cp['x_max']) / 2.0
                    next_dummy_state.y = (cp['y_min'] + cp['y_max']) / 2.0
                else:
                    if u_node.x is not None:
                        p_u = np.array([u_node.x, u_node.y, 1.0])
                        p_proj = np.dot(M, p_u)
                        next_dummy_state.x = p_proj[0]
                        next_dummy_state.y = p_proj[1]
                next_dummy_state.w = u_node.w
                next_dummy_state.h = u_node.h

            transition_cost = w1 * C_k + w2 * C_a + w3 * C_p
            total_cost = base_cost + transition_cost
            
            if total_cost < best_dummy_cost:
                best_dummy_cost = total_cost
                best_dummy_prev = u_idx
                best_dummy_state = next_dummy_state
                
        V[t][d_idx] = {
            'cost': best_dummy_cost,
            'prev': best_dummy_prev,
            'state': best_dummy_state
        }

    # Backtracking
    print("Backtracking optimal trajectory...")
    opt_path = []
    
    # find min cost in last frame
    last_t = len(detections) - 1
    best_last_node = min(V[last_t].keys(), key=lambda k: V[last_t][k]['cost'])
    
    curr_node = best_last_node
    for t in range(last_t, -1, -1):
        info = V[t][curr_node]
        opt_path.append(info['state'])
        curr_node = info['prev']
        
    opt_path.reverse()
    
    # Mathematical Interpolation for remaining Dummy nodes without attachment
    print("Interpolating unattached Dummy segments...")
    for i in range(len(opt_path)):
        if opt_path[i].is_dummy and opt_path[i].x is None:
            # simple linear interpolation based on frame ID
            # find prev valid
            prev_valid = None
            for j in range(i-1, -1, -1):
                if opt_path[j].x is not None and opt_path[j].w is not None:
                    prev_valid = opt_path[j]
                    break
            # find next valid
            next_valid = None
            for j in range(i+1, len(opt_path)):
                if opt_path[j].x is not None and opt_path[j].w is not None:
                    next_valid = opt_path[j]
                    break
                    
            if prev_valid and next_valid:
                alpha = (i - prev_valid.frame_id) / (next_valid.frame_id - prev_valid.frame_id)
                opt_path[i].x = prev_valid.x + alpha * (next_valid.x - prev_valid.x)
                opt_path[i].y = prev_valid.y + alpha * (next_valid.y - prev_valid.y)
                opt_path[i].w = prev_valid.w + alpha * (next_valid.w - prev_valid.w)
                opt_path[i].h = prev_valid.h + alpha * (next_valid.h - prev_valid.h)
            elif prev_valid:
                opt_path[i].x = prev_valid.x
                opt_path[i].y = prev_valid.y
                opt_path[i].w = prev_valid.w
                opt_path[i].h = prev_valid.h
            elif next_valid:
                opt_path[i].x = next_valid.x
                opt_path[i].y = next_valid.y
                opt_path[i].w = next_valid.w
                opt_path[i].h = next_valid.h

    # Export trajectory
    trajectory_out = []
    for node in opt_path:
        trajectory_out.append({
            "frame_id": detections[node.frame_id]["frame_id"],
            "x": float(node.x) if node.x is not None else -1,
            "y": float(node.y) if node.y is not None else -1,
            "w": float(node.w) if node.w is not None else -1,
            "h": float(node.h) if node.h is not None else -1,
            "is_dummy": node.is_dummy,
            "attached_player_id": node.attached_player_id
        })
        
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_json, 'w') as f:
        json.dump(trajectory_out, f, indent=4)
        
    print(f"Phase 3-4-5 Complete! Trajectory saved to {args.output_json}")

if __name__ == "__main__":
    build_graph_and_viterbi()
