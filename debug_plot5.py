import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

results_dir = r"d:\sebastian\Tesis\VM_FOOTBALL\tactical_results\batch_eval"
base_test_dir = r"d:\sebastian\Tesis\VM_FOOTBALL\data\tracking\SoccerNet\tracking\test\test"
master_seq = "SNMOT-116"
master_dir = os.path.join(base_test_dir, master_seq)

det_json = os.path.join(results_dir, f"{master_seq}_detections.json")
traj_json = os.path.join(results_dir, f"{master_seq}_trajectory.json")

with open(det_json, 'r') as f: dets = json.load(f)
with open(traj_json, 'r') as f: trajs = json.load(f)

# Extract GT
gt_x = {}
for lbl_path in sorted(glob.glob(os.path.join(master_dir, "labels", "*.txt"))):
    f_id = int(os.path.splitext(os.path.basename(lbl_path))[0])
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.split()
            if parts[0] == "5": gt_x[f_id] = float(parts[1])

# Extract YOLO noise
yolo_x, yolo_f = [], []
for d in dets:
    for c in d['ball_candidates']:
        yolo_f.append(d['frame_id'])
        yolo_x.append(c['x_center']/1920) 

# Extract Viterbi
vit_f = [t['frame_id'] for t in trajs]
vit_x = [t['x']/1920 for t in trajs]

plt.figure(figsize=(15, 6))
plt.plot(list(gt_x.keys()), list(gt_x.values()), color='black', lw=3, label='Ground Truth')
plt.scatter(yolo_f, yolo_x, color='red', s=5, alpha=0.2, label='YOLO Detections (Noise)')
plt.plot(vit_f, vit_x, color='#66b3ff', lw=2, label='Viterbi Trajectory')

for d in dets:
    if len(d['ball_candidates']) == 0:
        plt.axvspan(d['frame_id'], d['frame_id']+1, color='gray', alpha=0.1)

plt.title(f'Gráfica 5: Análisis Temporal de Trayectoria ({master_seq})')
plt.xlabel('Frame ID'), plt.ylabel('Normalized X Coordinate')
plt.legend(), plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, "plot5_temporal_masterpiece.png"), dpi=300)
print("Plot 5 generated successfully!")
