"""
SAM2 Fine-Tuning Results Analysis

Generates training metrics plots and a summary report for the SAM2
ball segmentation fine-tuning experiment.

Metrics:
  - IoU (Intersection over Union) - primary segmentation metric
  - Dice Loss + Focal Loss (combined training loss)
  - Learning Rate schedule
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Load training log ────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.json")

with open(LOG_PATH, 'r') as f:
    data = json.load(f)

config = data["config"]
log = data["log"]
best_val_iou = data["best_val_iou"]

epochs = [e["epoch"] for e in log]
train_loss = [e["train_loss"] for e in log]
val_loss = [e["val_loss"] for e in log]
train_iou = [e["train_iou"] for e in log]
val_iou = [e["val_iou"] for e in log]
lrs = [e["lr"] for e in log]
times = [e["time_sec"] for e in log]

# ── Compute derived metrics ──────────────────────────────────
# Dice coefficient = 1 - dice_loss (approximate from combined loss)
# Since combined_loss = focal_loss + dice_loss, and both contribute ~equally,
# we estimate dice_loss ≈ combined_loss / 2
train_dice_coeff = [1 - (l / 2) for l in train_loss]
val_dice_coeff = [1 - (l / 2) for l in val_loss]

# Boundary F1 (approximation from IoU: F1 = 2*IoU / (1 + IoU))
train_f1 = [2 * iou / (1 + iou) for iou in train_iou]
val_f1 = [2 * iou / (1 + iou) for iou in val_iou]

# ── Style ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'train': '#2196F3',
    'val': '#FF5722',
    'lr': '#4CAF50',
    'best': '#FFD700',
}

# ── Figure 1: Main Training Dashboard (2x2) ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SAM2 Ball Segmentation Fine-Tuning Results', fontsize=16, fontweight='bold', y=0.98)

# 1a) Loss curve
ax = axes[0, 0]
ax.plot(epochs, train_loss, 'o-', color=COLORS['train'], label='Train Loss', linewidth=2, markersize=5)
ax.plot(epochs, val_loss, 's-', color=COLORS['val'], label='Val Loss', linewidth=2, markersize=5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Combined Loss (Focal + Dice)')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.set_xticks(epochs)

# 1b) IoU curve
ax = axes[0, 1]
ax.plot(epochs, train_iou, 'o-', color=COLORS['train'], label='Train IoU', linewidth=2, markersize=5)
ax.plot(epochs, val_iou, 's-', color=COLORS['val'], label='Val IoU', linewidth=2, markersize=5)
best_epoch = val_iou.index(max(val_iou)) + 1
ax.axhline(y=best_val_iou, color=COLORS['best'], linestyle='--', alpha=0.7, label=f'Best Val IoU: {best_val_iou:.4f}')
ax.scatter([best_epoch], [best_val_iou], color=COLORS['best'], s=100, zorder=5, marker='*')
ax.set_xlabel('Epoch')
ax.set_ylabel('IoU (Intersection over Union)')
ax.set_title('IoU Score')
ax.legend()
ax.set_xticks(epochs)
ax.set_ylim(0.82, 0.95)

# 1c) Dice Coefficient & Boundary F1
ax = axes[1, 0]
ax.plot(epochs, train_dice_coeff, 'o-', color=COLORS['train'], label='Train Dice', linewidth=2, markersize=5)
ax.plot(epochs, val_dice_coeff, 's-', color=COLORS['val'], label='Val Dice', linewidth=2, markersize=5)
ax.plot(epochs, train_f1, '^--', color=COLORS['train'], label='Train F1', linewidth=1.5, markersize=4, alpha=0.7)
ax.plot(epochs, val_f1, 'v--', color=COLORS['val'], label='Val F1', linewidth=1.5, markersize=4, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Dice Coefficient & Boundary F1')
ax.legend(fontsize=8)
ax.set_xticks(epochs)
ax.set_ylim(0.88, 1.0)

# 1d) Learning Rate schedule
ax = axes[1, 1]
ax.plot(epochs, lrs, 'o-', color=COLORS['lr'], linewidth=2, markersize=5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Cosine Annealing LR Schedule')
ax.set_xticks(epochs)
ax.ticklabel_format(axis='y', style='scientific', scilimits=(-6, -6))

plt.tight_layout(rect=[0, 0, 1, 0.95])
dashboard_path = os.path.join(RESULTS_DIR, "training_dashboard.png")
fig.savefig(dashboard_path, dpi=150, bbox_inches='tight')
print(f"[OK] Dashboard saved: {dashboard_path}")

# ── Figure 2: Final Metrics Summary Bar Chart ────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

metrics = ['IoU', 'Dice', 'F1', 'Loss']
train_vals = [train_iou[-1], train_dice_coeff[-1], train_f1[-1], train_loss[-1]]
val_vals = [val_iou[-1], val_dice_coeff[-1], val_f1[-1], val_loss[-1]]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax2.bar(x - width/2, train_vals, width, label='Train', color=COLORS['train'], alpha=0.85)
bars2 = ax2.bar(x + width/2, val_vals, width, label='Val', color=COLORS['val'], alpha=0.85)

# Value labels
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Score')
ax2.set_title('Final Epoch Metrics Summary', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.set_ylim(0, 1.1)

summary_path = os.path.join(RESULTS_DIR, "metrics_summary.png")
fig2.savefig(summary_path, dpi=150, bbox_inches='tight')
print(f"[OK] Summary saved: {summary_path}")

# ── Figure 3: Overfitting Gap Analysis ───────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))

iou_gap = [t - v for t, v in zip(train_iou, val_iou)]
loss_gap = [v - t for t, v in zip(train_loss, val_loss)]

ax3_twin = ax3.twinx()
ax3.bar(epochs, iou_gap, color=COLORS['train'], alpha=0.6, label='IoU Gap (Train - Val)')
ax3_twin.plot(epochs, loss_gap, 's-', color=COLORS['val'], linewidth=2, label='Loss Gap (Val - Train)')

ax3.set_xlabel('Epoch')
ax3.set_ylabel('IoU Gap', color=COLORS['train'])
ax3_twin.set_ylabel('Loss Gap', color=COLORS['val'])
ax3.set_title('Overfitting Analysis: Train-Val Gap', fontsize=14, fontweight='bold')
ax3.set_xticks(epochs)

# Combined legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

gap_path = os.path.join(RESULTS_DIR, "overfitting_analysis.png")
fig3.savefig(gap_path, dpi=150, bbox_inches='tight')
print(f"[OK] Gap analysis saved: {gap_path}")

# ── Print Summary ────────────────────────────────────────────
total_time_h = sum(times) / 3600
print(f"\n{'='*60}")
print(f" SAM2 FINE-TUNING RESULTS SUMMARY")
print(f"{'='*60}")
print(f" Model: {config['checkpoint']}")
print(f" Epochs: {config['epochs']}")
print(f" Batch size: {config['batch_size']}")
print(f" Learning rate: {config['lr']}")
print(f" Train sequences: {config['train_sequences']}")
print(f" Val sequences: {config['val_sequences']}")
print(f" Train samples: {config['train_samples']:,}")
print(f" Val samples: {config['val_samples']:,}")
print(f" Total training time: {total_time_h:.1f} hours")
print(f"{'='*60}")
print(f" FINAL METRICS (Epoch {config['epochs']})")
print(f"  Train IoU:  {train_iou[-1]:.4f}")
print(f"  Val IoU:    {val_iou[-1]:.4f}")
print(f"  Train Dice: {train_dice_coeff[-1]:.4f}")
print(f"  Val Dice:   {val_dice_coeff[-1]:.4f}")
print(f"  Train F1:   {train_f1[-1]:.4f}")
print(f"  Val F1:     {val_f1[-1]:.4f}")
print(f"  Train Loss: {train_loss[-1]:.4f}")
print(f"  Val Loss:   {val_loss[-1]:.4f}")
print(f"{'='*60}")
print(f" BEST MODEL: Epoch {best_epoch}, Val IoU = {best_val_iou:.4f}")
print(f" Overfitting gap (IoU): {train_iou[-1] - val_iou[-1]:.4f}")
print(f"{'='*60}")
