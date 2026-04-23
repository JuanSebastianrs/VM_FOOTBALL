"""
TacticalVision AI: mAP Detection Dashboard (Premium Dark Theme)
Visualizes COCO mAP metrics from rfdetr_evaluation_full.json.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard_style import (
    setup_premium_style, stat_box, style_axis, suptitle,
    BG, CARD, BORDER, TEXT, TEXT_SEC, GRID,
    BLUE, BLUE_DIM, GREEN, PURPLE, ORANGE, RED, CYAN, YELLOW, PINK,
    PALETTE_BLUE, PALETTE_GREEN, PALETTE_PURPLE, PALETTE_ORANGE,
    gradient_bar_colors, make_gradient_cmap
)

OUTPUT_DIR = "results_final"
EVAL_DIR = "results_final/evaluation"


def load_map_data():
    """Load mAP results from rfdetr_evaluation_full.json."""
    json_path = os.path.join(EVAL_DIR, "rfdetr_evaluation_full.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] {json_path} not found. Run evaluate_models.py first.")
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_map_dashboard():
    setup_premium_style()
    data = load_map_data()
    if not data:
        return

    # Use best model (Best_EMA = Best_Total, both have same scores)
    best_name = max(data, key=lambda k: data[k]["mAP@0.5:0.95"])
    best = data[best_name]

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.88, bottom=0.08)

    suptitle(fig, "RF-DETR Detection Performance (COCO Evaluation)", y=0.96)

    # ── Panel 1: Main mAP Bar Chart ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['mAP@0.5:0.95', 'mAP@0.50', 'mAP@0.75']
    vals = [best[m] for m in metrics]
    labels = ['mAP\n@50:95', 'mAP\n@50', 'mAP\n@75']
    colors = gradient_bar_colors(3, PALETTE_BLUE)

    bars = ax1.bar(labels, vals, color=colors, edgecolor=[BLUE]*3,
                   linewidth=1.2, width=0.55, zorder=3)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=12,
                 fontweight='bold', color=TEXT)
        # Glow effect
        ax1.bar(bar.get_x() + bar.get_width()/2, v, width=bar.get_width()*1.15,
                color=BLUE, alpha=0.08, zorder=1)

    ax1.set_ylim(0, 1.0)
    style_axis(ax1, title="Core mAP Scores", ylabel="Average Precision")
    stat_box(ax1, f"Best: {best_name}\nmAP@50 = {best['mAP@0.50']:.4f}")

    # ── Panel 2: mAP by Object Size ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    size_metrics = ['mAP_S', 'mAP_M', 'mAP_L']
    size_vals = [best[m] for m in size_metrics]
    size_labels = ['Small\n(< 32²px)', 'Medium\n(32-96²px)', 'Large\n(> 96²px)']
    size_colors = [PURPLE, ORANGE, GREEN]

    bars = ax2.bar(size_labels, size_vals, color=size_colors, edgecolor='white',
                   linewidth=0.8, width=0.50, alpha=0.85, zorder=3)
    for bar, v, c in zip(bars, size_vals, size_colors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color=c)

    ax2.set_ylim(0, 0.8)
    style_axis(ax2, title="mAP by Object Size", ylabel="AP@0.5:0.95")
    stat_box(ax2, f"Small AP: {size_vals[0]:.4f}\nLarge AP: {size_vals[2]:.4f}")

    # ── Panel 3: Per-Class AP@0.50 (Radial / Horizontal) ─────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    per_class = best['per_class']
    cls_names = list(per_class.keys())
    cls_ap50 = [per_class[c]['AP@0.50'] for c in cls_names]
    cls_ap_main = [per_class[c]['AP@0.5:0.95'] for c in cls_names]
    cls_labels = [c.capitalize() for c in cls_names]

    y_pos = np.arange(len(cls_names))
    cls_colors = [BLUE, ORANGE, GREEN]

    # Draw horizontal grouped bars
    bar_h = 0.30
    b1 = ax3.barh(y_pos + bar_h/2, cls_ap50, height=bar_h, color=cls_colors,
                  alpha=0.9, edgecolor='white', linewidth=0.6, label='AP@0.50', zorder=3)
    b2 = ax3.barh(y_pos - bar_h/2, cls_ap_main, height=bar_h,
                  color=[c + '80' for c in ['#58A6FF','#F0883E','#3FB950']],
                  edgecolor='white', linewidth=0.6, label='AP@50:95', zorder=3)

    for bar, v in zip(b1, cls_ap50):
        ax3.text(v + 0.015, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', ha='left', va='center', fontsize=10,
                 fontweight='bold', color=TEXT)
    for bar, v in zip(b2, cls_ap_main):
        ax3.text(v + 0.015, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', ha='left', va='center', fontsize=10, color=TEXT_SEC)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(cls_labels, fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1.15)
    ax3.legend(loc='lower right', fontsize=9)
    style_axis(ax3, title="Per-Class Average Precision", xlabel="AP")
    ax3.invert_yaxis()

    # ── Panel 4: Average Recall Breakdown ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    recall_metrics = ['AR@1', 'AR@10', 'AR@100']
    recall_vals = [best[m] for m in recall_metrics]
    recall_labels = ['AR@1\n(1 det)', 'AR@10\n(10 det)', 'AR@100\n(100 det)']
    recall_colors = gradient_bar_colors(3, PALETTE_GREEN)

    bars = ax4.bar(recall_labels, recall_vals, color=recall_colors,
                   edgecolor=[GREEN]*3, linewidth=1.2, width=0.50, zorder=3)
    for bar, v in zip(bars, recall_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color=TEXT)

    ax4.set_ylim(0, 0.8)
    style_axis(ax4, title="Average Recall", ylabel="AR")
    stat_box(ax4, f"AR@100 = {recall_vals[2]:.4f}")

    # ── Panel 5: AR by Size ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ar_size = [best.get('AR_S', 0), best.get('AR_M', 0), best.get('AR_L', 0)]
    ar_labels = ['Small', 'Medium', 'Large']
    ar_colors = [PURPLE, ORANGE, GREEN]

    # Donut chart style
    wedges, texts = ax5.pie(
        ar_size, labels=None, colors=ar_colors,
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2)
    )

    # Center text
    ax5.text(0, 0, f"AR\nmean\n{np.mean(ar_size):.3f}",
             ha='center', va='center', fontsize=13, fontweight='bold',
             color=TEXT)

    # Legend
    legend_labels = [f'{l}: {v:.4f}' for l, v in zip(ar_labels, ar_size)]
    ax5.legend(wedges, legend_labels, loc='center left',
               bbox_to_anchor=(-0.3, 0.5), fontsize=10)
    ax5.set_title("Recall by Object Size", fontsize=14, fontweight='bold',
                  color=TEXT, pad=12)

    # ── Panel 6: Model Comparison Table ──────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Build comparison table
    table_data = []
    model_names = list(data.keys())
    for name in model_names:
        m = data[name]
        table_data.append([
            name.replace('Best_', ''),
            f"{m['mAP@0.5:0.95']:.4f}",
            f"{m['mAP@0.50']:.4f}",
            f"{m['mAP@0.75']:.4f}",
            f"{m['fps']:.1f}",
            f"{m['model_size_mb']:.0f} MB"
        ])

    col_labels = ['Model', 'mAP', 'mAP@50', 'mAP@75', 'FPS', 'Size']

    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor(BLUE_DIM)
            cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        else:
            cell.set_facecolor(CARD)
            cell.set_text_props(color=TEXT)
            # Highlight best model row
            if table_data[row-1][0] == best_name.replace('Best_', ''):
                cell.set_facecolor('#1a2a1a')

    ax6.set_title("Model Checkpoint Comparison", fontsize=14,
                  fontweight='bold', color=TEXT, pad=12, y=0.95)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "dashboard_detection_map.png")
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] mAP Dashboard saved to {out_path}")


if __name__ == "__main__":
    generate_map_dashboard()
