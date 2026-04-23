"""
TacticalVision AI: Master Tracking Metrics Dashboard (v2)
Shows official TrackEval results + SoccerNet benchmark comparison.
Reads from tracking_global_metrics.json.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard_style import (
    setup_premium_style, stat_box, style_axis, suptitle,
    BG, CARD, BORDER, TEXT, TEXT_SEC, GRID,
    BLUE, BLUE_DIM, GREEN, GREEN_DIM, PURPLE, PURPLE_DIM,
    ORANGE, ORANGE_DIM, RED, RED_DIM, CYAN, YELLOW, PINK,
    gradient_bar_colors
)

METRICS_JSON = "results_final/evaluation/tracking_global_metrics.json"
OUTPUT_PATH  = "results_final/dashboard_tracking_metrics.png"

# ── SoccerNet Tracking 2023 Challenge Benchmark Data ─────────────────────────
# From the official SoccerNet Tracking challenge (CVPR 2023)
# Source: https://github.com/SoccerNet/sn-tracking + EvalAI leaderboard
SOCCERNET_BENCHMARK = {
    "MOT4MOT (1st)":  {"HOTA": 66.27, "DetA": 70.32, "AssA": 62.62, "MOTA": 80.1, "IDF1": 73.5},
    "ByteTrack":      {"HOTA": 55.80, "DetA": 60.10, "AssA": 52.50, "MOTA": 72.0, "IDF1": 64.0},
    "DeepSORT":       {"HOTA": 45.20, "DetA": 55.30, "AssA": 37.60, "MOTA": 62.0, "IDF1": 50.0},
}


def load_metrics():
    """Load official TrackEval metrics from JSON."""
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, 'r') as f:
            data = json.load(f)
        print(f"[OK] Métricas cargadas de {METRICS_JSON}")
        return data
    print(f"[ERROR] {METRICS_JSON} not found. Run evaluate_tracking_official.py first.")
    return None


def plot_tracking_metrics():
    setup_premium_style()
    metrics = load_metrics()
    if not metrics:
        return

    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30,
                          left=0.05, right=0.97, top=0.88, bottom=0.06)
    suptitle(fig, "TacticalVision AI · Official Tracking Evaluation (TrackEval)", y=0.96)

    # ── Panel 1: Radar Chart (Mean HOTA family) ──────────────────────────
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    ax_radar.set_facecolor(CARD)

    radar_labels = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
    radar_values = [metrics.get(k, 0) for k in radar_labels]
    radar_desc = ["HOTA\n(mean)", "DetA\n(mean)", "AssA\n(mean)", "MOTA", "IDF1"]
    radar_colors = [PURPLE, GREEN, ORANGE, BLUE, CYAN]

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    values_closed = radar_values + radar_values[:1]
    angles_closed = angles + angles[:1]

    ax_radar.fill(angles_closed, values_closed, alpha=0.15, color=CYAN)
    ax_radar.plot(angles_closed, values_closed, color=CYAN, linewidth=2.5,
                  marker='o', markersize=8, markerfacecolor=CYAN,
                  markeredgecolor='white', markeredgewidth=1.5,
                  path_effects=[pe.withStroke(linewidth=4, foreground=CYAN + '40')])

    for angle, val, color in zip(angles, radar_values, radar_colors):
        ax_radar.text(angle, val + 7, f"{val:.1f}%", ha='center', va='center',
                      fontsize=11, fontweight='bold', color=color,
                      path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

    ax_radar.set_thetagrids(np.degrees(angles), radar_desc, fontsize=9, color=TEXT_SEC)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_yticks([20, 40, 60, 80, 100])
    ax_radar.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color=TEXT_SEC)
    ax_radar.spines['polar'].set_color(BORDER)
    ax_radar.grid(color=GRID, alpha=0.4, linewidth=0.5)
    ax_radar.set_title("Core Metrics Overview", fontsize=13, fontweight='bold',
                       color=TEXT, pad=20)

    # ── Panel 2: HOTA@IoU Breakdown ──────────────────────────────────────
    ax_iou = fig.add_subplot(gs[0, 1])
    
    # Show mean vs IoU@50 for HOTA, DetA, AssA
    iou_labels = ["HOTA", "DetA", "AssA"]
    mean_vals = [metrics.get(k, 0) for k in iou_labels]
    at50_vals = [metrics.get(f"{k}@50", 0) for k in iou_labels]
    
    x = np.arange(len(iou_labels))
    w = 0.32
    
    bars_mean = ax_iou.bar(x - w/2, mean_vals, w, label='Mean (all IoU)',
                           color=[PURPLE_DIM, GREEN_DIM, ORANGE_DIM],
                           edgecolor='white', linewidth=0.8, zorder=3)
    bars_50 = ax_iou.bar(x + w/2, at50_vals, w, label='@IoU=0.50',
                         color=[PURPLE, GREEN, ORANGE],
                         edgecolor='white', linewidth=0.8, zorder=3)
    
    for bars, vals in [(bars_mean, mean_vals), (bars_50, at50_vals)]:
        for bar, v in zip(bars, vals):
            ax_iou.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color=TEXT)
    
    ax_iou.set_xticks(x)
    ax_iou.set_xticklabels(iou_labels, fontsize=12, fontweight='bold')
    ax_iou.set_ylim(0, 105)
    ax_iou.legend(loc='upper right', fontsize=9)
    style_axis(ax_iou, title="HOTA Family: Mean vs @IoU=0.50", ylabel="Score (%)")

    # ── Panel 3: Detailed Stats ──────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    
    # Key stats card
    stats_text = [
        ("MOTA", f"{metrics.get('MOTA', 0):.2f}%", BLUE),
        ("MOTP", f"{metrics.get('MOTP', 0):.2f}%", GREEN),
        ("IDF1", f"{metrics.get('IDF1', 0):.2f}%", CYAN),
        ("IDP / IDR", f"{metrics.get('IDP', 0):.1f}% / {metrics.get('IDR', 0):.1f}%", ORANGE),
        ("ID Switches", f"{metrics.get('IDSW', 0):,}", RED),
        ("True Positives", f"{metrics.get('CLR_TP', 0):,}", GREEN),
        ("False Positives", f"{metrics.get('CLR_FP', 0):,}", YELLOW),
        ("False Negatives", f"{metrics.get('CLR_FN', 0):,}", ORANGE),
    ]
    
    y_start = 0.92
    for i, (label, value, color) in enumerate(stats_text):
        y = y_start - i * 0.115
        # Background bar
        ax_stats.add_patch(FancyBboxPatch(
            (0.02, y - 0.04), 0.96, 0.09,
            boxstyle="round,pad=0.02", facecolor=CARD,
            edgecolor=BORDER, linewidth=0.8, transform=ax_stats.transAxes
        ))
        # Color indicator
        ax_stats.add_patch(FancyBboxPatch(
            (0.02, y - 0.04), 0.015, 0.09,
            boxstyle="round,pad=0.001", facecolor=color, alpha=0.8,
            transform=ax_stats.transAxes
        ))
        ax_stats.text(0.08, y, label, transform=ax_stats.transAxes,
                      fontsize=11, color=TEXT_SEC, va='center')
        ax_stats.text(0.95, y, value, transform=ax_stats.transAxes,
                      fontsize=13, fontweight='bold', color=color,
                      ha='right', va='center')
    
    ax_stats.set_title("Detailed Statistics", fontsize=13, fontweight='bold',
                       color=TEXT, pad=12)

    # ── Panel 4-5-6 (bottom): SoccerNet Comparison ──────────────────────
    ax_cmp = fig.add_subplot(gs[1, :])
    
    # Data setup
    cmp_metrics = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
    our_vals = [metrics.get(k, 0) for k in cmp_metrics]
    
    all_methods = {}
    all_methods["TacticalVision (Ours)"] = our_vals
    for name, bench in SOCCERNET_BENCHMARK.items():
        all_methods[name] = [bench.get(k, 0) for k in cmp_metrics]
    
    method_names = list(all_methods.keys())
    n_methods = len(method_names)
    n_metrics = len(cmp_metrics)
    
    x = np.arange(n_metrics)
    total_width = 0.75
    bar_width = total_width / n_methods
    
    method_colors = [CYAN, PURPLE, BLUE, ORANGE]
    method_alphas = [1.0, 0.7, 0.6, 0.5]
    
    for i, (name, vals) in enumerate(all_methods.items()):
        offset = (i - n_methods/2 + 0.5) * bar_width
        bars = ax_cmp.bar(x + offset, vals, bar_width, label=name,
                         color=method_colors[i], alpha=method_alphas[i],
                         edgecolor='white', linewidth=0.6, zorder=3)
        
        # Value labels for "ours"
        if i == 0:
            for bar, v in zip(bars, vals):
                ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{v:.1f}', ha='center', va='bottom', fontsize=9,
                            fontweight='bold', color=CYAN)
    
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(cmp_metrics, fontsize=13, fontweight='bold')
    ax_cmp.set_ylim(0, 100)
    ax_cmp.legend(loc='upper right', fontsize=10, ncol=2)
    style_axis(ax_cmp, title="SoccerNet Tracking Benchmark Comparison (CVPR 2023)",
               ylabel="Score (%)")
    
    # Add note
    ax_cmp.text(0.01, 0.02,
                "Note: TacticalVision uses RF-DETR trained on SoccerNet GT. "
                "Benchmark methods use their own detectors. Direct comparison "
                "is illustrative, not apples-to-apples.",
                transform=ax_cmp.transAxes, fontsize=8, color=TEXT_SEC,
                style='italic', alpha=0.7)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Dashboard guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_tracking_metrics()
