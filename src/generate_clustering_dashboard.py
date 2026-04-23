"""
TacticalVision AI: Clustering Quality Dashboard (Premium Dark Theme)
Visualizes Silhouette Score and Davies-Bouldin Index from clustering_metrics.json.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard_style import (
    setup_premium_style, stat_box, style_axis, suptitle, mean_line,
    BG, CARD, BORDER, TEXT, TEXT_SEC, GRID,
    BLUE, GREEN, GREEN_DIM, PURPLE, PURPLE_DIM, ORANGE, RED, CYAN, YELLOW,
    PALETTE_PURPLE, PALETTE_GREEN, gradient_bar_colors
)

CLUSTERING_JSON = "results_final/evaluation/clustering_metrics.json"
OUTPUT_PATH = "results_final/dashboard_clustering.png"


def load_clustering_data():
    """Load clustering results from JSON (pre-computed by evaluate_clustering.py)."""
    if os.path.exists(CLUSTERING_JSON):
        with open(CLUSTERING_JSON, 'r') as f:
            data = json.load(f)
        return data["sequences"], data["silhouette_scores"], data["davies_bouldin_scores"]
    
    print(f"[WARN] {CLUSTERING_JSON} not found. Attempting live computation...")
    try:
        from src.evaluate_clustering import run_clustering_eval
        return run_clustering_eval()
    except Exception as e:
        print(f"[ERROR] Could not compute clustering: {e}")
        return None, None, None


def generate_clustering_dashboard():
    setup_premium_style()
    seq_names, sil_scores, db_scores = load_clustering_data()
    
    if seq_names is None or len(seq_names) == 0:
        print("[ERROR] No clustering data available. Skipping.")
        return

    sil = np.array(sil_scores)
    db = np.array(db_scores)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28,
                          left=0.06, right=0.96, top=0.88, bottom=0.08)
    suptitle(fig, "Team Clustering Quality (Silhouette & Davies-Bouldin Index)", y=0.96)

    # ── Panel 1: Silhouette Distribution ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_bins = 15
    counts, bins, patches = ax1.hist(sil, bins=n_bins, color=PURPLE, alpha=0.7,
                                      edgecolor=BG, linewidth=1.2, zorder=3)
    
    # Gradient coloring by bin
    colors = gradient_bar_colors(len(patches), PALETTE_PURPLE)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    # KDE overlay
    if len(sil) > 3:
        kde = gaussian_kde(sil, bw_method=0.3)
        x_range = np.linspace(sil.min() - 0.05, sil.max() + 0.05, 200)
        kde_vals = kde(x_range) * len(sil) * (bins[1] - bins[0])
        ax1.plot(x_range, kde_vals, color=PURPLE, linewidth=2.5, alpha=0.9, zorder=4)
    
    mean_line(ax1, sil.mean(), label_fmt="Mean: {:.3f}", color=CYAN)
    ax1.legend(fontsize=9)
    style_axis(ax1, title="Silhouette Score Distribution",
               xlabel="Silhouette Score (closer to 1 = better)", ylabel="Count")
    stat_box(ax1, f"μ = {sil.mean():.3f}\nσ = {sil.std():.3f}\nmed = {np.median(sil):.3f}")

    # ── Panel 2: Davies-Bouldin Distribution ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    
    counts2, bins2, patches2 = ax2.hist(db, bins=n_bins, color=GREEN, alpha=0.7,
                                         edgecolor=BG, linewidth=1.2, zorder=3)
    
    colors2 = gradient_bar_colors(len(patches2), PALETTE_GREEN)
    for patch, color in zip(patches2, colors2):
        patch.set_facecolor(color)
    
    if len(db) > 3:
        kde2 = gaussian_kde(db, bw_method=0.3)
        x_range2 = np.linspace(db.min() - 0.05, db.max() + 0.05, 200)
        kde_vals2 = kde2(x_range2) * len(db) * (bins2[1] - bins2[0])
        ax2.plot(x_range2, kde_vals2, color=GREEN, linewidth=2.5, alpha=0.9, zorder=4)
    
    mean_line(ax2, db.mean(), label_fmt="Mean: {:.3f}", color=CYAN)
    ax2.legend(fontsize=9)
    style_axis(ax2, title="Davies-Bouldin Index Distribution",
               xlabel="DBI (closer to 0 = better)", ylabel="Count")
    stat_box(ax2, f"μ = {db.mean():.3f}\nσ = {db.std():.3f}\nmed = {np.median(db):.3f}")

    # ── Panel 3: Per-Sequence Silhouette (sorted bar) ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Sort by silhouette score
    sort_idx = np.argsort(sil)[::-1]
    sorted_sil = sil[sort_idx]
    sorted_names = [seq_names[i].replace("SNMOT-", "") for i in sort_idx]
    
    bar_colors = gradient_bar_colors(len(sorted_sil), PALETTE_PURPLE)
    ax3.bar(range(len(sorted_sil)), sorted_sil, color=bar_colors,
            edgecolor=BG, linewidth=0.5, zorder=3)
    ax3.axhline(sil.mean(), color=CYAN, ls='--', lw=1.5, alpha=0.8, zorder=4)
    
    # Only show every 5th label to avoid crowding
    ax3.set_xticks(range(0, len(sorted_names), 5))
    ax3.set_xticklabels([sorted_names[i] for i in range(0, len(sorted_names), 5)],
                        rotation=45, fontsize=8)
    ax3.set_ylim(0, max(sorted_sil) * 1.15)
    style_axis(ax3, title="Silhouette per Sequence (Sorted)",
               xlabel="Sequence", ylabel="Silhouette")

    # ── Panel 4: Silhouette vs Davies-Bouldin Scatter ────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    
    scatter = ax4.scatter(sil, db, c=sil, cmap='magma', s=80, alpha=0.85,
                          edgecolors='white', linewidth=0.6, zorder=3)
    # Glow
    ax4.scatter(sil, db, c=sil, cmap='magma', s=200, alpha=0.15, zorder=2)
    
    plt.colorbar(scatter, ax=ax4, label='Silhouette Score', pad=0.02)
    
    # Reference lines
    ax4.axvline(sil.mean(), color=PURPLE, ls='--', lw=1.2, alpha=0.6)
    ax4.axhline(db.mean(), color=GREEN, ls='--', lw=1.2, alpha=0.6)
    
    # Fit trendline
    if len(sil) > 2:
        z = np.polyfit(sil, db, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(sil.min(), sil.max(), 100)
        ax4.plot(x_fit, p(x_fit), color=RED, ls='--', lw=1.5, alpha=0.7,
                 label=f'Trend (r={np.corrcoef(sil, db)[0,1]:.2f})')
    
    ax4.legend(loc='upper right', fontsize=9)
    style_axis(ax4, title="Silhouette vs Davies-Bouldin",
               xlabel="Silhouette Score ↑", ylabel="Davies-Bouldin Index ↓")
    stat_box(ax4, f"r = {np.corrcoef(sil, db)[0,1]:.3f}\n(Strong negative)")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Clustering Dashboard saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_clustering_dashboard()
