"""
TacticalVision AI: Paper-Quality Dashboard Generator (Premium Dark Theme)
Consolidates metrics from batch evaluation into four premium visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard_style import (
    setup_premium_style, stat_box, style_axis, mean_line, suptitle,
    BG, CARD, BORDER, TEXT, TEXT_SEC, GRID,
    BLUE, BLUE_DIM, GREEN, GREEN_DIM, PURPLE, PURPLE_DIM,
    ORANGE, ORANGE_DIM, RED, RED_DIM, CYAN, YELLOW, PINK
)

try:
    from src.evaluate_clustering import run_clustering_eval
except ImportError:
    run_clustering_eval = None

CSV_PATH = "results_final/evaluation/batch_evaluation.csv"
OUT_DIR = "results_final"


def _hist_with_kde(ax, data, color, dim_color, xlabel, n_bins=15):
    """Draw a styled histogram with KDE overlay."""
    ax.hist(data, bins=n_bins, color=color, alpha=0.5, edgecolor=color,
            linewidth=0.6, zorder=2)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_vals = kde(x_range)
        bin_width = (data.max() - data.min()) / n_bins
        ax.plot(x_range, kde_vals * len(data) * bin_width,
                color=color, linewidth=2.5, alpha=0.9, zorder=3,
                path_effects=[pe.withStroke(linewidth=4, foreground=dim_color + '60')])
    except Exception:
        pass
    ax.set_xlabel(xlabel, fontsize=11, color=TEXT_SEC)


def generate_ball_tracking_dashboard(df):
    """Dashboard 1: Ball F1, Precision, Recall distributions."""
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    suptitle(fig, "Ball Detection & Tracking Potential", y=0.97)

    # ── F1 Distribution (full width) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _hist_with_kde(ax1, df['Ball_F1'], ORANGE, ORANGE_DIM, "F1-Score")
    mean_line(ax1, df['Ball_F1'].mean(), label_fmt="Mean F1: {:.3f}", color=CYAN)
    style_axis(ax1, title="Ball F1-Score Distribution (Across 49 Sequences)",
               ylabel="Count")
    ax1.legend(loc='upper left', framealpha=0.8)
    stat_box(ax1, f"μ = {df['Ball_F1'].mean():.3f}\n"
                   f"σ = {df['Ball_F1'].std():.3f}\n"
                   f"med = {df['Ball_F1'].median():.3f}")

    # ── Precision vs Recall Scatter ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    # Glow
    ax2.scatter(df['Ball_Rec'], df['Ball_Prec'], c=df['Ball_F1'], cmap='magma',
                s=180, alpha=0.12, edgecolors='none', zorder=2)
    sc = ax2.scatter(df['Ball_Rec'], df['Ball_Prec'], c=df['Ball_F1'],
                     cmap='magma', s=70, alpha=0.9, edgecolors='white',
                     linewidth=0.4, zorder=3)
    cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("F1 Score", fontsize=10, color=TEXT_SEC)
    cbar.ax.tick_params(colors=TEXT_SEC)
    cbar.outline.set_edgecolor(BORDER)
    style_axis(ax2, title="Ball Precision vs. Recall",
               xlabel="Recall", ylabel="Precision")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    # ── FPPI Density ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _hist_with_kde(ax3, df['Ball_FPPI'], YELLOW, '#8B7500', "FPPI")
    mean_line(ax3, df['Ball_FPPI'].mean(), label_fmt="Mean FPPI: {:.3f}",
              color=CYAN)
    style_axis(ax3, title="False Positives per Image (FPPI)",
               ylabel="Count")
    ax3.legend(loc='upper right', framealpha=0.8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(os.path.join(OUT_DIR, "dashboard_ball_potential.png"),
                dpi=200, bbox_inches='tight')
    plt.close()


def generate_clustering_dashboard():
    """Dashboard 2: Team Clustering quality (Silhouette & Davies-Bouldin)."""
    clustering_json = "results_final/evaluation/clustering_metrics.json"
    
    if os.path.exists(clustering_json):
        import json
        with open(clustering_json, 'r') as f:
            cdata = json.load(f)
        seq_names = cdata["sequences"]
        sil_scores = cdata["silhouette_scores"]
        db_scores = cdata["davies_bouldin_scores"]
        print(f"[OK] Clustering data loaded from {clustering_json}")
    elif run_clustering_eval is not None:
        print("Running clustering evaluation for dashboard data...")
        seq_names, sil_scores, db_scores = run_clustering_eval()
    else:
        print("[Warn] No clustering data available. Skipping.")
        return

    if not sil_scores:
        print("[Error] No clustering scores obtained.")
        return

    sil_arr = np.array(sil_scores)
    db_arr = np.array(db_scores)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    suptitle(fig, "Team Semantic Clustering Stability (HSV Descriptor)", y=0.97)

    # ── Silhouette ───────────────────────────────────────────────────────
    _hist_with_kde(axes[0], sil_arr, PURPLE, PURPLE_DIM,
                   "Score (Interval [-1, 1])")
    mean_line(axes[0], sil_arr.mean(), label_fmt="Mean: {:.3f}", color=CYAN)
    style_axis(axes[0], title="Silhouette Score (Cohesion & Separation)",
               ylabel="Density")
    axes[0].legend(loc='upper right', framealpha=0.8)
    stat_box(axes[0], f"μ = {sil_arr.mean():.3f}\nσ = {sil_arr.std():.3f}")

    # ── Davies-Bouldin ───────────────────────────────────────────────────
    _hist_with_kde(axes[1], db_arr, PINK, '#A0507A',
                   "Index (Lower = Better)")
    mean_line(axes[1], db_arr.mean(), label_fmt="Mean: {:.3f}", color=CYAN)
    style_axis(axes[1], title="Davies-Bouldin Index (Cluster Overlap)",
               ylabel="Density")
    axes[1].legend(loc='upper right', framealpha=0.8)
    stat_box(axes[1], f"μ = {db_arr.mean():.3f}\nσ = {db_arr.std():.3f}")

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, "dashboard_clustering_metrics.png"),
                dpi=200, bbox_inches='tight')
    plt.close()


def generate_mota_players_dashboard(df):
    """Dashboard 3: MOTA and IDF1 violin + swarm plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    suptitle(fig, "Player Tracking Robustness Analysis", y=0.97)

    for ax_idx, (col, color, dim, title) in enumerate([
        ('MOTA', GREEN, GREEN_DIM, "Active Player MOTA Distribution"),
        ('IDF1', BLUE, BLUE_DIM, "Identity Preservation (IDF1)")
    ]):
        ax = axes[ax_idx]
        data = df[col].dropna()

        # Violin
        parts = ax.violinplot(data, positions=[0], showmeans=False,
                              showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.5)

        # Strip/swarm overlay
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(jitter, data, color=color, alpha=0.5, s=30,
                   edgecolors='white', linewidth=0.3, zorder=3)

        # Mean line
        ax.axhline(data.mean(), color=CYAN, ls='--', lw=2, alpha=0.9,
                   label=f"Mean: {data.mean():.3f}", zorder=4)

        # Quartile lines
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        ax.axhline(q1, color=TEXT_SEC, ls=':', lw=1, alpha=0.5, zorder=4)
        ax.axhline(q3, color=TEXT_SEC, ls=':', lw=1, alpha=0.5, zorder=4)

        style_axis(ax, title=title, ylabel=f"{col} Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticks([])
        ax.legend(loc='upper right', framealpha=0.8)
        stat_box(ax, f"μ = {data.mean():.3f}\nQ1 = {q1:.3f}\nQ3 = {q3:.3f}",
                 x=0.03, ha='left')

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, "dashboard_mota_players.png"),
                dpi=200, bbox_inches='tight')
    plt.close()


def generate_rmse_dashboard(df):
    """Dashboard 4: RMSE Distribution with statistical annotations."""
    fig, ax = plt.subplots(figsize=(16, 7))
    suptitle(fig, "Ball Localization Precision (RMSE Distribution)", y=0.97)

    _hist_with_kde(ax, df['Ball_RMSE'], RED, RED_DIM, "Root Mean Square Error (Pixels)",
                   n_bins=20)
    mean_line(ax, df['Ball_RMSE'].mean(),
              label_fmt="Mean RMSE: {:.1f}px", color=CYAN)
    # Median line
    ax.axvline(df['Ball_RMSE'].median(), color=YELLOW, ls=':', lw=1.8,
               alpha=0.9, zorder=5,
               label=f"Median RMSE: {df['Ball_RMSE'].median():.1f}px")

    style_axis(ax, ylabel="Frequency")
    ax.legend(loc='upper right', framealpha=0.8)
    stat_box(ax,
             f"Min: {df['Ball_RMSE'].min():.1f}px\n"
             f"Max: {df['Ball_RMSE'].max():.1f}px\n"
             f"Std: {df['Ball_RMSE'].std():.1f}px\n"
             f"Med: {df['Ball_RMSE'].median():.1f}px")

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, "dashboard_rmse_distribution.png"),
                dpi=200, bbox_inches='tight')
    plt.close()


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[Error] CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    setup_premium_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating Ball Tracking Dashboard...")
    generate_ball_tracking_dashboard(df)

    print("Generating Clustering Dashboard...")
    generate_clustering_dashboard()

    print("Generating Player Tracking Dashboard...")
    generate_mota_players_dashboard(df)

    print("Generating RMSE Dashboard...")
    generate_rmse_dashboard(df)

    print("\n[SUCCESS] All dashboards generated in results_final/")


if __name__ == "__main__":
    main()
