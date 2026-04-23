"""
TacticalVision AI: Detailed MOT Performance Dashboard (Premium Dark Theme)
Focuses on tracking stability: MOTA, IDF1, MOTP, and ID Switches.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard_style import (
    setup_premium_style, stat_box, style_axis, mean_line, suptitle,
    gradient_bar_colors,
    BG, CARD, BORDER, TEXT, TEXT_SEC, GRID,
    BLUE, BLUE_DIM, GREEN, GREEN_DIM, ORANGE, ORANGE_DIM,
    RED, RED_DIM, CYAN, YELLOW, PURPLE,
    PALETTE_BLUE, PALETTE_GREEN, PALETTE_ORANGE
)


def get_args():
    parser = argparse.ArgumentParser("Generate MOT Detailed Dashboard")
    parser.add_argument("--csv_file", type=str,
                        default="results_final/evaluation/batch_evaluation.csv")
    parser.add_argument("--out_dir", type=str,
                        default="results_final/evaluation")
    return parser.parse_args()


def premium_hist(ax, data, color, dim_color, title, xlabel, mean_color=CYAN):
    """Styled histogram with KDE and mean line."""
    ax.hist(data, bins=15, color=color, alpha=0.55, edgecolor=color,
            linewidth=0.6, zorder=2)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data.dropna())
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_vals = kde(x_range)
        bin_width = (data.max() - data.min()) / 15
        ax.plot(x_range, kde_vals * len(data) * bin_width,
                color=color, linewidth=2.5, alpha=0.9, zorder=3,
                path_effects=[pe.withStroke(linewidth=4, foreground=dim_color + '60')])
    except Exception:
        pass
    mean_line(ax, data.mean(), color=mean_color)
    style_axis(ax, title=title, xlabel=xlabel, ylabel="Count")
    ax.legend(loc='upper left', framealpha=0.8)


def main():
    args = get_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return

    df = pd.read_csv(args.csv_file)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_active = df[df['Tracks_Eval'] > 0].copy()

    setup_premium_style()
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    suptitle(fig, "TacticalVision AI · Detailed MOT Performance", y=0.98)

    # ── 1. MOTA Distribution ────────────────────────────────────────────
    premium_hist(axes[0, 0], df_active['MOTA'], GREEN, GREEN_DIM,
                 "MOTA Distribution (Accuracy)", "MOTA")
    stat_box(axes[0, 0],
             f"μ = {df_active['MOTA'].mean():.3f}\n"
             f"σ = {df_active['MOTA'].std():.3f}\n"
             f"min = {df_active['MOTA'].min():.3f}")

    # ── 2. IDF1 Distribution ────────────────────────────────────────────
    premium_hist(axes[0, 1], df_active['IDF1'], BLUE, BLUE_DIM,
                 "IDF1 Distribution (Long-term Identity)", "IDF1")
    stat_box(axes[0, 1],
             f"μ = {df_active['IDF1'].mean():.3f}\n"
             f"σ = {df_active['IDF1'].std():.3f}")

    # ── 3. MOTP Distribution ────────────────────────────────────────────
    premium_hist(axes[1, 0], df_active['MOTP'], ORANGE, ORANGE_DIM,
                 "MOTP Distribution (Lower = Better)", "MOTP",
                 mean_color=YELLOW)
    stat_box(axes[1, 0],
             f"μ = {df_active['MOTP'].mean():.3f}\n"
             f"best = {df_active['MOTP'].min():.3f}")

    # ── 4. ID Switches (Log) ────────────────────────────────────────────
    df_active['log_idsw'] = np.log1p(df_active['ID_Switches'])
    premium_hist(axes[1, 1], df_active['log_idsw'], RED, RED_DIM,
                 "ID Switches Distribution (Log Scale)", "log(1 + ID_Switches)",
                 mean_color=YELLOW)
    stat_box(axes[1, 1],
             f"Total IDSW = {int(df_active['ID_Switches'].sum()):,}\n"
             f"μ per seq = {df_active['ID_Switches'].mean():.0f}")

    # ── 5. Top 10 by IDF1 ───────────────────────────────────────────────
    top_idf1 = df_active.nlargest(10, 'IDF1')
    colors_idf1 = gradient_bar_colors(10, PALETTE_BLUE)
    axes[2, 0].barh(range(10), top_idf1['IDF1'].values, height=0.65,
                    color=colors_idf1, edgecolor='white', linewidth=0.3,
                    alpha=0.9, zorder=3)
    for i, (val, c) in enumerate(zip(top_idf1['IDF1'].values, colors_idf1)):
        axes[2, 0].barh(i, val, height=0.85, color=c, alpha=0.1, zorder=1)
        axes[2, 0].text(val + 0.01, i, f"{val:.3f}", va='center',
                        fontsize=10, color=BLUE, fontweight='bold')
    axes[2, 0].set_yticks(range(10))
    axes[2, 0].set_yticklabels(top_idf1['Sequence'].values, fontsize=10)
    axes[2, 0].set_xlim(0, 1.08)
    axes[2, 0].invert_yaxis()
    style_axis(axes[2, 0], title="Top 10: Identity Stability (IDF1)",
               xlabel="IDF1")

    # ── 6. MOTA vs IDF1 Correlation ─────────────────────────────────────
    ax6 = axes[2, 1]
    sizes = 30 + df_active['ID_Switches'] * 0.5
    # Glow
    ax6.scatter(df_active['MOTA'], df_active['IDF1'],
                c=df_active['ID_Switches'], cmap='YlOrRd',
                s=sizes * 2, alpha=0.12, edgecolors='none', zorder=2)
    # Core
    sc = ax6.scatter(df_active['MOTA'], df_active['IDF1'],
                     c=df_active['ID_Switches'], cmap='YlOrRd',
                     s=sizes, alpha=0.9, edgecolors='white',
                     linewidth=0.4, zorder=3)
    cbar = plt.colorbar(sc, ax=ax6, pad=0.02)
    cbar.set_label("ID Switches", fontsize=10, color=TEXT_SEC)
    cbar.ax.tick_params(colors=TEXT_SEC)
    cbar.outline.set_edgecolor(BORDER)
    style_axis(ax6, title="Tracking Correlation: MOTA vs IDF1 (Size=IDSW)",
               xlabel="MOTA", ylabel="IDF1")

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plot_path = os.path.join(args.out_dir, "mot_detailed_dashboard.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Detailed MOT Dashboard saved to: {plot_path}")


if __name__ == "__main__":
    main()
