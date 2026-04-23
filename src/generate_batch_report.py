"""
TacticalVision AI: Batch Evaluation Dashboard (Premium Dark Theme)
Reads the CSV from batch_evaluator.py and produces:
1. A 4x2 panel dashboard PNG
2. A top5_winners.txt file
3. An enriched CSV with Composite_Score
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dashboard_style import (
    setup_premium_style, stat_box, style_axis, mean_line, suptitle,
    gradient_bar_colors,
    BG, CARD, BORDER, TEXT, TEXT_SEC, GRID,
    BLUE, BLUE_DIM, GREEN, GREEN_DIM, PURPLE, PURPLE_DIM,
    ORANGE, ORANGE_DIM, RED, CYAN, YELLOW, PINK,
    PALETTE_BLUE, PALETTE_GREEN, PALETTE_PURPLE, PALETTE_ORANGE
)


def get_args():
    parser = argparse.ArgumentParser("Generate Batch Dashboard")
    parser.add_argument("--csv_file", type=str,
                        default="results_final/evaluation/batch_evaluation.csv")
    parser.add_argument("--out_dir", type=str,
                        default="results_final/evaluation")
    return parser.parse_args()


def premium_hist(ax, data, color, dim_color, title, xlabel, mean_color=CYAN):
    """Styled histogram with KDE and mean line."""
    ax.hist(data, bins=15, color=color, alpha=0.55, edgecolor=color,
            linewidth=0.6, zorder=2)
    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data.dropna())
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_vals = kde(x_range)
        # Scale KDE to match histogram counts
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

    # ── Composite Score ──────────────────────────────────────────────────
    rmse_max = df['Ball_RMSE'].max()
    inv_rmse = (1.0 - df['Ball_RMSE'] / rmse_max) if rmse_max > 0 else 0.0
    df['Composite_Score'] = (
        0.30 * df['Ball_F1'] +
        0.25 * df['MOTA'] +
        0.20 * df['Team_Clustering_Acc'] +
        0.15 * inv_rmse +
        0.10 * df['IDF1']
    )
    df = df.sort_values(by='Composite_Score', ascending=False).reset_index(drop=True)

    # Save enriched CSV
    enriched_csv = os.path.join(args.out_dir, "batch_evaluation_enriched.csv")
    df.to_csv(enriched_csv, index=False)

    # ── Summary Statistics ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    cols = ['Ball_F1', 'Ball_RMSE', 'MOTA', 'IDF1',
            'Team_Clustering_Acc', 'Ref_Rec', 'GK_Rec']
    for col in cols:
        if col in df.columns:
            vals = df[col]
            print(f"  {col:25s}: mean={vals.mean():.3f}  "
                  f"median={vals.median():.3f}  "
                  f"std={vals.std():.3f}  "
                  f"min={vals.min():.3f}  max={vals.max():.3f}")
    print("=" * 60)

    # ── Top 5 Winners ────────────────────────────────────────────────────
    winners_file = os.path.join(args.out_dir, "top5_winners.txt")
    with open(winners_file, 'w') as f:
        f.write("TOP 5 WINNING SEQUENCES\n")
        f.write("Ranked by composite score: "
                "0.30*Ball_F1 + 0.25*MOTA + 0.20*Clustering + "
                "0.15*InvRMSE + 0.10*IDF1\n\n")
        for rank, (_, row) in enumerate(df.head(5).iterrows(), 1):
            f.write(f"#{rank}: {row['Sequence']} "
                    f"(Composite: {row['Composite_Score']:.3f})\n")
            f.write(f"   Ball:    F1={row['Ball_F1']:.3f}  "
                    f"RMSE={row['Ball_RMSE']:.1f}px  "
                    f"Prec@50={row.get('Ball_Prec_IoU50',0):.3f}  "
                    f"Rec@50={row.get('Ball_Rec_IoU50',0):.3f}\n")
            f.write(f"   Players: MOTA={row['MOTA']:.3f}  "
                    f"IDF1={row['IDF1']:.3f}  "
                    f"IDSW={int(row['ID_Switches'])}\n")
            f.write(f"   Teams:   Accuracy={row['Team_Clustering_Acc']:.3f}  "
                    f"Ref_Rec={row['Ref_Rec']:.3f}  "
                    f"GK_Rec={row['GK_Rec']:.3f}\n\n")
    print(f"Winners saved to {winners_file}")

    # ── 4x2 Dashboard ────────────────────────────────────────────────────
    setup_premium_style()
    fig, axes = plt.subplots(4, 2, figsize=(20, 28))
    suptitle(fig, "TacticalVision AI · Batch Evaluation Dashboard", y=0.98)

    # ── 1. Ball F1 Distribution ──────────────────────────────────────────
    premium_hist(axes[0, 0], df['Ball_F1'], BLUE, BLUE_DIM,
                 "Ball Tracking F1-Score Distribution", "F1-Score")
    stat_box(axes[0, 0],
             f"μ = {df['Ball_F1'].mean():.3f}\nσ = {df['Ball_F1'].std():.3f}")

    # ── 2. MOTA Distribution ────────────────────────────────────────────
    premium_hist(axes[0, 1], df['MOTA'], GREEN, GREEN_DIM,
                 "Player Tracking MOTA Distribution", "MOTA")
    stat_box(axes[0, 1],
             f"μ = {df['MOTA'].mean():.3f}\nσ = {df['MOTA'].std():.3f}")

    # ── 3. Top 10 by Ball F1 ────────────────────────────────────────────
    top_f1 = df.nlargest(10, 'Ball_F1')
    colors_f1 = gradient_bar_colors(10, PALETTE_BLUE)
    bars = axes[1, 0].barh(range(10), top_f1['Ball_F1'].values, height=0.65,
                           color=colors_f1, edgecolor='white', linewidth=0.3,
                           alpha=0.9, zorder=3)
    # Glow
    for i, (val, c) in enumerate(zip(top_f1['Ball_F1'].values, colors_f1)):
        axes[1, 0].barh(i, val, height=0.85, color=c, alpha=0.1, zorder=1)
        axes[1, 0].text(val + 0.01, i, f"{val:.3f}", va='center',
                        fontsize=10, color=BLUE, fontweight='bold')
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels(top_f1['Sequence'].values, fontsize=10)
    axes[1, 0].set_xlim(0, 1.08)
    axes[1, 0].invert_yaxis()
    style_axis(axes[1, 0], title="Top 10 Sequences: Ball F1", xlabel="Ball F1")

    # ── 4. Top 10 by MOTA ───────────────────────────────────────────────
    top_mota = df.nlargest(10, 'MOTA')
    colors_mota = gradient_bar_colors(10, PALETTE_GREEN)
    axes[1, 1].barh(range(10), top_mota['MOTA'].values, height=0.65,
                    color=colors_mota, edgecolor='white', linewidth=0.3,
                    alpha=0.9, zorder=3)
    for i, (val, c) in enumerate(zip(top_mota['MOTA'].values, colors_mota)):
        axes[1, 1].barh(i, val, height=0.85, color=c, alpha=0.1, zorder=1)
        axes[1, 1].text(val + 0.01, i, f"{val:.3f}", va='center',
                        fontsize=10, color=GREEN, fontweight='bold')
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].set_yticklabels(top_mota['Sequence'].values, fontsize=10)
    axes[1, 1].set_xlim(0, 1.08)
    axes[1, 1].invert_yaxis()
    style_axis(axes[1, 1], title="Top 10 Sequences: Player MOTA", xlabel="MOTA")

    # ── 5. Ball RMSE Distribution ────────────────────────────────────────
    premium_hist(axes[2, 0], df['Ball_RMSE'], ORANGE, ORANGE_DIM,
                 "Ball Tracking RMSE Distribution (px)", "RMSE (pixels)",
                 mean_color=YELLOW)
    stat_box(axes[2, 0],
             f"μ = {df['Ball_RMSE'].mean():.1f}px\n"
             f"med = {df['Ball_RMSE'].median():.1f}px\n"
             f"σ = {df['Ball_RMSE'].std():.1f}px")

    # ── 6. Team Clustering Accuracy ──────────────────────────────────────
    premium_hist(axes[2, 1], df['Team_Clustering_Acc'], PURPLE, PURPLE_DIM,
                 "Team Clustering Accuracy Distribution", "Accuracy")
    stat_box(axes[2, 1],
             f"μ = {df['Team_Clustering_Acc'].mean():.3f}\n"
             f"min = {df['Team_Clustering_Acc'].min():.3f}")

    # ── 7. F1 vs MOTA Scatter ────────────────────────────────────────────
    ax7 = axes[3, 0]
    scatter_colors = df['Team_Clustering_Acc'].values
    # Glow layer
    ax7.scatter(df['Ball_F1'], df['MOTA'], c=scatter_colors, cmap='cool',
                s=180, alpha=0.15, edgecolors='none', zorder=2)
    # Core layer
    sc = ax7.scatter(df['Ball_F1'], df['MOTA'], c=scatter_colors, cmap='cool',
                     s=70, alpha=0.9, edgecolors='white', linewidth=0.4, zorder=3)
    cbar = plt.colorbar(sc, ax=ax7, pad=0.02)
    cbar.set_label("Team Clustering Acc", fontsize=10, color=TEXT_SEC)
    cbar.ax.tick_params(colors=TEXT_SEC)
    cbar.outline.set_edgecolor(BORDER)
    style_axis(ax7, title="Ball F1 vs Player MOTA (color = Team Acc)",
               xlabel="Ball F1-Score", ylabel="MOTA")

    # ── 8. Composite Score Rank ──────────────────────────────────────────
    n_top = min(15, len(df))
    top_overall = df.head(n_top)
    colors_comp = gradient_bar_colors(n_top, PALETTE_PURPLE)
    axes[3, 1].barh(range(n_top), top_overall['Composite_Score'].values,
                    height=0.65, color=colors_comp, edgecolor='white',
                    linewidth=0.3, alpha=0.9, zorder=3)
    for i, (val, c) in enumerate(zip(top_overall['Composite_Score'].values,
                                     colors_comp)):
        axes[3, 1].barh(i, val, height=0.85, color=c, alpha=0.1, zorder=1)
        axes[3, 1].text(val + 0.01, i, f"{val:.3f}", va='center',
                        fontsize=9, color=PURPLE, fontweight='bold')
    axes[3, 1].set_yticks(range(n_top))
    axes[3, 1].set_yticklabels(top_overall['Sequence'].values, fontsize=9)
    axes[3, 1].set_xlim(0, 1.08)
    axes[3, 1].invert_yaxis()
    style_axis(axes[3, 1], title="Top 15 Overall Winners (Composite Score)",
               xlabel="Composite Score")

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plot_path = os.path.join(args.out_dir, "batch_evaluation_dashboard.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Dashboard saved to {plot_path}")


if __name__ == "__main__":
    main()
