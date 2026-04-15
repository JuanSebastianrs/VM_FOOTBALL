"""
Generate Batch Evaluation Dashboard
Reads the CSV from batch_evaluator.py and produces:
1. A 4x2 panel dashboard PNG
2. A top5_winners.txt file
3. An enriched CSV with Composite_Score
4. Summary statistics printed to console
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser("Generate Batch Dashboard")
    parser.add_argument("--csv_file", type=str,
                        default="results_final/evaluation/batch_evaluation.csv")
    parser.add_argument("--out_dir", type=str,
                        default="results_final/evaluation")
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return

    df = pd.read_csv(args.csv_file)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ── Composite Score ──────────────────────────────────────────────────
    # Normalize RMSE (lower is better) to [0, 1]
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
    print(f"Enriched CSV saved to {enriched_csv}")

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
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle("TacticalVision AI: Batch Evaluation Dashboard",
                 fontsize=22, weight='bold', y=0.98)

    # 1. Ball F1 Distribution
    sns.histplot(df['Ball_F1'], bins=15, kde=True, ax=axes[0, 0], color='#2196F3')
    axes[0, 0].axvline(df['Ball_F1'].mean(), color='red', ls='--', label=f"Mean: {df['Ball_F1'].mean():.3f}")
    axes[0, 0].set_title("Ball Tracking F1-Score Distribution")
    axes[0, 0].set_xlabel("F1-Score")
    axes[0, 0].legend()

    # 2. MOTA Distribution
    sns.histplot(df['MOTA'], bins=15, kde=True, ax=axes[0, 1], color='#4CAF50')
    axes[0, 1].axvline(df['MOTA'].mean(), color='red', ls='--', label=f"Mean: {df['MOTA'].mean():.3f}")
    axes[0, 1].set_title("Player Tracking MOTA Distribution")
    axes[0, 1].set_xlabel("MOTA")
    axes[0, 1].legend()

    # 3. Top 10 by Ball F1
    top_f1 = df.nlargest(10, 'Ball_F1')
    sns.barplot(data=top_f1, x='Ball_F1', y='Sequence', hue='Sequence',
                ax=axes[1, 0], palette='Blues_r', legend=False)
    axes[1, 0].set_title("Top 10 Sequences: Ball F1")
    axes[1, 0].set_xlim(0, 1.0)

    # 4. Top 10 by MOTA
    top_mota = df.nlargest(10, 'MOTA')
    sns.barplot(data=top_mota, x='MOTA', y='Sequence', hue='Sequence',
                ax=axes[1, 1], palette='Greens_r', legend=False)
    axes[1, 1].set_title("Top 10 Sequences: Player MOTA")
    axes[1, 1].set_xlim(0, 1.0)

    # 5. Ball RMSE Distribution
    sns.histplot(df['Ball_RMSE'], bins=15, kde=True, ax=axes[2, 0], color='#FF9800')
    axes[2, 0].axvline(df['Ball_RMSE'].mean(), color='red', ls='--', label=f"Mean: {df['Ball_RMSE'].mean():.1f}px")
    axes[2, 0].set_title("Ball Tracking RMSE Distribution (px)")
    axes[2, 0].set_xlabel("RMSE (pixels)")
    axes[2, 0].legend()

    # 6. Team Clustering Accuracy Distribution
    sns.histplot(df['Team_Clustering_Acc'], bins=15, kde=True, ax=axes[2, 1], color='#9C27B0')
    axes[2, 1].axvline(df['Team_Clustering_Acc'].mean(), color='red', ls='--',
                       label=f"Mean: {df['Team_Clustering_Acc'].mean():.3f}")
    axes[2, 1].set_title("Team Clustering Accuracy Distribution")
    axes[2, 1].set_xlabel("Accuracy")
    axes[2, 1].legend()

    # 7. F1 vs MOTA Scatter (correlation)
    scatter = axes[3, 0].scatter(df['Ball_F1'], df['MOTA'],
                                  c=df['Team_Clustering_Acc'], cmap='viridis',
                                  s=80, edgecolors='white', linewidth=0.5)
    axes[3, 0].set_title("Ball F1 vs Player MOTA (color = Team Acc)")
    axes[3, 0].set_xlabel("Ball F1-Score")
    axes[3, 0].set_ylabel("MOTA")
    plt.colorbar(scatter, ax=axes[3, 0], label="Team Clustering Acc")

    # 8. Composite Score Rank (Top 15)
    top_overall = df.head(min(15, len(df)))
    sns.barplot(data=top_overall, x='Composite_Score', y='Sequence', hue='Sequence',
                ax=axes[3, 1], palette='flare', legend=False)
    axes[3, 1].set_title("Top 15 Overall Winners (Composite Score)")
    axes[3, 1].set_xlim(0, 1.0)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plot_path = os.path.join(args.out_dir, "batch_evaluation_dashboard.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Dashboard saved to {plot_path}")


if __name__ == "__main__":
    main()
