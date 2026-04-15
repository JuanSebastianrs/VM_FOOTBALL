"""
Generate Detailed MOT Evaluation Dashboard
Focuses on tracking stability: MOTA, IDF1, MOTP, and ID Switches.
Saves to results_final/evaluation/mot_detailed_dashboard.png
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
    parser = argparse.ArgumentParser("Generate MOT Detailed Dashboard")
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

    # Filtering rows with 0 tracks to avoid skewed distributions
    df_active = df[df['Tracks_Eval'] > 0].copy()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle("TacticalVision AI: Detailed MOT Performance Dashboard",
                 fontsize=22, weight='bold', y=0.98)

    # 1. MOTA Distribution
    sns.histplot(df_active['MOTA'], bins=15, kde=True, ax=axes[0, 0], color='#2E7D32')
    axes[0, 0].axvline(df_active['MOTA'].mean(), color='red', ls='--', 
                       label=f"Mean MOTA: {df_active['MOTA'].mean():.3f}")
    axes[0, 0].set_title("MOTA Distribution (Accuracy)")
    axes[0, 0].set_xlabel("MOTA")
    axes[0, 0].legend()

    # 2. IDF1 Distribution
    sns.histplot(df_active['IDF1'], bins=15, kde=True, ax=axes[0, 1], color='#1565C0')
    axes[0, 1].axvline(df_active['IDF1'].mean(), color='red', ls='--', 
                       label=f"Mean IDF1: {df_active['IDF1'].mean():.3f}")
    axes[0, 1].set_title("IDF1 Distribution (Long-term Identity)")
    axes[0, 1].set_xlabel("IDF1")
    axes[0, 1].legend()

    # 3. MOTP Distribution (Spatial Dissimilarity)
    sns.histplot(df_active['MOTP'], bins=15, kde=True, ax=axes[1, 0], color='#F9A825')
    axes[1, 0].axvline(df_active['MOTP'].mean(), color='red', ls='--', 
                       label=f"Mean MOTP: {df_active['MOTP'].mean():.3f}")
    axes[1, 0].set_title("MOTP Distribution (Spatial Dissimilarity - Lower is Better)")
    axes[1, 0].set_xlabel("MOTP (avg normalized distance)")
    axes[1, 0].legend()

    # 4. ID Switches (Log Distribution)
    # Using log scale for ID Switches as they can vary wildly
    df_active['log_idsw'] = np.log1p(df_active['ID_Switches'])
    sns.histplot(df_active['log_idsw'], bins=15, kde=True, ax=axes[1, 1], color='#C62828')
    axes[1, 1].set_title("ID Switches Distribution (Log Scale)")
    axes[1, 1].set_xlabel("log(1 + ID_Switches)")
    
    # 5. Top 10 by IDF1
    top_idf1 = df_active.nlargest(10, 'IDF1')
    sns.barplot(data=top_idf1, x='IDF1', y='Sequence', hue='Sequence',
                ax=axes[2, 0], palette='Blues_r', legend=False)
    axes[2, 0].set_title("Top 10 Sequences: Identity Stability (IDF1)")
    axes[2, 0].set_xlim(0, 1.0)

    # 6. MOTA vs IDF1 Correlation
    sns.scatterplot(data=df_active, x='MOTA', y='IDF1', hue='ID_Switches', 
                    size='ID_Switches', sizes=(20, 200), ax=axes[2, 1], palette='flare')
    axes[2, 1].set_title("Tracking Correlation: MOTA vs IDF1 (Size=IDSW)")
    axes[2, 1].set_xlabel("MOTA")
    axes[2, 1].set_ylabel("IDF1")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(args.out_dir, "mot_detailed_dashboard.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Detailed MOT Dashboard saved to: {plot_path}")

if __name__ == "__main__":
    main()
