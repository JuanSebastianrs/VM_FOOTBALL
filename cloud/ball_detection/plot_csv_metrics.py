import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the paths to the results.csv of each experiment
# We assume the user has downloaded their respective results.csv to local folders
experiments = {
    "Exp_0_Nano_640_Base": "cloud/ball_detection/results/exp1_nano_640/results.csv",
    "Exp_0_Nano_1280_Base": "cloud/ball_detection/results/exp2_nano_1280/results.csv",
    "Exp_0_Small_640_Base": "cloud/ball_detection/results/exp3_small_640/results.csv",
    "Exp_1_Nano_1280_Baseline": "cloud/ball_detection/results/exp_01_baseline/results.csv",
    "Exp_2_Nano_1280_DataCentric": "cloud/ball_detection/results/exp_02_data_centric/results.csv",
}

# Metrics to extract and plot
metrics_to_plot = {
    "mAP50": "       metrics/mAP50(B)",
    "Recall": "      metrics/recall(B)",
    "Precision": "   metrics/precision(B)"
}

os.makedirs("cloud/ball_detection/results/comparisons", exist_ok=True)

# Generate one plot per metric
for metric_name, column_name in metrics_to_plot.items():
    plt.figure(figsize=(10, 6))
    plt.title(f"Ablation Study: Validation {metric_name} vs Epochs (Ball)", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Track if we found any data for this metric
    data_found = False

    for exp_name, csv_path in experiments.items():
        if os.path.exists(csv_path):
            try:
                # Read CSV
                df = pd.read_csv(csv_path)
                # YOLO results.csv columns have leading spaces, strip them for matching if needed 
                # or use exact matches with spaces as defined in the dict
                
                # Check if column exists (handling potential whitespace issues)
                matched_col = None
                for col in df.columns:
                    if col.strip() == column_name.strip():
                        matched_col = col
                        break
                
                if matched_col:
                    epochs = df.iloc[:, 0]  # First column is always epoch
                    values = df[matched_col]
                    
                    # Highlight the best model with a thicker line
                    linewidth = 3.0 if "DataCentric" in exp_name else 1.5
                    alpha = 1.0 if "DataCentric" in exp_name else 0.7
                    
                    plt.plot(epochs, values, label=f"{exp_name} (Max: {values.max():.3f})", 
                             linewidth=linewidth, alpha=alpha)
                    data_found = True
                else:
                    print(f"Column '{column_name.strip()}' not found in {exp_name}")
            except Exception as e:
                print(f"Error parsing {csv_path}: {e}")
        else:
            print(f"Missing results file: {csv_path}")

    if data_found:
        plt.legend(loc="lower right", fontsize=10)
        save_path = f"cloud/ball_detection/results/comparisons/combined_{metric_name.lower()}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gáfica de {metric_name} guardada en: {save_path}")
    plt.close()

print("Generación de gráficas completada.")
