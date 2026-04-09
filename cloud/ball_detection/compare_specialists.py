import pandas as pd
import os

try:
    df1 = pd.read_csv("cloud/ball_detection/results/exp01_results.csv")
    df1.columns = df1.columns.str.strip()
    last_row1 = df1.iloc[-1]
    # Metrics usually are: metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
    map50_1 = last_row1['metrics/mAP50(B)']
    map50_95_1 = last_row1['metrics/mAP50-95(B)']
    epoch1 = last_row1['epoch']
    print(f"EXP 01 (Baseline): Epoch {epoch1}, mAP50: {map50_1:.4f}, mAP50-95: {map50_95_1:.4f}")
except Exception as e:
    print(f"Error reading EXP 01: {e}")

try:
    df2 = pd.read_csv("cloud/ball_detection/results/exp02_results.csv")
    df2.columns = df2.columns.str.strip()
    last_row2 = df2.iloc[-1]
    map50_2 = last_row2['metrics/mAP50(B)']
    map50_95_2 = last_row2['metrics/mAP50-95(B)']
    epoch2 = last_row2['epoch']
    print(f"EXP 02 (Data-Centric): Epoch {epoch2}, mAP50: {map50_2:.4f}, mAP50-95: {map50_95_2:.4f}")
except Exception as e:
    print(f"Error reading EXP 02: {e}")

try:
    df3 = pd.read_csv("cloud/ball_detection/results/exp_03_p2_sota/results.csv")
    df3.columns = df3.columns.str.strip()
    last_row3 = df3.iloc[-1]
    map50_3 = last_row3['metrics/mAP50(B)']
    map50_95_3 = last_row3['metrics/mAP50-95(B)']
    epoch3 = last_row3['epoch']
    print(f"EXP 03 (SOTA-P2): Epoch {epoch3}, mAP50: {map50_3:.4f}, mAP50-95: {map50_95_3:.4f}")
except Exception as e:
    print(f"Error reading EXP 03: {e}")
