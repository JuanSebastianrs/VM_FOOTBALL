import sys

try:
    with open('cloud/ball_detection/results/job_full_logs.txt', 'r', encoding='utf-16le') as f:
        lines = f.readlines()
        
    print("=== Last 100 RAW lines ===")
    for ln in lines[-100:]:
        print(ln.strip())
except Exception as e:
    print(f"Error reading file: {e}")
