import json
import sys

try:
    with open('cloud/ball_detection/results/jobs.json', 'r') as f:
        data = json.load(f)
        
    for j in data:
        name = j.get('name', '').split('/')[-1]
        state = j.get('state', '')
        created = j.get('createTime', '')
        err_msg = j.get('error', {}).get('message', 'None')
        print(f"ID: {name} | State: {state} | Created: {created}")
        print(f"Error: {err_msg[:200]}\n")
except Exception as e:
    print(f"Failed to parse jobs: {e}")
