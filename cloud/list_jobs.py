import subprocess
import json

try:
    cmd = ['gcloud', 'ai', 'custom-jobs', 'list', '--region=us-central1', '--limit=5', '--format=json']
    res = subprocess.check_output(cmd, text=True)
    jobs = json.loads(res)
    for j in jobs:
        name = j.get('name', '').split('/')[-1]
        state = j.get('state', '')
        created = j.get('createTime', '')
        err_msg = j.get('error', {}).get('message', 'None')
        print(f"ID: {name}")
        print(f"  State: {state} | Created: {created}")
        print(f"  Error: {err_msg[:250]}\n")
except Exception as e:
    print(f"Error: {e}")
