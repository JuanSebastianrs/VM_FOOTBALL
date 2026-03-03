# ============================================================
# RF-DETR Player Detection - Vertex AI Spot Job Launcher
# ============================================================
# Submits a Custom Job to train RF-DETR for player detection.
# Downloads dataset tar from GCS, converts YOLO→COCO, trains RF-DETR.
#
# Usage:
#   .\cloud\submit_rfdetr_job.ps1                 # T4 GPU (default)
#   .\cloud\submit_rfdetr_job.ps1 -GPU L4         # L4 GPU (if available)
#   .\cloud\submit_rfdetr_job.ps1 -SmokeTest      # 1 epoch validation
#   .\cloud\submit_rfdetr_job.ps1 -DryRun         # Config check only
# ============================================================

param(
    [ValidateSet("L4", "T4")]
    [string]$GPU = "T4",
    [switch]$DryRun,
    [switch]$SmokeTest
)

$PROJECT_ID = "vm-football-489116"
$REGION = "us-central1"
$DATA_BUCKET = "vm-football-data"
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"
$JOB_NAME = "rfdetr-player-$($GPU.ToLower())-$TIMESTAMP"

# Pre-built PyTorch 2.2 container (CUDA 12.1, Python 3.10)
$CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest"

# GPU config
if ($GPU -eq "L4") {
    $ACCELERATOR_TYPE = "NVIDIA_L4"
    $MACHINE_TYPE = "g2-standard-8"
}
else {
    $ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
    $MACHINE_TYPE = "n1-standard-8"
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " RF-DETR Player Detection - Vertex AI Job" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  GPU:       $GPU ($ACCELERATOR_TYPE)"
Write-Host "  Machine:   $MACHINE_TYPE"
Write-Host "  Container: pytorch-gpu.2-2.py310"
Write-Host "  SmokeTest: $($SmokeTest.IsPresent)"
Write-Host "  Classes:   1 (player)"
Write-Host ""

# --- Step 1: Validate ---
Write-Host "[1/5] Validating Environment..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID 2>$null
gcloud services enable aiplatform.googleapis.com 2>$null
gcloud services enable storage.googleapis.com 2>$null
Write-Host "   [OK] APIs validated" -ForegroundColor Green

# --- Step 2: Upload Training Code ---
Write-Host ""
Write-Host "[2/5] Uploading Training Code..." -ForegroundColor Yellow

# Clean previous rfdetr code
gcloud storage rm -r "gs://$DATA_BUCKET/code/rfdetr_player_detection/" 2>$null

# Upload rfdetr_player_detection package
gcloud storage cp -r "cloud/rfdetr_player_detection" "gs://$DATA_BUCKET/code/"
Write-Host "   [OK] Code uploaded to gs://$DATA_BUCKET/code/rfdetr_player_detection/" -ForegroundColor Green

# --- Dry Run ---
if ($DryRun) {
    Write-Host ""
    Write-Host "[DRY RUN] Configuration validated. No job submitted." -ForegroundColor Green
    Write-Host "  Would create job: $JOB_NAME"
    Write-Host "  GPU: $ACCELERATOR_TYPE on $MACHINE_TYPE"
    Write-Host "  Container: $CONTAINER_URI"
    Write-Host "  Dataset: gs://$DATA_BUCKET/reorganized_dataset.tar.gz"
    exit 0
}

# --- Step 3: Configure Startup Script ---
Write-Host ""
Write-Host "[3/5] Configuring Job..." -ForegroundColor Yellow

if ($SmokeTest) {
    Write-Host "   [INFO] Smoke Test: overriding to 1 epoch, batch 2" -ForegroundColor Yellow
}

# Build startup script content
# IMPORTANT: Single-quote here-string prevents PowerShell interpolation.
# Bash $ variables pass through. PowerShell vars are replaced afterward.

$startupScript = @'
#!/bin/bash
set -euo pipefail

# ============================================================
# Trap: upload crash log on ANY failure
# ============================================================
GCS_OUTPUT="gs://PLACEHOLDER_BUCKET/models/rfdetr_player"
OUTPUT_DIR="/workspace/runs"

cleanup_on_error() {
    echo ""
    echo "============================================"
    echo " CRASH DETECTED - Uploading error logs"
    echo "============================================"
    
    mkdir -p "$OUTPUT_DIR"
    {
        echo "Crash at: $(date)"
        echo "Exit code: $?"
        echo "Last command: ${BASH_COMMAND:-unknown}"
        echo ""
        echo "GPU info:"
        nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
        echo ""
        echo "Disk usage:"
        df -h 2>/dev/null || true
        echo ""
        echo "Memory:"
        free -h 2>/dev/null || true
        echo ""
        echo "Python packages:"
        pip list 2>/dev/null | grep -iE "rfdetr|pydantic|torch|numpy|transformers|accelerate|timm" || true
    } > "$OUTPUT_DIR/CRASH_LOG.txt"
    
    gcloud storage rsync -r "$OUTPUT_DIR" "$GCS_OUTPUT/" 2>/dev/null || true
    echo "[OK] Crash artifacts uploaded to $GCS_OUTPUT"
}

trap cleanup_on_error ERR

echo '======================================'
echo ' RF-DETR Player Detection - Vertex AI'
echo '======================================'
echo "Started at: $(date)"
echo "GPU target: PLACEHOLDER_GPU_INFO"

# ---- Phase 0: Environment Cleanup ----
echo '[0/5] Preparing environment...'
pip uninstall -y torch_xla 2>/dev/null || true
export PJRT_DEVICE=CPU
export MKL_SERVICE_FORCE_INTEL=1
export PYTHONWARNINGS=ignore::UserWarning

# ---- Phase 1: GPU Check ----
echo '[1/5] GPU Sanity Check...'
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('[FATAL] No GPU detected!')
    print('  torch.cuda.is_available():', torch.cuda.is_available())
    print('  CUDA_VISIBLE_DEVICES:', __import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'not set'))
    sys.exit(1)
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
cc = torch.cuda.get_device_capability(0)
print(f'GPU: {name} ({mem:.1f} GB, cc {cc[0]}.{cc[1]})')
# Compute test
x = torch.randn(64, 64, device='cuda', requires_grad=True)
y = x @ x.t()
loss = y.sum()
loss.backward()
del x, y, loss
torch.cuda.empty_cache()
print('[OK] GPU forward+backward pass verified')
"
echo '[OK] GPU ready'

# ---- Phase 2: Download Code ----
echo '[2/5] Downloading training code...'
mkdir -p /workspace/rfdetr_player_detection
cd /workspace/rfdetr_player_detection
gcloud storage cp -r gs://PLACEHOLDER_BUCKET/code/rfdetr_player_detection/* .

echo "Files downloaded:"
ls -la
echo ""

# Verify critical files exist
for required_file in train_cloud.py config.yaml setup.py; do
    if [ ! -f "$required_file" ]; then
        echo "[FATAL] Missing required file: $required_file"
        ls -la
        exit 1
    fi
done
echo "[OK] All required files present"

# ---- Phase 3: Install Dependencies ----
echo '[3/5] Installing dependencies...'

# Pin pydantic V2 BEFORE anything else (CRITICAL for rfdetr)
pip install -q --force-reinstall "pydantic>=2.5.0"

# Pin numpy <2 (prevents ABI breaks with PyTorch)
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"

# Install from setup.py (includes rfdetr, transformers, etc.)
pip install -q --no-cache-dir -e .

# Re-pin after setup.py install (some deps may pull numpy 2.x or pydantic V1)
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"
pip install -q --force-reinstall "pydantic>=2.5.0"

# Force headless opencv LAST (prevents rfdetr/ultralytics from overwriting)
pip install -q --force-reinstall --no-deps "opencv-python-headless>=4.8.0"

echo '[OK] Dependencies installed'
echo "Key packages:"
pip list 2>/dev/null | grep -iE "rfdetr|pydantic|torch|numpy|transformers|accelerate|timm|opencv|supervision" || true

PLACEHOLDER_SMOKE_OVERRIDE

# ---- Phase 4: Set GCS env vars ----
echo '[4/5] Configuring GCS paths...'
export GCS_DATASET_PATH=gs://PLACEHOLDER_BUCKET/reorganized_dataset/
export GCS_OUTPUT_PATH=gs://PLACEHOLDER_BUCKET/models/rfdetr_player
export GCS_COCO_CACHE=gs://PLACEHOLDER_BUCKET/coco_cache_player
export VERTEX_AI_JOB=true

# ---- Phase 5: Train ----
echo '[5/5] Starting RF-DETR Player Detection Training...'
echo "Working directory: $(pwd)"
echo "Python: $(python3 --version)"
echo ""

# Run training with unbuffered output for real-time logs
python3 -u train_cloud.py
TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Training exited with code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ''
echo '======================================'
echo ' TRAINING COMPLETE - Shutting down'
echo '======================================'
'@

# Now replace placeholders with actual PowerShell variables
$startupScript = $startupScript.Replace("PLACEHOLDER_BUCKET", $DATA_BUCKET)
$startupScript = $startupScript.Replace("PLACEHOLDER_GPU_INFO", "$GPU ($ACCELERATOR_TYPE)")

# Handle smoke test override
if ($SmokeTest) {
    $smokeBlock = @'
# SMOKE TEST: Override config for quick validation
echo "[SMOKE] Overriding config to 1 epoch, batch 2..."
python3 -c "
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
for name in c.get('rfdetr',{}).get('experiments',{}):
    c['rfdetr']['experiments'][name]['epochs'] = 1
    c['rfdetr']['experiments'][name]['batch_size'] = 2
    c['rfdetr']['experiments'][name]['grad_accum_steps'] = 1
with open('config.yaml','w') as f:
    yaml.dump(c, f, default_flow_style=False)
print('[SMOKE] Config overridden: 1 epoch, batch 2, grad_accum 1')
"
'@
    $startupScript = $startupScript.Replace("PLACEHOLDER_SMOKE_OVERRIDE", $smokeBlock)
}
else {
    $startupScript = $startupScript.Replace("PLACEHOLDER_SMOKE_OVERRIDE", "# No smoke test override")
}

# Save and upload startup script (ensure LF line endings for Linux)
$scriptPath = [System.IO.Path]::Combine($env:TEMP, "rfdetr_startup.sh")
$startupScript | Out-File -FilePath $scriptPath -Encoding utf8 -NoNewline
(Get-Content $scriptPath -Raw).Replace("`r`n", "`n") | Set-Content -Path $scriptPath -NoNewline
gcloud storage cp $scriptPath "gs://$DATA_BUCKET/code/rfdetr_player_detection/startup.sh" 2>$null
Remove-Item $scriptPath -ErrorAction SilentlyContinue

Write-Host "   [OK] Job configured (with error trapping and crash upload)" -ForegroundColor Green

# --- Step 4: Submit Job ---
Write-Host ""
Write-Host "[4/5] Submitting Job to Vertex AI..." -ForegroundColor Yellow

# Generate YAML job spec
# STANDARD strategy: On-demand pricing, guaranteed no preemptions
$jobSpecYaml = @"
workerPoolSpecs:
  - machineSpec:
      machineType: $MACHINE_TYPE
      acceleratorType: $ACCELERATOR_TYPE
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 300
    containerSpec:
      imageUri: $CONTAINER_URI
      command:
        - bash
        - -c
        - |
          gcloud storage cp gs://$DATA_BUCKET/code/rfdetr_player_detection/startup.sh /tmp/startup.sh
          chmod +x /tmp/startup.sh
          bash /tmp/startup.sh
scheduling:
  timeout: 172800s
  strategy: STANDARD
"@

$yamlPath = [System.IO.Path]::Combine($env:TEMP, "rfdetr_job_spec.yaml")
$jobSpecYaml | Out-File -FilePath $yamlPath -Encoding utf8

# Submit
$JOB_ID = ""
try {
    $submitOutput = gcloud ai custom-jobs create `
        --region=$REGION `
        --display-name=$JOB_NAME `
        --config=$yamlPath `
        --format="value(name)"
    $JOB_ID = $submitOutput -replace "projects/.*/customJobs/", ""
}
catch {
    Write-Host "   [ERROR] Submission Failed:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

if ([string]::IsNullOrWhiteSpace($JOB_ID)) {
    Write-Host "   [ERROR] Submission Failed (No Job ID returned)." -ForegroundColor Red
    exit 1
}

Remove-Item $yamlPath -ErrorAction SilentlyContinue
Write-Host "   [OK] Job Submitted! ID: $JOB_ID" -ForegroundColor Green
Write-Host "   [INFO] STANDARD Strategy: Dedicated GPU for uninterrupted training" -ForegroundColor DarkGray

# --- Step 5: Streaming Logs ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MONITORING: $JOB_ID" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Streaming logs... (Ctrl+C to stop, job continues)" -ForegroundColor Gray

Start-Sleep -Seconds 10
try {
    gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION
}
catch {
    Write-Host "`n   [INFO] Log streaming interrupted." -ForegroundColor Yellow
}

# --- Poll & Download ---
Write-Host ""
Write-Host "[POST] Checking job status..." -ForegroundColor Yellow

do {
    $status = gcloud ai custom-jobs describe $JOB_ID --region=$REGION --format="value(state)"
    if ($status -eq "JOB_STATE_SUCCEEDED" -or $status -eq "JOB_STATE_FAILED" -or $status -eq "JOB_STATE_CANCELLED") {
        break
    }
    Write-Host "   Status: $status (checking in 30s)..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
} until ($false)

if ($status -eq "JOB_STATE_SUCCEEDED") {
    Write-Host "   [SUCCESS] Training Complete!" -ForegroundColor Green

    # Download results
    $localResults = "results\rfdetr_player_$TIMESTAMP"
    New-Item -ItemType Directory -Force -Path $localResults | Out-Null

    Write-Host "   Downloading results from GCS..." -ForegroundColor Cyan
    gcloud storage cp -r "gs://$DATA_BUCKET/models/rfdetr_player/*" "$localResults\"

    Write-Host ""
    Write-Host "   [OK] Results saved to: $localResults" -ForegroundColor Green
    Write-Host "   Contains: weights (.pth), training history, plots"
}
else {
    Write-Host "   [ERROR] Job finished with status: $status" -ForegroundColor Red
    Write-Host "   Check logs: gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION"

    # Try to download crash log
    Write-Host "   Attempting to download crash log..." -ForegroundColor Yellow
    $crashDir = "results\rfdetr_crash_$TIMESTAMP"
    New-Item -ItemType Directory -Force -Path $crashDir | Out-Null
    gcloud storage cp "gs://$DATA_BUCKET/models/rfdetr_player/CRASH_LOG.txt" "$crashDir\" 2>$null
    if (Test-Path "$crashDir\CRASH_LOG.txt") {
        Write-Host "   [INFO] Crash log saved to: $crashDir\CRASH_LOG.txt" -ForegroundColor Yellow
        Get-Content "$crashDir\CRASH_LOG.txt"
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Cyan
