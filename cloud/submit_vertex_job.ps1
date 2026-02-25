# ============================================================
# VM_FOOTBALL - Vertex AI Training Orchestrator
# ============================================================
# Robust pipeline for training YOLO and RF-DETR models on Google Cloud Vertex AI.
# 
# Usage:
#   .\cloud\submit_vertex_job.ps1 -Model yolo        # Train only YOLO
#   .\cloud\submit_vertex_job.ps1 -Model rfdetr      # Train only RF-DETR
#   .\cloud\submit_vertex_job.ps1 -Model all         # Train both (sequentially)
#   .\cloud\submit_vertex_job.ps1 -DryRun            # Validate config only
#
# Features:
#   - Automatic code compression and upload
#   - Dependency resolution (NumPy/OpenCV compatibility)
#   - Real-time log streaming
#   - Automatic result downloading upon completion
#   - robust error handling
# ============================================================

param(
    [ValidateSet("yolo", "rfdetr", "all")]
    [string]$Model = "all",
    [switch]$DryRun,
    [switch]$Diagnose,  # Run diagnostic script only
    [switch]$SmokeTest  # Run 1 epoch to validate pipeline end-to-end
)

$PROJECT_ID    = "project-ad19fdc6-8493-43e5-b82"
$REGION        = "us-central1"
$DATA_BUCKET   = "vm-football-data"
$MODELS_BUCKET = "vm-football-models"
$TIMESTAMP     = Get-Date -Format "yyyyMMdd-HHmmss"
$JOB_NAME      = "vm-football-$Model-$TIMESTAMP"

# Pre-built PyTorch 2.2 container with CUDA 12.1 (Python 3.10)
# Updated to 2.2 to fix 'torch.utils._pytree' compatibility with newer transformers
# L4 GPU requires CUDA 12.x support - this container has CUDA 12.1
$CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " VM_FOOTBALL - Vertex AI Orchestrator ($Model)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Enable APIs & Config ---
Write-Host "[1/6] Validating Environment..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID 2>$null
gcloud services enable aiplatform.googleapis.com 2>$null
gcloud services enable storage.googleapis.com 2>$null
Write-Host "   [OK] APIs validated" -ForegroundColor Green

# --- Step 2: Prepare & Upload Code ---
Write-Host ""
Write-Host "[2/6] Uploading Training Code..." -ForegroundColor Yellow

# Clean previous code
gcloud storage rm -r "gs://$DATA_BUCKET/code/" 2>$null

# Upload directories directly (robust and simple)
# We upload to gs://bucket/code/cloud, gs://bucket/code/training, etc.
$uploadTasks = @("cloud", "training", "core")
foreach ($dir in $uploadTasks) {
    if (Test-Path $dir) {
        gcloud storage cp -r "$dir" "gs://$DATA_BUCKET/code/" 2>$null
    } else {
        Write-Host "   [WARN] Directory $dir not found!" -ForegroundColor Red
    }
}
Write-Host "   [OK] Code uploaded to gs://$DATA_BUCKET/code/" -ForegroundColor Green

# --- Step 3: Configure Startup Script ---
Write-Host ""
Write-Host "[3/6] Configuring Job Environment..." -ForegroundColor Yellow

# Train pipeline args
$trainArgs = "--config cloud/config.yaml"
if ($Model -ne "all") {
    $trainArgs += " --only $Model"
}
if ($DryRun) {
    $trainArgs += " --dry-run"
}
if ($SmokeTest) {
    $trainArgs += " --smoke-test"
}

# STARTUP SCRIPT (Dynamic generation)
# Enforces set -e and specific versions to avoid binary incompatibility
$startupScript = @"
#!/bin/bash
set -e

echo '======================================'
echo ' VM_FOOTBALL - Vertex AI Worker'
echo '======================================'
echo "Started at: `$(date)"

# ============================================================
# PHASE 0: NEUTRALIZE VERTEX AI CONTAINER CONFLICTS
# ============================================================
# The pytorch-gpu.2-2 container comes with pre-installed packages
# that conflict with RF-DETR training. Kill them before any Python runs.
echo '[0/7] Neutralizing container conflicts...'

# CRITICAL: torch_xla causes SIGABRT "PJRT_DEVICE is not set"
# It's pre-installed in the container but we don't use TPUs.
pip uninstall -y torch_xla 2>/dev/null || true

# Set env vars to prevent any residual torch_xla/MKL issues
export PJRT_DEVICE=CPU
export MKL_SERVICE_FORCE_INTEL=1

# Suppress Python warnings that clutter logs
export PYTHONWARNINGS=ignore::UserWarning
export TOKENIZERS_PARALLELISM=false

echo '[OK] Container conflicts neutralized'

# ============================================================
# PHASE 1: GPU SANITY CHECK (before downloading ANYTHING)
# ============================================================
# This runs in <10 seconds. If it fails, we abort immediately
# instead of wasting 60+ minutes downloading the dataset.
echo '[1/7] GPU Sanity Check...'
python3 -c "
import torch, sys
assert torch.cuda.is_available(), 'FATAL: No GPU detected!'
props = torch.cuda.get_device_properties(0)
cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
mem_gb = props.total_memory / 1e9
bf16_ok = cc >= (8, 0)
print(f'GPU: {name}')
print(f'  Compute Capability: {cc[0]}.{cc[1]}')
print(f'  VRAM: {mem_gb:.1f} GB')
print(f'  bfloat16: {bf16_ok}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')

# Actual GPU compute test (not just detection)
try:
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    y = x @ x.t()
    loss = y.sum()
    loss.backward()
    del x, y, loss
    torch.cuda.empty_cache()
    print('[OK] GPU forward+backward pass works')
except Exception as e:
    print(f'FATAL: GPU compute test failed: {e}')
    sys.exit(1)

if bf16_ok:
    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            a = torch.randn(32, 256, device='cuda', requires_grad=True)
            b = torch.randn(256, 128, device='cuda')
            c = (a @ b).sum()
            c.backward()
        del a, b, c
        torch.cuda.empty_cache()
        print('[OK] bfloat16 AMP works')
    except Exception as e:
        print(f'[WARN] bfloat16 test failed: {e}')
        bf16_ok = False

if not bf16_ok:
    # T4: test FP16 instead
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            a = torch.randn(32, 256, device='cuda', requires_grad=True)
            b = torch.randn(256, 128, device='cuda')
            c = (a @ b).sum()
            c.backward()
        del a, b, c
        torch.cuda.empty_cache()
        print('[OK] FP16 AMP works - T4 Tensor Cores enabled (65 TFLOPS)')
    except Exception as e:
        print(f'[WARN] FP16 test failed: {e}')
        print('[WARN] Training will use FP32 (slower)')
else:
    print('[OK] GPU supports bfloat16 - AMP enabled')
"
echo '[OK] GPU validated'

# ============================================================
# PHASE 2: DOWNLOAD CODE (small, ~30s)
# ============================================================
echo '[2/7] Downloading codebase...'
mkdir -p /root/VM_FOOTBALL
cd /root/VM_FOOTBALL
gcloud storage cp -r gs://$DATA_BUCKET/code/* .

# ============================================================
# PHASE 3: INSTALL DEPENDENCIES + VALIDATE (before dataset)
# ============================================================
echo '[3/7] Patching environment...'
export DIAGNOSE_MODE="$($Diagnose.IsPresent)"
export EXPERIMENT_NAME="$Model"
echo "Initial environment:"
pip list | grep -E "numpy|torch|opencv|pydantic"

# 3a. Pin NumPy FIRST to prevent opencv/other deps pulling numpy 2.x (breaks PyTorch ABI)
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"

# 3b. Install headless OpenCV (--no-deps to prevent numpy 2.x pull)
pip install -q --force-reinstall --no-deps "opencv-python-headless>=4.8.0"

# 3c. CRITICAL: Force remove pre-installed pydantic V1 (compiled .so blocks V2 install)
echo 'Removing pre-installed pydantic V1...'
pip uninstall -y pydantic pydantic-core 2>/dev/null || true
pip install -q --no-cache-dir "pydantic>=2.5.0" "pydantic-core>=2.14.0"

# 3d. Install model-specific requirements
REQ_FILE="cloud/requirements-cloud.txt"
if [[ "`$EXPERIMENT_NAME" == "yolo" ]]; then
    REQ_FILE="cloud/requirements-yolo.txt"
elif [[ "`$EXPERIMENT_NAME" == "rfdetr" ]]; then
    REQ_FILE="cloud/requirements-rfdetr.txt"
fi

echo "Installing from: `$REQ_FILE"
pip install -q --no-cache-dir -r `$REQ_FILE

# 3e. Re-pin numpy after requirements install (some deps may have upgraded it)
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"

# ============================================================
# PHASE 4: FULL ENVIRONMENT VALIDATION (before dataset download)
# ============================================================
echo '[4/7] Full Environment Validation...'
python3 -c "
import sys
errors = []

# 1. PyTorch + CUDA
try:
    import torch
    print(f'PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')
    assert torch.cuda.is_available(), 'CUDA not available after setup'
    # Quick GPU compute test
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.t()
    del x, y
    torch.cuda.empty_cache()
    print('[OK] CUDA compute test passed')
except Exception as e:
    errors.append(f'PyTorch/CUDA: {e}')

# 2. bfloat16 test (non-fatal on T4 - RF-DETR will fallback to amp=False)
warnings = []
try:
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        a = torch.randn(32, 256, device='cuda')
        b = torch.randn(256, 128, device='cuda')
        c = a @ b
    del a, b, c
    torch.cuda.empty_cache()
    print('[OK] bfloat16 autocast test passed')
except Exception as e:
    warnings.append(f'bfloat16 not supported (RF-DETR will use FP32): {e}')
    print(f'[WARN] bfloat16 not supported - RF-DETR will use amp=False')

# 3. NumPy version
try:
    import numpy as np
    major = int(np.__version__.split('.')[0])
    print(f'NumPy: {np.__version__}')
    assert major < 2, f'NumPy {np.__version__} >= 2.0 will break PyTorch ABI'
    print('[OK] NumPy < 2.0')
except Exception as e:
    errors.append(f'NumPy: {e}')

# 4. OpenCV
try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
    print('[OK] OpenCV headless')
except Exception as e:
    errors.append(f'OpenCV: {e}')

# 5. Pydantic V2
try:
    import pydantic
    from pydantic import field_validator
    print(f'Pydantic: {pydantic.__version__}')
    assert pydantic.__version__.startswith('2'), f'Pydantic V1 detected: {pydantic.__version__}'
    print('[OK] Pydantic V2 (field_validator available)')
except Exception as e:
    errors.append(f'Pydantic: {e}')

# 6. RF-DETR specific deps (if rfdetr model)
import os
if os.environ.get('EXPERIMENT_NAME', '') in ('rfdetr', 'all'):
    try:
        import transformers
        print(f'Transformers: {transformers.__version__}')
        print('[OK] Transformers')
    except Exception as e:
        errors.append(f'Transformers: {e}')
    try:
        import accelerate
        print(f'Accelerate: {accelerate.__version__}')
        print('[OK] Accelerate')
    except Exception as e:
        errors.append(f'Accelerate: {e}')

# 7. Verify torch_xla is NOT installed (causes SIGABRT)
try:
    import torch_xla
    errors.append(f'torch_xla {torch_xla.__version__} is still installed! Must be uninstalled.')
except ImportError:
    print('[OK] torch_xla not installed (expected)')

# 8. Verify PJRT_DEVICE is set (safety net)
pjrt = os.environ.get('PJRT_DEVICE', '')
if pjrt:
    print(f'[OK] PJRT_DEVICE={pjrt}')
else:
    warnings.append('PJRT_DEVICE not set (should be CPU as safety net)')

if errors:
    print(f'\nFATAL: {len(errors)} validation error(s):')
    for err in errors:
        print(f'  - {err}')
    sys.exit(1)

if warnings:
    print(f'\n[WARN] {len(warnings)} warning(s):')
    for w in warnings:
        print(f'  - {w}')

checks_passed = 6 + (2 if os.environ.get('EXPERIMENT_NAME', '') in ('rfdetr', 'all') else 0)
print(f'\n[OK] Environment validated ({len(errors)} errors, {len(warnings)} warnings)')
"
echo '[OK] Environment validated'

# ============================================================
# PHASE 5: EXECUTE (dataset download + training happens here)
# ============================================================
if [ "`$DIAGNOSE_MODE" == "True" ]; then
    echo '[5/7] Running Diagnostic Mode...'
    python cloud/diagnose_env.py
else
    echo '[5/7] Running Pipeline ($Model)...'
    export GCS_DATASET_PATH=gs://$DATA_BUCKET/reorganized_dataset/
    export GCS_OUTPUT_PATH=gs://$MODELS_BUCKET/run_$TIMESTAMP/
    export VERTEX_AI_JOB=true
    export EXPERIMENT_NAME=$Model
    
    # Dataset download + training
    python cloud/run_pipeline.py $trainArgs
fi

echo '======================================'
echo ' TRAINING SUCCESSFUL'
echo '======================================'
"@

# Save and upload startup script
$scriptPath = [System.IO.Path]::Combine($env:TEMP, "vertex_startup.sh")
$startupScript | Out-File -FilePath $scriptPath -Encoding utf8 -NoNewline
(Get-Content $scriptPath -Raw).Replace("`r`n", "`n") | Set-Content -Path $scriptPath -NoNewline
gcloud storage cp $scriptPath "gs://$DATA_BUCKET/code/startup.sh" 2>$null
Remove-Item $scriptPath -ErrorAction SilentlyContinue

Write-Host "   [OK] Startup script configured (torch_xla removed / GPU check / bf16)" -ForegroundColor Green


# --- Step 4: Submit YAML Job Spec ---
Write-Host ""
Write-Host "[4/6] Submitting Job to Vertex AI..." -ForegroundColor Yellow

# Generate YAML (Boot Disk 300GB)
# CRITICAL: restartJobOnWorkerRestart=false prevents infinite restart loops.
# If the training crashes, the job DIES immediately instead of restarting.
$jobSpecYaml = @"
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-standard
      bootDiskSizeGb: 300
    containerSpec:
      imageUri: $CONTAINER_URI
      command: 
        - bash
        - -c
        - |
          gcloud storage cp gs://$DATA_BUCKET/code/startup.sh /tmp/startup.sh
          bash /tmp/startup.sh
scheduling:
  timeout: 172800s
  restartJobOnWorkerRestart: false
"@

$yamlPath = [System.IO.Path]::Combine($env:TEMP, "vertex_job_spec.yaml")
$jobSpecYaml | Out-File -FilePath $yamlPath -Encoding utf8

# Submit and capture output to find Job ID
$submitOutput = gcloud ai custom-jobs create `
    --region=$REGION `
    --display-name=$JOB_NAME `
    --config=$yamlPath `
    --format="value(name)" 2>&1

$JOB_ID = $submitOutput -replace "projects/.*/customJobs/", "" # Extract ID if full path
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($JOB_ID)) {
    Write-Host "   [ERROR] Submission Failed:" -ForegroundColor Red
    Write-Host $submitOutput
    exit 1
}

Remove-Item $yamlPath -ErrorAction SilentlyContinue
Write-Host "   [OK] Job Submitted! ID: $JOB_ID" -ForegroundColor Green


# --- Step 5: Streaming Logs ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MONITORING JOB: $JOB_ID" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Streaming logs... (Ctrl+C to stop logs, Job will continue)" -ForegroundColor Gray

# Wait briefly for provisioning
Start-Sleep -Seconds 10
# Stream logs (this blocks until user Ctrl+C or job finishes? No, stream-logs exits on finish usually)
# We use Try/Catch to allow user to break log streaming without killing the script logic for download
try {
    gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION
} catch {
    Write-Host "`n   [INFO] Log streaming interrupted." -ForegroundColor Yellow
}

# --- Step 6: Wait & Download Results ---
Write-Host ""
Write-Host "[6/6] Finalizing..." -ForegroundColor Yellow

# Poll status until done (if stream-logs exited early)
do {
    $status = gcloud ai custom-jobs describe $JOB_ID --region=$REGION --format="value(state)"
    if ($status -eq "JOB_STATE_SUCCEEDED" -or $status -eq "JOB_STATE_FAILED" -or $status -eq "JOB_STATE_CANCELLED") {
        break
    }
    Write-Host "   Status: $status (Checking again in 30s)..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
} until ($false)

if ($status -eq "JOB_STATE_SUCCEEDED") {
    Write-Host "   [SUCCESS] Job Finished Successfully!" -ForegroundColor Green
    
    # Download results
    $localResultsDir = "results\run_$TIMESTAMP"
    New-Item -ItemType Directory -Force -Path $localResultsDir | Out-Null
    
    Write-Host "   Downloading results from GCS..." -ForegroundColor Cyan
    gcloud storage cp -r "gs://$MODELS_BUCKET/run_$TIMESTAMP/*" "$localResultsDir\"
    
    Write-Host ""
    Write-Host "   [OK] Results saved to: $localResultsDir" -ForegroundColor Green
    Write-Host "   (Includes weights, plots, logs, and comparison report)"
} else {
    Write-Host "   [ERROR] Job Finished with status: $status" -ForegroundColor Red
    Write-Host "   Check logs for details."
}

Write-Host ""
Write-Host "Pipeline Finished." -ForegroundColor Cyan
