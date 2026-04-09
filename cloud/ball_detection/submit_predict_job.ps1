# ============================================================
# SAM2 Data Generator - Vertex AI Custom Job Launcher
# ============================================================
# Uses pre-built PyTorch container (same as YOLO training job).
# No Docker build needed — installs deps at startup.
#
# Usage:
#   .\cloud\ball_detection\submit_predict_job.ps1             # POC (test_seq_116)
#   .\cloud\ball_detection\submit_predict_job.ps1 -Full       # Full dataset
# ============================================================

param(
    [switch]$Full
)

$PROJECT_ID = "project-ad19fdc6-8493-43e5-b82"
$REGION = "us-central1"
$BUCKET = "vm-football-data"
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"

# Pre-built PyTorch 2.2 container (same as training job - proven to work)
$CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " SAM2 Data Generator - Vertex AI Job"       -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

if ($Full) {
    Write-Host "  Mode: FULL DATASET" -ForegroundColor Yellow
    $JOB_NAME = "sam2-datagen-full-$TIMESTAMP"
    $TIMEOUT = "86400s"
} else {
    Write-Host "  Mode: POC (test_seq_116)" -ForegroundColor Green
    $JOB_NAME = "sam2-datagen-poc-$TIMESTAMP"
    $TIMEOUT = "3600s"
}
Write-Host ""

# ---- Step 1: Upload predict_cloud.py to GCS ----
Write-Host "[1/3] Uploading inference code to GCS..." -ForegroundColor Yellow
gcloud storage cp "cloud/ball_detection/predict_cloud.py" "gs://$BUCKET/code/ball_detection/"
Write-Host "  [OK] Code uploaded" -ForegroundColor Green

# ---- Step 2: Verify model exists in GCS ----
Write-Host ""
Write-Host "[2/3] Verifying GCS data..." -ForegroundColor Yellow
$modelExists = gcloud storage ls "gs://$BUCKET/models/yolo_ball_data_centric.pt" 2>$null
if (-not $modelExists) {
    Write-Host "  Uploading model to GCS..." -ForegroundColor Yellow
    gcloud storage cp "models/yolo_ball_data_centric.pt" "gs://$BUCKET/models/"
}
Write-Host "  [OK] Model verified" -ForegroundColor Green

if (-not $Full) {
    $seqExists = gcloud storage ls "gs://$BUCKET/datasets/test_seq_116/SNMOT-116_000001.jpg" 2>$null
    if (-not $seqExists) {
        Write-Host "  Uploading test_seq_116 to GCS (~50MB)..." -ForegroundColor Yellow
        gcloud storage cp -r "datasets/test_seq_116" "gs://$BUCKET/datasets/"
    }
    Write-Host "  [OK] Test sequence verified" -ForegroundColor Green
}

# ---- Step 3: Build startup script & submit ----
Write-Host ""
Write-Host "[3/3] Submitting Custom Job: $JOB_NAME" -ForegroundColor Yellow

# Build the args for predict_cloud.py
if ($Full) {
    $PREDICT_ARGS = "--gcs_bucket=$BUCKET --gcs_model_path=models/yolo_ball_data_centric.pt --gcs_train_prefix=reorganized_dataset/images/train/ --gcs_output_base=data_generation/sam2_inputs --mode=full"
} else {
    $PREDICT_ARGS = "--gcs_bucket=$BUCKET --gcs_model_path=models/yolo_ball_data_centric.pt --gcs_sequence_dir=datasets/test_seq_116/ --gcs_output_base=data_generation/sam2_inputs --mode=poc"
}

# Startup script: install deps + run predict_cloud.py
# Uses single-quote here-string to prevent PowerShell variable interpolation
$startupScript = @'
#!/bin/bash
set -euo pipefail

echo '======================================'
echo ' SAM2 Data Generator - Vertex AI'
echo '======================================'
echo "Started at: $(date)"

# ---- Phase 0: Environment ----
echo '[0/4] Preparing environment...'
pip uninstall -y torch_xla 2>/dev/null || true
export PJRT_DEVICE=CPU
export MKL_SERVICE_FORCE_INTEL=1
export PYTHONWARNINGS=ignore::UserWarning

# ---- Phase 1: GPU Check ----
echo '[1/4] GPU Sanity Check...'
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('[FATAL] No GPU detected!')
    sys.exit(1)
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {name} ({mem:.1f} GB)')
x = torch.randn(64, 64, device='cuda', requires_grad=True)
y = x @ x.t()
loss = y.sum()
loss.backward()
del x, y, loss
torch.cuda.empty_cache()
print('[OK] GPU forward+backward pass verified')
"
echo '[OK] GPU ready'

# ---- Phase 2: Install Dependencies ----
echo '[2/4] Installing dependencies...'
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"
pip install -q --no-cache-dir ultralytics>=8.0.0 google-cloud-storage lapx
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"
pip install -q --force-reinstall --no-deps "opencv-python-headless>=4.8.0"
echo '[OK] Dependencies installed'
pip list 2>/dev/null | grep -iE "ultralytics|torch|numpy|opencv" || true

# ---- Phase 3: Download Script ----
echo '[3/4] Downloading inference script...'
mkdir -p /workspace/inference
cd /workspace/inference
gcloud storage cp gs://PLACEHOLDER_BUCKET/code/ball_detection/predict_cloud.py .
echo '[OK] Script downloaded'

# ---- Phase 4: Run Inference ----
echo '[4/4] Starting SAM2 Data Generation...'
python3 -u predict_cloud.py PLACEHOLDER_ARGS
INFERENCE_EXIT=$?

if [ $INFERENCE_EXIT -ne 0 ]; then
    echo "[ERROR] Inference exited with code $INFERENCE_EXIT"
    exit $INFERENCE_EXIT
fi

echo ''
echo '======================================'
echo ' JOB COMPLETE - Shutting down'
echo '======================================'
'@

# Replace placeholders with PowerShell variables
$startupScript = $startupScript.Replace("PLACEHOLDER_BUCKET", $BUCKET)
$startupScript = $startupScript.Replace("PLACEHOLDER_ARGS", $PREDICT_ARGS)

# Save startup script and upload to GCS
$scriptPath = [System.IO.Path]::Combine($env:TEMP, "predict_startup.sh")
$startupScript | Out-File -FilePath $scriptPath -Encoding utf8 -NoNewline
(Get-Content $scriptPath -Raw).Replace("`r`n", "`n") | Set-Content -Path $scriptPath -NoNewline
gcloud storage cp $scriptPath "gs://$BUCKET/code/ball_detection/predict_startup.sh"
Remove-Item $scriptPath -ErrorAction SilentlyContinue

# Generate YAML job spec
$jobSpecYaml = @"
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-4
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 100
    containerSpec:
      imageUri: $CONTAINER_URI
      command:
        - bash
        - -c
        - |
          gcloud storage cp gs://$BUCKET/code/ball_detection/predict_startup.sh /tmp/startup.sh
          chmod +x /tmp/startup.sh
          bash /tmp/startup.sh
scheduling:
  timeout: $TIMEOUT
"@

$yamlPath = [System.IO.Path]::Combine($env:TEMP, "predict_job_spec.yaml")
$jobSpecYaml | Out-File -FilePath $yamlPath -Encoding utf8

Write-Host ""
Write-Host "  Job Spec:" -ForegroundColor DarkGray
Get-Content $yamlPath | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
Write-Host ""

# Submit the Custom Job
$submitOutput = gcloud ai custom-jobs create `
    --region=$REGION `
    --display-name=$JOB_NAME `
    --config=$yamlPath `
    --format="value(name)" 2>&1

$JOB_ID = $submitOutput -replace "projects/.*/customJobs/", ""
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($JOB_ID)) {
    Write-Host "  [ERROR] Job submission failed:" -ForegroundColor Red
    Write-Host $submitOutput
    Remove-Item $yamlPath -ErrorAction SilentlyContinue
    exit 1
}

Remove-Item $yamlPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " JOB SUBMITTED: $JOB_NAME" -ForegroundColor Green
Write-Host " JOB ID: $JOB_ID" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Stream logs:" -ForegroundColor Cyan
Write-Host "  gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION"
Write-Host ""
Write-Host "Check status:" -ForegroundColor Cyan
Write-Host "  gcloud ai custom-jobs describe $JOB_ID --region=$REGION --format='value(state)'"
Write-Host ""
Write-Host "Check results (after completion):" -ForegroundColor Cyan
Write-Host "  gcloud storage ls gs://$BUCKET/data_generation/sam2_inputs/"
Write-Host ""

# Stream logs
Write-Host "Streaming logs... (Ctrl+C to stop, job continues)" -ForegroundColor Gray
Start-Sleep -Seconds 15
try {
    gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION
} catch {
    Write-Host "`n  [INFO] Log streaming interrupted." -ForegroundColor Yellow
}

# Poll final status
Write-Host ""
Write-Host "[POST] Checking final job status..." -ForegroundColor Yellow
do {
    $status = gcloud ai custom-jobs describe $JOB_ID --region=$REGION --format="value(state)"
    if ($status -eq "JOB_STATE_SUCCEEDED" -or $status -eq "JOB_STATE_FAILED" -or $status -eq "JOB_STATE_CANCELLED") {
        break
    }
    Write-Host "  Status: $status (checking in 30s)..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
} until ($false)

if ($status -eq "JOB_STATE_SUCCEEDED") {
    Write-Host "  [SUCCESS] Data Generation Complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Results at: gs://$BUCKET/data_generation/sam2_inputs/" -ForegroundColor Cyan
    gcloud storage ls "gs://$BUCKET/data_generation/sam2_inputs/"
} else {
    Write-Host "  [ERROR] Job finished with status: $status" -ForegroundColor Red
    Write-Host "  Check logs: gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION"
}

Write-Host ""
Write-Host "Done." -ForegroundColor Cyan
