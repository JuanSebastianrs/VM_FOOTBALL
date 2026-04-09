# ============================================================
# SAM2 Pipeline Launcher - Vertex AI Custom Jobs
# ============================================================
# Step 1: Pseudo-mask generation (SAM2 zero-shot + ByteTrack boxes)
# Step 2: Fine-tuning (SAM2 hiera-small, frozen encoder)
#
# Usage:
#   .\training\segmentation\submit_sam2_job.ps1 -Step masks   # Pseudo-masks
#   .\training\segmentation\submit_sam2_job.ps1 -Step train   # Fine-tuning
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("masks", "train")]
    [string]$Step
)

$PROJECT_ID = "project-ad19fdc6-8493-43e5-b82"
$REGION = "us-central1"
$BUCKET = "vm-football-data"
$CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest"
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " SAM2 Pipeline - Step: $Step"                 -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ---- Upload scripts to GCS ----
Write-Host ""
Write-Host "[1/2] Uploading scripts to GCS..." -ForegroundColor Yellow
gcloud storage cp "training/segmentation/generate_pseudomasks.py" "gs://$BUCKET/code/segmentation/" 2>$null
gcloud storage cp "training/segmentation/finetune_sam2.py" "gs://$BUCKET/code/segmentation/" 2>$null
Write-Host "  [OK] Scripts uploaded" -ForegroundColor Green

# ---- Configure job ----
if ($Step -eq "masks") {
    $JOB_NAME = "sam2-pseudomasks-$TIMESTAMP"
    $TIMEOUT = "21600s"  # 6h max

    $SCRIPT_NAME = "generate_pseudomasks.py"
    $SCRIPT_ARGS = "--gcs_bucket=$BUCKET --gcs_frames_prefix=reorganized_dataset/images/train/ --gcs_coords_prefix=data_generation/sam2_inputs/bytetrack/ --gcs_output_prefix=data_generation/sam2_training --sam2_checkpoint=facebook/sam2.1-hiera-small --min_conf=0.20"
    $PIP_DEPS = "sam2 google-cloud-storage opencv-python-headless"

} else {
    $JOB_NAME = "sam2-finetune-$TIMESTAMP"
    $TIMEOUT = "14400s"  # 4h max

    $SCRIPT_NAME = "finetune_sam2.py"
    $SCRIPT_ARGS = "--gcs_bucket=$BUCKET --gcs_training_prefix=data_generation/sam2_training/ --gcs_output_prefix=models/sam2_ball_finetuned --sam2_checkpoint=facebook/sam2.1-hiera-small --epochs=8 --batch_size=4 --lr=1e-5"
    $PIP_DEPS = "sam2 google-cloud-storage opencv-python-headless"
}

Write-Host "  Job: $JOB_NAME" -ForegroundColor Cyan
Write-Host "  Script: $SCRIPT_NAME"
Write-Host "  Timeout: $TIMEOUT"
Write-Host ""

# Build startup script 
$startupScript = @"
#!/bin/bash
set -euo pipefail

echo '======================================'
echo ' SAM2 Pipeline: $Step'
echo '======================================'
echo "Started at: `$(date)"

# ---- Environment ----
pip uninstall -y torch_xla 2>/dev/null || true
export PJRT_DEVICE=CPU
export MKL_SERVICE_FORCE_INTEL=1
export PYTHONWARNINGS=ignore::UserWarning

# ---- GPU Check ----
echo '[1/5] GPU Check...'
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('[FATAL] No GPU!')
    sys.exit(1)
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {name} ({mem:.1f} GB)')
" || { echo '[FATAL] GPU check failed'; exit 1; }

# ---- Install Dependencies ----
echo '[2/5] Installing base dependencies...'
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"
pip install -q --no-cache-dir google-cloud-storage
pip install -q --force-reinstall --no-deps "opencv-python-headless>=4.8.0"

# ---- Install SAM2 from GitHub ----
echo '[3/5] Installing SAM2 from facebookresearch/sam2...'
cd /tmp
if [ -d "sam2" ]; then rm -rf sam2; fi
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[notebooks]" 2>&1 | tail -5
cd /tmp

# Verify SAM2 import
python3 -c "
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
print('[OK] SAM2 import successful')
" || { echo '[FATAL] SAM2 import failed!'; exit 1; }

echo '[OK] Dependencies:'
pip list 2>/dev/null | grep -iE "sam2|SAM|torch|numpy|opencv" || true

# ---- Download Script ----
echo '[4/5] Downloading script...'
mkdir -p /workspace/segmentation
cd /workspace/segmentation
gcloud storage cp gs://$BUCKET/code/segmentation/$SCRIPT_NAME .

# ---- Run ----
echo '[5/5] Running $SCRIPT_NAME...'
python3 -u $SCRIPT_NAME $SCRIPT_ARGS
EXIT_CODE=`$?

if [ `$EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Script exited with code `$EXIT_CODE"
    exit `$EXIT_CODE
fi

echo ''
echo '======================================'
echo ' JOB COMPLETE'
echo '======================================'
"@

# Save startup script and upload
$scriptPath = [System.IO.Path]::Combine($env:TEMP, "sam2_startup.sh")
$startupScript | Out-File -FilePath $scriptPath -Encoding utf8 -NoNewline
(Get-Content $scriptPath -Raw).Replace("`r`n", "`n") | Set-Content -Path $scriptPath -NoNewline
gcloud storage cp $scriptPath "gs://$BUCKET/code/segmentation/sam2_startup_${Step}.sh" 2>$null
Remove-Item $scriptPath -ErrorAction SilentlyContinue

# ---- Submit Vertex AI Job ----
Write-Host "[2/2] Submitting job..." -ForegroundColor Yellow

$jobSpecYaml = @"
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: $CONTAINER_URI
      command:
        - bash
        - -c
        - |
          gcloud storage cp gs://$BUCKET/code/segmentation/sam2_startup_${Step}.sh /tmp/startup.sh
          chmod +x /tmp/startup.sh
          bash /tmp/startup.sh
scheduling:
  timeout: $TIMEOUT
"@

$yamlPath = [System.IO.Path]::Combine($env:TEMP, "sam2_job_spec.yaml")
$jobSpecYaml | Out-File -FilePath $yamlPath -Encoding utf8

Write-Host ""
Write-Host "  Job Spec:" -ForegroundColor DarkGray
Get-Content $yamlPath | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
Write-Host ""

gcloud ai custom-jobs create `
    --region=$REGION `
    --display-name=$JOB_NAME `
    --config=$yamlPath

Remove-Item $yamlPath -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Job submission failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " JOB SUBMITTED: $JOB_NAME"                    -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Monitor:" -ForegroundColor Cyan
Write-Host "  gcloud ai custom-jobs list --region=$REGION --filter='displayName~sam2' --format='table(name,displayName,state)' --limit=5"
Write-Host ""
Write-Host "Stream logs (get JOB_ID from above):" -ForegroundColor Cyan
Write-Host "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=$REGION"
Write-Host ""

if ($Step -eq "masks") {
    Write-Host "Results:" -ForegroundColor Cyan
    Write-Host "  gcloud storage ls gs://$BUCKET/data_generation/sam2_training/"
    Write-Host ""
    Write-Host "After masks complete, run training:" -ForegroundColor Yellow
    Write-Host "  .\training\segmentation\submit_sam2_job.ps1 -Step train"
} else {
    Write-Host "Results:" -ForegroundColor Cyan
    Write-Host "  gcloud storage ls gs://$BUCKET/models/sam2_ball_finetuned/"
}
