# ============================================================
# Jersey Number Recognition - Vertex AI Job Launcher
# ============================================================
# Trains jersey number classifiers (processed vs full) on sn-jersey.
#
# Usage:
#   .\cloud\submit_jersey_job.ps1                 # T4 GPU (default)
#   .\cloud\submit_jersey_job.ps1 -GPU L4         # L4 GPU (if available)
#   .\cloud\submit_jersey_job.ps1 -SmokeTest      # 2-epoch smoketest
#   .\cloud\submit_jersey_job.ps1 -DryRun         # Config check only
# ============================================================

param(
    [ValidateSet("L4", "T4")]
    [string]$GPU = "T4",
    [string]$ConfigFile = "config.yaml",
    [string]$Region = "us-central1",
    [switch]$AutoRegion,
    [switch]$OnDemand,
    [switch]$DryRun,
    [switch]$SmokeTest
)

$PROJECT_ID = "vm-football-489116"
$REGION = $Region
$DATA_BUCKET = "vm-football-data"
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"
$JOB_NAME = "jersey-number-$($GPU.ToLower())-$TIMESTAMP"

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
Write-Host " Jersey Number Recognition - Vertex AI Job" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  GPU:       $GPU ($ACCELERATOR_TYPE)"
Write-Host "  Machine:   $MACHINE_TYPE"
Write-Host "  Region:    $REGION"
Write-Host "  Container: pytorch-gpu.2-2.py310"
Write-Host "  OnDemand:  $($OnDemand.IsPresent)"
Write-Host "  AutoRegion:$($AutoRegion.IsPresent)"
Write-Host "  SmokeTest: $($SmokeTest.IsPresent)"
Write-Host ""

$SCHEDULING_STRATEGY = if ($OnDemand) { "STANDARD" } else { "SPOT" }

# --- Step 1: Validate ---
Write-Host "[1/5] Validating Environment..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID 2>$null
gcloud services enable aiplatform.googleapis.com 2>$null
gcloud services enable storage.googleapis.com 2>$null
Write-Host "   [OK] APIs validated" -ForegroundColor Green

# --- Step 2: Upload Training Code ---
Write-Host ""
Write-Host "[2/5] Uploading Training Code..." -ForegroundColor Yellow

gcloud storage cp -r "cloud/jersey_number" "gs://$DATA_BUCKET/code/"
Write-Host "   [OK] Code uploaded to gs://$DATA_BUCKET/code/jersey_number/" -ForegroundColor Green

# --- Dry Run ---
if ($DryRun) {
    Write-Host ""
    Write-Host "[DRY RUN] Configuration validated. No job submitted." -ForegroundColor Green
    Write-Host "  Would create job: $JOB_NAME"
    Write-Host "  GPU: $ACCELERATOR_TYPE on $MACHINE_TYPE"
    Write-Host "  Container: $CONTAINER_URI"
    Write-Host "  Config: cloud/jersey_number/$ConfigFile"
    exit 0
}

# --- Step 3: Configure Startup Script ---
Write-Host ""
Write-Host "[3/5] Configuring Job..." -ForegroundColor Yellow

$startupScript = @'
#!/bin/bash
set -euo pipefail

# ============================================================
# Trap: upload crash log on ANY failure
# ============================================================
GCS_OUTPUT="gs://PLACEHOLDER_BUCKET/models/jersey_number"
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
        pip list 2>/dev/null | grep -iE "torch|numpy|opencv|pillow|yaml" || true
    } > "$OUTPUT_DIR/CRASH_LOG.txt"

    gsutil -m cp -r "$OUTPUT_DIR/*" "$GCS_OUTPUT/" 2>/dev/null || true
    echo "[OK] Crash artifacts uploaded to $GCS_OUTPUT"
}

trap cleanup_on_error ERR

echo '======================================'
echo ' Jersey Number Recognition - Vertex AI'
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
mkdir -p /workspace/jersey_number
cd /workspace/jersey_number
gcloud storage cp -r gs://PLACEHOLDER_BUCKET/code/jersey_number/* .

echo "Files downloaded:"
ls -la
echo ""

for required_file in train_cloud.py config.yaml requirements.txt preprocess_sn_jersey.py train_sn_jersey.py; do
    if [ ! -f "$required_file" ]; then
        echo "[FATAL] Missing required file: $required_file"
        ls -la
        exit 1
    fi
done
echo "[OK] All required files present"

# ---- Phase 3: Install Dependencies ----
echo '[3/5] Installing dependencies...'

pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"
pip install -q --force-reinstall "pydantic>=2.5.0"
pip install -q --no-cache-dir -r requirements.txt
pip install -q --force-reinstall "opencv-python-headless>=4.8.0" --no-deps
pip install -q "python-json-logger<3" 2>/dev/null || pip install -q python-json-logger 2>/dev/null || true

python3 -c "import numpy, torch; print(f'[OK] numpy={numpy.__version__}, torch={torch.__version__}')"

echo '[OK] Dependencies installed'

PLACEHOLDER_SMOKE_OVERRIDE

# ---- Phase 4: Set GCS env vars ----
echo '[4/5] Configuring GCS paths...'
export GCS_OUTPUT_PATH=gs://PLACEHOLDER_BUCKET/models/jersey_number
export VERTEX_AI_JOB=true

# ---- Phase 5: Train ----
echo '[5/5] Starting Jersey Number Training...'
echo "Working directory: $(pwd)"
echo "Python: $(python3 --version)"

PLACEHOLDER_TRAIN_COMMAND
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

$startupScript = $startupScript.Replace("PLACEHOLDER_BUCKET", $DATA_BUCKET)
$startupScript = $startupScript.Replace("PLACEHOLDER_GPU_INFO", "$GPU ($ACCELERATOR_TYPE)")

if ($SmokeTest -and $ConfigFile -eq "config.yaml") {
    $smokeBlock = @'
# SMOKE TEST: Use smoketest config (2 epochs)
echo "[SMOKE] Switching to config_smoketest.yaml..."
if [ -f "config_smoketest.yaml" ]; then
    cp config_smoketest.yaml config.yaml
    echo "[SMOKE] Config replaced with smoketest version"
    echo "[SMOKE] Contents:"
    cat config.yaml
else
    echo "[FATAL] config_smoketest.yaml not found!"
    ls -la
    exit 1
fi
'@
    $startupScript = $startupScript.Replace("PLACEHOLDER_SMOKE_OVERRIDE", $smokeBlock)
}
else {
    $startupScript = $startupScript.Replace("PLACEHOLDER_SMOKE_OVERRIDE", "# No smoke test override")
}

$trainCommand = "python3 -u train_cloud.py --config $ConfigFile"
if ($SmokeTest) {
    $trainCommand = "$trainCommand --smoketest"
}
$startupScript = $startupScript.Replace("PLACEHOLDER_TRAIN_COMMAND", $trainCommand)

$scriptPath = [System.IO.Path]::Combine($env:TEMP, "jersey_startup.sh")
$startupScript | Out-File -FilePath $scriptPath -Encoding utf8 -NoNewline
(Get-Content $scriptPath -Raw).Replace("`r`n", "`n") | Set-Content -Path $scriptPath -NoNewline
gcloud storage cp $scriptPath "gs://$DATA_BUCKET/code/jersey_number/startup.sh" 2>$null
Remove-Item $scriptPath -ErrorAction SilentlyContinue

Write-Host "   [OK] Job configured (with error trapping and crash upload)" -ForegroundColor Green

# --- Step 4: Submit Job ---
Write-Host ""
Write-Host "[4/5] Submitting Job to Vertex AI..." -ForegroundColor Yellow

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
          gcloud storage cp gs://$DATA_BUCKET/code/jersey_number/startup.sh /tmp/startup.sh
          chmod +x /tmp/startup.sh
          bash /tmp/startup.sh
scheduling:
  strategy: $SCHEDULING_STRATEGY
  timeout: 432000s
  restartJobOnWorkerRestart: true
"@

$yamlPath = [System.IO.Path]::Combine($env:TEMP, "jersey_job_spec.yaml")
$jobSpecYaml | Out-File -FilePath $yamlPath -Encoding utf8

$regionsToTry = @($REGION)
if ($AutoRegion) {
    $fallbackRegions = @("us-east1", "us-west1", "northamerica-northeast1")
    foreach ($r in $fallbackRegions) {
        if ($r -ne $REGION) {
            $regionsToTry += $r
        }
    }
}

$JOB_ID = ""
$lastError = ""
$usedRegion = ""
foreach ($tryRegion in $regionsToTry) {
    Write-Host "   [INFO] Trying region: $tryRegion" -ForegroundColor DarkGray
    try {
        $submitOutput = gcloud ai custom-jobs create `
            --region=$tryRegion `
            --display-name=$JOB_NAME `
            --config=$yamlPath `
            --format="value(name)" 2>&1

        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($submitOutput)) {
            $JOB_ID = $submitOutput -replace "projects/.*/customJobs/", ""
            $usedRegion = $tryRegion
            break
        }

        $lastError = ($submitOutput | Out-String)
        if ($lastError -match "RESOURCE_EXHAUSTED|quota metrics exceed quota limits") {
            Write-Host "   [WARN] Quota exhausted in $tryRegion. Trying next region..." -ForegroundColor Yellow
            continue
        }
        break
    }
    catch {
        $lastError = $_.Exception.Message
        if ($lastError -match "RESOURCE_EXHAUSTED|quota metrics exceed quota limits") {
            Write-Host "   [WARN] Quota exhausted in $tryRegion. Trying next region..." -ForegroundColor Yellow
            continue
        }
        break
    }
}

if ([string]::IsNullOrWhiteSpace($JOB_ID)) {
    Write-Host "   [ERROR] Submission Failed (No Job ID returned)." -ForegroundColor Red
    if (-not [string]::IsNullOrWhiteSpace($lastError)) {
        Write-Host $lastError
    }
    exit 1
}

Remove-Item $yamlPath -ErrorAction SilentlyContinue
Write-Host "   [OK] Job Submitted! ID: $JOB_ID" -ForegroundColor Green
Write-Host "   [INFO] Region used: $usedRegion" -ForegroundColor DarkGray
if ($OnDemand) {
    Write-Host "   [INFO] Strategy: STANDARD (on-demand)" -ForegroundColor DarkGray
}
else {
    Write-Host "   [INFO] Strategy: SPOT (preemptible)" -ForegroundColor DarkGray
    Write-Host "   [INFO] Spot: if preempted, will auto-restart and resume from checkpoint" -ForegroundColor DarkGray
}

# --- Step 5: Streaming Logs ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MONITORING: $JOB_ID" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Streaming logs... (Ctrl+C to stop, job continues)" -ForegroundColor Gray

Start-Sleep -Seconds 10
try {
    gcloud ai custom-jobs stream-logs $JOB_ID --region=$usedRegion
}
catch {
    Write-Host "`n   [INFO] Log streaming interrupted." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Cyan
