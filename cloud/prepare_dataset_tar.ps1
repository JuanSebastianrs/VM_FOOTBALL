# ============================================================
# Prepare Dataset Tar for Fast Cloud Training
# ============================================================
# Run this ONCE to compress the dataset into a single tar file.
# This makes Vertex AI jobs download 1 file instead of 159K+.
#
# Strategy: Download to SSD -> Tar locally -> Upload 1 file
# (FUSE tar is too slow: 159K files over network = 2+ hours)
#
# Cost: ~$0.10-0.30 (n1-standard-8 CPU-only, ~15-30 min)
#
# Usage:
#   .\cloud\prepare_dataset_tar.ps1
# ============================================================

$BUCKET = "vm-football-data"
$DATASET_DIR = "reorganized_dataset"
$TAR_NAME = "reorganized_dataset.tar.gz"
$GCS_TAR = "gs://$BUCKET/$TAR_NAME"

$PROJECT_ID = "project-ad19fdc6-8493-43e5-b82"
$REGION = "us-central1"
$TIMESTAMP = Get-Date -Format "yyyyMMdd-HHmmss"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Prepare Dataset Tar for Cloud Training"     -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check if tar already exists in GCS
$existCheck = gcloud storage ls $GCS_TAR 2>&1
if ($LASTEXITCODE -eq 0 -and $existCheck -match $TAR_NAME) {
    Write-Host "[OK] Tar already exists: $GCS_TAR" -ForegroundColor Green
    gcloud storage ls -l $GCS_TAR
    Write-Host ""
    Write-Host "To recreate, first delete it:"
    Write-Host "  gcloud storage rm $GCS_TAR"
    exit 0
}

Write-Host "[INFO] Creating tar via Vertex AI worker (CPU-only, no GPU)..." -ForegroundColor Yellow
Write-Host "  Strategy: Download to SSD -> Tar locally -> Upload" -ForegroundColor Gray

# Build the tar script with single-quote here-string (no PS interpolation)
$tarScript = @'
#!/bin/bash
set -euo pipefail

echo "============================================"
echo " Dataset Tar Preparation"
echo "============================================"
echo "Started at: $(date)"
START_TIME=$(date +%s)

BUCKET="PLACEHOLDER_BUCKET"
DATASET="PLACEHOLDER_DATASET"
TAR_NAME="PLACEHOLDER_TAR"
EXPECTED_FILES=159000  # approx from previous run

# ---- Progress helper ----
show_progress() {
    local current=$1 total=$2 start=$3 label=$4
    local pct=$((current * 100 / (total > 0 ? total : 1)))
    local elapsed=$(( $(date +%s) - start ))
    local bar_len=30
    local filled=$(( pct * bar_len / 100 ))
    local empty=$(( bar_len - filled ))
    local bar=$(printf '%0.s#' $(seq 1 $filled 2>/dev/null) ; printf '%0.s-' $(seq 1 $empty 2>/dev/null))

    # ETA
    local eta="?"
    if [ "$current" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
        local remaining=$(( (total - current) * elapsed / current ))
        local eta_min=$(( remaining / 60 ))
        local eta_sec=$(( remaining % 60 ))
        eta="${eta_min}m${eta_sec}s"
    fi

    echo "  [$bar] ${pct}% (${current}/${total}) | Elapsed: ${elapsed}s | ETA: ${eta} | $label"
}

echo ""
echo "=== Step 1/3: Download dataset to local SSD ==="
echo "  Source: gs://$BUCKET/$DATASET"
echo "  Target: /workspace/temp_dataset"
echo "  Expected: ~$EXPECTED_FILES files"
echo ""

mkdir -p /workspace/temp_dataset
DL_START=$(date +%s)

# Start download in background
gcloud storage cp -r "gs://$BUCKET/$DATASET/*" /workspace/temp_dataset/ &
DL_PID=$!

# Monitor progress every 10 seconds
while kill -0 $DL_PID 2>/dev/null; do
    CURRENT=$(find /workspace/temp_dataset -type f 2>/dev/null | wc -l)
    show_progress "$CURRENT" "$EXPECTED_FILES" "$DL_START" "downloading"
    sleep 10
done

# Wait for download to finish and check exit code
wait $DL_PID
DL_EXIT=$?
if [ $DL_EXIT -ne 0 ]; then
    echo "[ERROR] Download failed with exit code $DL_EXIT"
    exit 1
fi

TOTAL=$(find /workspace/temp_dataset -type f | wc -l)
DL_ELAPSED=$(( $(date +%s) - DL_START ))
DL_SPEED=$(( TOTAL / (DL_ELAPSED > 0 ? DL_ELAPSED : 1) ))
DATASET_SIZE=$(du -sh /workspace/temp_dataset | cut -f1)
echo ""
echo "[OK] Step 1 DONE: Downloaded $TOTAL files ($DATASET_SIZE) in ${DL_ELAPSED}s (~${DL_SPEED} files/s)"

echo ""
echo "=== Step 2/3: Compress locally on NVMe SSD ==="
echo "  Source: /workspace/temp_dataset -> /workspace/$DATASET"
echo "  Target: /workspace/$TAR_NAME"
echo ""

# Rename to correct folder name before tar
mv /workspace/temp_dataset "/workspace/$DATASET"

TAR_START=$(date +%s)

# Start tar in background
tar -czf "/workspace/$TAR_NAME" -C /workspace "$DATASET" &
TAR_PID=$!

# Monitor tar size growth every 15 seconds
while kill -0 $TAR_PID 2>/dev/null; do
    if [ -f "/workspace/$TAR_NAME" ]; then
        TAR_CURRENT=$(du -m "/workspace/$TAR_NAME" 2>/dev/null | cut -f1)
        TAR_ELAPSED=$(( $(date +%s) - TAR_START ))
        TAR_SPEED=$(( TAR_CURRENT / (TAR_ELAPSED > 0 ? TAR_ELAPSED : 1) ))
        echo "  [TAR] ${TAR_CURRENT} MB written | ${TAR_ELAPSED}s elapsed | ~${TAR_SPEED} MB/s"
    fi
    sleep 15
done

wait $TAR_PID
TAR_EXIT=$?
if [ $TAR_EXIT -ne 0 ]; then
    echo "[ERROR] Tar failed with exit code $TAR_EXIT"
    exit 1
fi

TAR_ELAPSED=$(( $(date +%s) - TAR_START ))
TAR_SIZE=$(du -h "/workspace/$TAR_NAME" | cut -f1)
TAR_SIZE_MB=$(du -m "/workspace/$TAR_NAME" | cut -f1)
echo ""
echo "[OK] Step 2 DONE: Tar created ($TAR_SIZE) in ${TAR_ELAPSED}s"

echo ""
echo "=== Step 3/3: Upload tar to GCS ==="
echo "  Source: /workspace/$TAR_NAME ($TAR_SIZE)"
echo "  Target: gs://$BUCKET/$TAR_NAME"
echo ""

UP_START=$(date +%s)
gcloud storage cp "/workspace/$TAR_NAME" "gs://$BUCKET/$TAR_NAME"
UP_ELAPSED=$(( $(date +%s) - UP_START ))
UP_SPEED=$(( TAR_SIZE_MB / (UP_ELAPSED > 0 ? UP_ELAPSED : 1) ))
echo ""
echo "[OK] Step 3 DONE: Uploaded in ${UP_ELAPSED}s (~${UP_SPEED} MB/s)"

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
echo ""
echo "============================================"
echo " ALL DONE!"
echo "============================================"
echo "  Total files:    $TOTAL"
echo "  Dataset size:   $DATASET_SIZE"
echo "  Tar size:       $TAR_SIZE"
echo "  Total time:     ${TOTAL_MIN} min (${TOTAL_ELAPSED}s)"
echo ""
echo "  Step 1 (download): ${DL_ELAPSED}s"
echo "  Step 2 (tar):      ${TAR_ELAPSED}s"
echo "  Step 3 (upload):   ${UP_ELAPSED}s"
echo ""
gcloud storage ls -l "gs://$BUCKET/$TAR_NAME"
echo "Finished at: $(date)"
'@

$tarScript = $tarScript.Replace("PLACEHOLDER_BUCKET", $BUCKET)
$tarScript = $tarScript.Replace("PLACEHOLDER_DATASET", $DATASET_DIR)
$tarScript = $tarScript.Replace("PLACEHOLDER_TAR", $TAR_NAME)

# Save and upload
$scriptPath = [System.IO.Path]::Combine($env:TEMP, "tar_dataset.sh")
$tarScript | Out-File -FilePath $scriptPath -Encoding utf8 -NoNewline
(Get-Content $scriptPath -Raw).Replace("`r`n", "`n") | Set-Content -Path $scriptPath -NoNewline
gcloud storage cp $scriptPath "gs://$BUCKET/code/tar_dataset.sh"
Remove-Item $scriptPath -ErrorAction SilentlyContinue

# Job spec: CPU-only, n1-standard-8 (more CPU = faster tar), big disk
$jobSpec = @"
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 500
    containerSpec:
      imageUri: us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-14.py310:latest
      command:
        - bash
        - -c
        - |
          gcloud storage cp gs://$BUCKET/code/tar_dataset.sh /tmp/tar_dataset.sh
          chmod +x /tmp/tar_dataset.sh
          bash /tmp/tar_dataset.sh
scheduling:
  timeout: 7200s
"@

$yamlPath = [System.IO.Path]::Combine($env:TEMP, "tar_job_spec.yaml")
$jobSpec | Out-File -FilePath $yamlPath -Encoding utf8

Write-Host "Submitting tar job (n1-standard-8, CPU-only)..." -ForegroundColor Yellow

$submitOutput = gcloud ai custom-jobs create `
    --region=$REGION `
    --display-name="prepare-dataset-tar-$TIMESTAMP" `
    --config=$yamlPath `
    --format="value(name)" 2>&1

$JOB_ID = $submitOutput -replace "projects/.*/customJobs/", ""
Remove-Item $yamlPath -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Job submission failed: $submitOutput" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Tar job submitted: $JOB_ID" -ForegroundColor Green
Write-Host "  Machine: n1-standard-8 (8 vCPUs, 30 GB RAM, no GPU)" -ForegroundColor Gray
Write-Host "  Estimated time: 15-30 min" -ForegroundColor Gray
Write-Host "  Estimated cost: ~$0.10-0.30" -ForegroundColor Gray
Write-Host ""
Write-Host "Streaming logs..." -ForegroundColor Gray

Start-Sleep -Seconds 10
try {
    gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION
} catch {
    Write-Host "`n[INFO] Log streaming interrupted." -ForegroundColor Yellow
    Write-Host "To resume: gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION"
}

# Check result
$status = gcloud ai custom-jobs describe $JOB_ID --region=$REGION --format="value(state)"
if ($status -eq "JOB_STATE_SUCCEEDED") {
    Write-Host ""
    Write-Host "[OK] Tar ready: $GCS_TAR" -ForegroundColor Green
    gcloud storage ls -l $GCS_TAR
    Write-Host ""
    Write-Host "Now run the training job:" -ForegroundColor Cyan
    Write-Host "  .\cloud\submit_ball_job.ps1 -SmokeTest -GPU T4"
} else {
    Write-Host "[ERROR] Job status: $status" -ForegroundColor Red
    Write-Host "Check logs: gcloud ai custom-jobs stream-logs $JOB_ID --region=$REGION"
}
