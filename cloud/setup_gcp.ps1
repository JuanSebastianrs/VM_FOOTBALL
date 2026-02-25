# ============================================================
# VM_FOOTBALL - GCP Setup Script (PowerShell / Windows)
# ============================================================
# Run this ONCE to set up all GCP resources.
# Prerequisites: gcloud CLI installed and authenticated
# Usage: .\cloud\setup_gcp.ps1
# ============================================================

# --- Configuration (edit if needed) ---
$PROJECT_ID    = "project-ad19fdc6-8493-43e5-b82"
$REGION        = "us-central1"
$ZONE          = "us-central1-a"
$DATA_BUCKET   = "vm-football-data"
$MODELS_BUCKET = "vm-football-models"
$VM_NAME       = "vm-football-training"
$MACHINE_TYPE  = "n1-standard-8"
$GPU_TYPE      = "nvidia-tesla-t4"
$DISK_SIZE     = "200"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " VM_FOOTBALL - GCP Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- Step 1: Set project ---
Write-Host ""
Write-Host "[1/6] Setting project: $PROJECT_ID" -ForegroundColor Yellow
gcloud config set project $PROJECT_ID 2>$null
gcloud config set compute/region $REGION 2>$null
gcloud config set compute/zone $ZONE 2>$null
Write-Host "   [OK] Project configured" -ForegroundColor Green

# --- Step 2: Enable APIs ---
Write-Host ""
Write-Host "[2/6] Enabling required APIs..." -ForegroundColor Yellow
gcloud services enable compute.googleapis.com 2>$null
gcloud services enable storage.googleapis.com 2>$null
Write-Host "   [OK] APIs enabled" -ForegroundColor Green

# --- Step 3: Create GCS buckets (using gcloud storage instead of gsutil) ---
Write-Host ""
Write-Host "[3/6] Creating GCS buckets..." -ForegroundColor Yellow

$existingBuckets = gcloud storage buckets list --format="value(name)" 2>$null

if ($existingBuckets -match $DATA_BUCKET) {
    Write-Host "   Bucket gs://$DATA_BUCKET already exists" -ForegroundColor Gray
} else {
    gcloud storage buckets create "gs://$DATA_BUCKET" --location=$REGION 2>$null
    Write-Host "   [OK] Created gs://$DATA_BUCKET" -ForegroundColor Green
}

if ($existingBuckets -match $MODELS_BUCKET) {
    Write-Host "   Bucket gs://$MODELS_BUCKET already exists" -ForegroundColor Gray
} else {
    gcloud storage buckets create "gs://$MODELS_BUCKET" --location=$REGION 2>$null
    Write-Host "   [OK] Created gs://$MODELS_BUCKET" -ForegroundColor Green
}

# --- Step 4: Upload dataset ---
Write-Host ""
Write-Host "[4/6] Uploading dataset to GCS..." -ForegroundColor Yellow
Write-Host "   This will take a while for ~17GB..." -ForegroundColor Gray

$DATASET_LOCAL = "datasets\reorganized_dataset"
if (Test-Path $DATASET_LOCAL) {
    Write-Host "   Starting parallel upload..." -ForegroundColor Gray
    gcloud storage cp -r $DATASET_LOCAL "gs://$DATA_BUCKET/" --verbosity=warning
    Write-Host "   [OK] Dataset uploaded to gs://$DATA_BUCKET/reorganized_dataset/" -ForegroundColor Green
} else {
    Write-Host "   [WARN] Local dataset not found at $DATASET_LOCAL" -ForegroundColor Yellow
    Write-Host "   Upload manually:" -ForegroundColor Yellow
    Write-Host "   gcloud storage cp -r <path> gs://$DATA_BUCKET/" -ForegroundColor Gray
}

# --- Step 5: Create VM with GPU ---
Write-Host ""
Write-Host "[5/6] Creating training VM..." -ForegroundColor Yellow

gcloud compute instances create $VM_NAME `
    --zone=$ZONE `
    --machine-type=$MACHINE_TYPE `
    --accelerator="type=$GPU_TYPE,count=1" `
    --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 `
    --image-project=deeplearning-platform-release `
    --boot-disk-size="${DISK_SIZE}GB" `
    --boot-disk-type=pd-ssd `
    --maintenance-policy=TERMINATE `
    --metadata="install-nvidia-driver=True" `
    --scopes=cloud-platform

Write-Host "   [OK] VM created: $VM_NAME" -ForegroundColor Green

# --- Step 6: Print connection info ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " SETUP COMPLETE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host ""
Write-Host "1. Connect to VM:" -ForegroundColor White
Write-Host "   gcloud compute ssh $VM_NAME --zone=$ZONE" -ForegroundColor Gray
Write-Host ""
Write-Host "2. On the VM, download dataset from GCS:" -ForegroundColor White
Write-Host "   mkdir -p /workspace/datasets" -ForegroundColor Gray
Write-Host "   gcloud storage cp -r gs://$DATA_BUCKET/reorganized_dataset/ /workspace/datasets/" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Install deps and run pipeline:" -ForegroundColor White
Write-Host "   pip install -r cloud/requirements-cloud.txt" -ForegroundColor Gray
Write-Host "   python cloud/run_pipeline.py --config cloud/config.yaml" -ForegroundColor Gray
Write-Host ""
Write-Host "4. When done, STOP the VM to save credits:" -ForegroundColor White
Write-Host "   gcloud compute instances stop $VM_NAME --zone=$ZONE" -ForegroundColor Gray
Write-Host ""
