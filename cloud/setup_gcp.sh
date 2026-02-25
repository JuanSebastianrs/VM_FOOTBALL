#!/bin/bash
# ============================================================
# VM_FOOTBALL - GCP Setup Script
# ============================================================
# Run this ONCE to set up all GCP resources.
# Prerequisites: gcloud CLI installed and authenticated
# Usage: bash cloud/setup_gcp.sh
# ============================================================

set -e

# --- Configuration (edit if needed) ---
PROJECT_ID="vm-football-tesis"
REGION="us-central1"
ZONE="us-central1-a"
DATA_BUCKET="vm-football-data"
MODELS_BUCKET="vm-football-models"
VM_NAME="vm-football-training"
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
DISK_SIZE="200"

echo "============================================"
echo "VM_FOOTBALL - GCP Setup"
echo "============================================"

# --- Step 0: Verify gcloud ---
if ! command -v gcloud &> /dev/null; then
    echo "[ERROR] gcloud CLI not found."
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# --- Step 1: Set project ---
echo ""
echo "[1/6] Setting project: $PROJECT_ID"
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

# --- Step 2: Enable APIs ---
echo ""
echo "[2/6] Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
echo "   [OK] APIs enabled"

# --- Step 3: Create GCS buckets ---
echo ""
echo "[3/6] Creating GCS buckets..."

if gsutil ls -b gs://$DATA_BUCKET 2>/dev/null; then
    echo "   Bucket gs://$DATA_BUCKET already exists"
else
    gsutil mb -l $REGION gs://$DATA_BUCKET
    echo "   [OK] Created gs://$DATA_BUCKET"
fi

if gsutil ls -b gs://$MODELS_BUCKET 2>/dev/null; then
    echo "   Bucket gs://$MODELS_BUCKET already exists"
else
    gsutil mb -l $REGION gs://$MODELS_BUCKET
    echo "   [OK] Created gs://$MODELS_BUCKET"
fi

# --- Step 4: Upload dataset ---
echo ""
echo "[4/6] Uploading dataset to GCS..."
echo "   This may take a while for ~17GB..."
echo ""

DATASET_LOCAL="datasets/reorganized_dataset"
if [ -d "$DATASET_LOCAL" ]; then
    gsutil -m cp -r $DATASET_LOCAL gs://$DATA_BUCKET/
    echo "   [OK] Dataset uploaded to gs://$DATA_BUCKET/reorganized_dataset/"
else
    echo "   [WARN] Local dataset not found at $DATASET_LOCAL"
    echo "   Upload manually: gsutil -m cp -r <path> gs://$DATA_BUCKET/"
fi

# --- Step 5: Create VM with GPU ---
echo ""
echo "[5/6] Creating training VM..."

gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=${DISK_SIZE}GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

echo "   [OK] VM created: $VM_NAME"

# --- Step 6: Print connection info ---
echo ""
echo "============================================"
echo "SETUP COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Connect to VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "2. On the VM, clone your repo and download dataset:"
echo "   git clone <your-repo-url> /workspace/VM_FOOTBALL"
echo "   cd /workspace/VM_FOOTBALL"
echo "   gsutil -m cp -r gs://$DATA_BUCKET/reorganized_dataset/ datasets/"
echo ""
echo "3. Install deps and run pipeline:"
echo "   pip install -r cloud/requirements-cloud.txt"
echo "   python cloud/run_pipeline.py --config cloud/config.yaml"
echo ""
echo "4. When done, STOP the VM to save credits:"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "5. To DELETE the VM entirely:"
echo "   gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
