#!/bin/bash
set -e

echo "============================================================"
echo " SAM2 INFERENCE: Test Sequence"
echo " $(date)"
echo "============================================================"

# GPU check
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU")
    echo "[OK] GPU: $GPU_INFO"
else
    echo "[WARN] No GPU detected, running on CPU"
fi

# Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip install -q google-cloud-storage opencv-python-headless numpy 2>/dev/null

# Install SAM2 from GitHub
pip install -q git+https://github.com/facebookresearch/sam2.git 2>/dev/null
echo "[OK] Dependencies installed"

# Download scripts
echo ""
echo "[2/3] Downloading inference script..."
gcloud storage cp gs://vm-football-data/code/segmentation/inference_sam2.py ./inference_sam2.py
ls -la inference_sam2.py

# Run inference
echo ""
echo "[3/3] Running inference..."
export PYTHONWARNINGS="ignore::UserWarning"
export MKL_SERVICE_FORCE_INTEL=1

python inference_sam2.py \
    --gcs_bucket vm-football-data \
    --sequence SNMOT-143 \
    --gcs_test_prefix test_sequences \
    --checkpoint_gcs_path models/sam2_ball_finetuned/best_model.pt \
    --output_gcs_prefix models/sam2_ball_finetuned/inference_videos \
    --fps 25

echo ""
echo "============================================================"
echo " INFERENCE COMPLETE - $(date)"
echo "============================================================"
