#!/bin/bash
set -euo pipefail

echo '======================================'
echo ' SAM2 Pseudo-Mask Generator'
echo '======================================'
echo "Started at: $(date)"

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
    print('[FATAL] No GPU!'); sys.exit(1)
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {name} ({mem:.1f} GB)')
"
echo '[OK] GPU check passed'

# ---- Install Dependencies ----
echo '[2/5] Installing base deps...'
pip install -q --force-reinstall "numpy>=1.23.0,<2.0.0"
pip install -q --no-cache-dir google-cloud-storage
pip install -q --force-reinstall --no-deps "opencv-python-headless>=4.8.0"

# ---- Install SAM2 from GitHub ----
echo '[3/5] Installing SAM2 from GitHub...'
cd /tmp
rm -rf sam2 || true
git clone https://github.com/facebookresearch/sam2.git
cd /tmp/sam2
pip install -e . 2>&1 | tail -10
echo '[OK] SAM2 pip install done'

# Verify SAM2 import works
python3 -c "
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
print('[OK] SAM2 import verified')
"
echo '[OK] All dependencies installed'
pip list 2>/dev/null | grep -iE "sam-2\|sam2\|torch\|numpy\|opencv" || true

# ---- Download Script ----
echo '[4/5] Downloading generate_pseudomasks.py...'
mkdir -p /workspace/segmentation
cd /workspace/segmentation
gcloud storage cp gs://vm-football-data/code/segmentation/generate_pseudomasks.py .
ls -la generate_pseudomasks.py

# ---- Run ----
echo '[5/5] Running generate_pseudomasks.py...'
python3 -u generate_pseudomasks.py \
    --gcs_bucket=vm-football-data \
    --gcs_frames_prefix=reorganized_dataset/images/train/ \
    --gcs_coords_prefix=data_generation/sam2_inputs/bytetrack/ \
    --gcs_output_prefix=data_generation/sam2_training \
    --sam2_checkpoint=facebook/sam2.1-hiera-small \
    --min_conf=0.20

echo ''
echo '======================================'
echo ' JOB COMPLETE'
echo "Finished at: $(date)"
echo '======================================'
