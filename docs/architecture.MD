# YOLO26 Ball Detection — Architecture & Training

## Overview

SOTA ball detection model using **YOLO26 Nano** on A100 GPU.
Replaces previous YOLOv11 approach. Uses STAL instead of P2 for small objects.

## Why YOLO26

| Feature | YOLOv11 (Previous) | YOLO26 (Current) |
|---|---|---|
| Architecture | YOLO11n + custom P2 YAML | YOLO26n-p2 (built-in P2) |
| Optimizer | AdamW / SGD | **MuSGD** (Muon-inspired) |
| Loss | Standard | **ProgLoss** (curriculum) |
| Label Assignment | TAL | **STAL** (small-target-aware) |
| NMS | Post-processing required | **NMS-free** end-to-end |
| DFL Module | Present | Removed (faster export) |

## YOLO26 Key Features

### MuSGD Optimizer
Hybrid SGD with Muon-inspired updates (from LLM training breakthroughs).
Applies Muon-style updates to high-dimensional parameters (conv weights),
standard SGD for lower-dimensional parameters. Better convergence stability.

### ProgLoss (Progressive Loss)
Curriculum strategy: focuses on easy objects early in training,
gradually shifts attention to harder objects (occluded/blurry balls).

### STAL (Small-Target-Aware Label Assignment)
Label assignment optimized for small objects. Critical for balls
that appear as only ~10-30 pixels in 1920x1080 broadcast frames.

### P2 Layer (Stride 4)
Extra prediction head at stride 4 feature map. Detects objects
smaller than 32x32 pixels. Built into `yolo26n-p2.pt` model.

## Training Configuration

```
Model:      yolo26n.pt (YOLO26 Nano — STAL for small objects)
GPU:        NVIDIA A100 80GB (Vertex AI)
Epochs:     300
Batch Size: 64 (A100 saturates Tensor Cores)
Image Size: 1280px
Optimizer:  MuSGD (lr0=0.001, lrf=0.001)
Warmup:     5 epochs
Patience:   50 (early stopping)
Target:     Ball only (class 5)
```

### Augmentation Hyperparameters

```yaml
mixup: 0.0          # Disabled — destructive for small objects
copy_paste: 0.3     # Critical — balances rare ball class
mosaic: 1.0         # Multi-scene diversity
close_mosaic: 30    # Disable mosaic last 30 epochs
degrees: 0.0        # No rotation (ball is round)
scale: 0.9          # Multi-scale robustness
hsv_h: 0.015        # Subtle hue variation
hsv_s: 0.7          # Saturation variation
hsv_v: 0.4          # Brightness variation
erasing: 0.0        # Never erase (could remove ball)
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│  submit_yolo26_job.ps1                              │
│  ├── Upload code to GCS                             │
│  ├── Build startup.sh                               │
│  └── Submit Vertex AI Spot Job (A100)               │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  Vertex AI (A100 GPU VM)                            │
│  ├── startup.sh                                     │
│  │   ├── GPU check                                  │
│  │   ├── Download code from GCS                     │
│  │   ├── pip install (ultralytics>=8.3.0)           │
│  │   └── python3 -u train_yolo26.py                 │
│  └── train_yolo26.py                                │
│      ├── Phase 0: Validate imports                  │
│      ├── Phase 1: GPU check (A100 optimizations)    │
│      ├── Phase 2: Download dataset (tar.gz)         │
│      ├── Phase 3: Validate dataset structure        │
│      ├── Phase 4: Create data.yaml                  │
│      ├── Phase 5: Train (300 epochs, spot resume)   │
│      │   ├── Check GCS for existing checkpoint      │
│      │   ├── YOLO26 model.train(...)                │
│      │   └── Per-epoch GCS sync                     │
│      ├── Phase 6: Evaluate (val + predictions)      │
│      └── Phase 7: Summary + final GCS upload        │
└─────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  GCS Output: gs://vm-football-data/models/          │
│              yolo26_ball_sota/                       │
│  ├── yolo26_ball_sota/                              │
│  │   ├── weights/best.pt    ← Use for TOTNet       │
│  │   ├── weights/last.pt                            │
│  │   ├── results.csv                                │
│  │   └── plots/                                     │
│  └── training_summary.json                          │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
training/segmentation/
  ├── config_yolo26.yaml      # Hyperparameters config
  ├── train_yolo26.py          # Cloud training script
  └── setup_yolo26.py          # pip package setup

cloud/
  └── submit_yolo26_job.ps1    # Vertex AI A100 launcher
```

## Usage

```powershell
# Dry run (validate config, no job submitted)
.\cloud\submit_yolo26_job.ps1 -DryRun

# Smoke test (1 epoch, batch 2 — validates end-to-end)
.\cloud\submit_yolo26_job.ps1 -SmokeTest

# Full training (300 epochs on A100, ~20-30 hours)
.\cloud\submit_yolo26_job.ps1

# Monitor logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=us-central1
```

## Post-Training: TOTNet Integration

After YOLO26 training completes, the `best.pt` weights will be used
as the ball detector backbone in the TOTNet tracking pipeline.