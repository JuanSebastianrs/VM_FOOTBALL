# v1.7 chain: legibility scoring on the 224px dataset -> per-frame training at 224
# (warm-started from v2_mixed, same recipe/data so the only change is resolution)
# -> val sweep -> test eval -> multi-sequence E2E gate.
$ErrorActionPreference = "Stop"

Write-Output "=== STEP 0: verify splits match v1 ==="
python -c "import json; a=json.load(open('datasets/jersey_tracking_v1/splits.json')); b=json.load(open('datasets/jersey_tracking_v2_224/splits.json')); same=(a==b); print('splits identical' if same else 'SPLITS DIFFER - copying v1'); same or json.dump(a, open('datasets/jersey_tracking_v2_224/splits.json','w'), indent=2)"
if ($LASTEXITCODE -ne 0) { throw "splits check failed" }

Write-Output "=== STEP 1: legibility scoring (224 dataset) ==="
python scripts/score_crop_legibility.py `
    --dataset_dir datasets/jersey_tracking_v2_224 `
    --legibility_model runs/jersey_legibility_v1/best.pt `
    --device cuda:0
if ($LASTEXITCODE -ne 0) { throw "legibility scoring failed" }

Write-Output "=== STEP 2: train per-frame at 224 (mixed data) ==="
python training/identification/train_jersey_perframe.py `
    --dataset_dir datasets/jersey_tracking_v2_224 `
    --output_dir runs/jersey_perframe_v3_224 `
    --init_from runs/jersey_perframe_v2_mixed/best.pt `
    --soccernet_index datasets/soccernet/jersey-2023/perframe_index_train.json `
    --min_legibility 0.5 --img_size 224 `
    --epochs 30 --batch_size 32 --lr 2e-4 --num_workers 4 --device cuda:0
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Output "=== STEP 3: val eval (production config) ==="
python scripts/evaluate_jersey_dataset_temporal.py `
    --dataset_dir datasets/jersey_tracking_v2_224 `
    --model_path runs/jersey_perframe_v3_224/best.pt `
    --split val --output_dir outputs/jersey_eval/temporal_perframe_v3_224 `
    --fusion_mode arithmetic --legibility_threshold 0.7 `
    --p1_threshold 0.75 --margin_threshold 0.15 --device cuda:0
if ($LASTEXITCODE -ne 0) { throw "val eval failed" }

Write-Output "=== STEP 4: test eval (49 sequences, production config) ==="
python scripts/evaluate_jersey_dataset_temporal.py `
    --dataset_dir datasets/jersey_tracking_v2_224 `
    --model_path runs/jersey_perframe_v3_224/best.pt `
    --split test --output_dir outputs/jersey_eval/temporal_perframe_v3_224 `
    --fusion_mode arithmetic --legibility_threshold 0.7 `
    --p1_threshold 0.75 --margin_threshold 0.15 --device cuda:0
if ($LASTEXITCODE -ne 0) { throw "test eval failed" }

Write-Output "=== STEP 5: multi-sequence E2E gate (v1.6 postproc) ==="
python scripts/evaluate_jersey_e2e_multi.py `
    --sequences SNMOT-148 SNMOT-116 SNMOT-132 SNMOT-190 `
    --model_path runs/jersey_perframe_v3_224/best.pt `
    --tag v3_224 `
    --link_fragments --split_on_switch --reassign_conflicts
if ($LASTEXITCODE -ne 0) { throw "E2E gate failed" }

Write-Output "CHAIN_DONE"
