# scripts/run_parseq_ft_eval_chain.ps1
# Cadena de evaluacion del PARSeq fine-tuneado (runs/parseq_jersey_ft/best.pt):
#   1) cache PARSeq-FT para val y test (mismo formato que parseq_base_*.npz)
#   2) sweep de parseq_weight en VAL (calibracion; config de produccion fija)
#   3) evaluacion final en TEST con el w elegido (pasarlo como argumento)
# Uso:
#   powershell -File scripts/run_parseq_ft_eval_chain.ps1            # pasos 1-2
#   powershell -File scripts/run_parseq_ft_eval_chain.ps1 -TestW 0.25  # paso 3

param([double]$TestW = -1)

$ErrorActionPreference = "Stop"
Set-Location "D:\sebastian\Tesis\VM_FOOTBALL"

$CKPT = "runs/parseq_jersey_ft/best.pt"
$DS   = "datasets/jersey_tracking_v2_224"
$MODEL = "runs/jersey_perframe_v3_224/best.pt"
$ROSTER = "datasets/jersey_tracking_v1/rosters.json"
$VAL_DIR = "outputs/_verify/jersey_val_sweep"
$TEST_DIR = "outputs/_verify/jersey_test_eval"
# Config de produccion v2.x (docs/jersey_perframe_v2.md SS18-19)
$PROD = @("--fusion_mode", "confidence_topk", "--p1_threshold", "0.90",
          "--margin_threshold", "0.30", "--legibility_threshold", "0.70",
          "--min_legible_frames", "4", "--roster_json", $ROSTER)

if ($TestW -lt 0) {
    if (-not (Test-Path "$VAL_DIR/parseq_ft_val.npz")) {
        python scripts/build_parseq_cache.py --dataset_dir $DS --split val `
            --model parseq --checkpoint $CKPT --output "$VAL_DIR/parseq_ft_val.npz"
    }
    if (-not (Test-Path "$TEST_DIR/parseq_ft_test.npz")) {
        python scripts/build_parseq_cache.py --dataset_dir $DS --split test `
            --model parseq --checkpoint $CKPT --output "$TEST_DIR/parseq_ft_test.npz"
    }
    foreach ($w in @(0.15, 0.25, 0.35, 0.50, 0.65)) {
        Write-Host "=== VAL sweep w=$w ==="
        python scripts/evaluate_jersey_dataset_temporal.py --dataset_dir $DS `
            --model_path $MODEL --split val --output_dir "$VAL_DIR/ft_w$w" `
            --parseq_cache "$VAL_DIR/parseq_ft_val.npz" --parseq_weight $w @PROD
    }
} else {
    Write-Host "=== TEST con w=$TestW ==="
    python scripts/evaluate_jersey_dataset_temporal.py --dataset_dir $DS `
        --model_path $MODEL --split test --output_dir "$TEST_DIR/ft_w$TestW" `
        --parseq_cache "$TEST_DIR/parseq_ft_test.npz" --parseq_weight $TestW @PROD
}
