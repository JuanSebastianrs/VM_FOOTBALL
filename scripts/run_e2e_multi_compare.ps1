# Compare v1.5 baseline postprocessing vs v2 postprocessing (fragment linking +
# switch detection + conflict reassignment) on multi-sequence E2E.
$ErrorActionPreference = "Stop"
$seqs = "SNMOT-148","SNMOT-116","SNMOT-132","SNMOT-190"

Write-Output "=== RUN 1: v15_baseline ==="
python scripts/evaluate_jersey_e2e_multi.py `
    --sequences $seqs `
    --model_path runs/jersey_perframe_v2_mixed/best.pt `
    --tag v15_baseline
if ($LASTEXITCODE -ne 0) { throw "baseline run failed" }

Write-Output "=== RUN 2: v2_postproc ==="
python scripts/evaluate_jersey_e2e_multi.py `
    --sequences $seqs `
    --model_path runs/jersey_perframe_v2_mixed/best.pt `
    --tag v2_postproc `
    --link_fragments --split_on_switch --reassign_conflicts
if ($LASTEXITCODE -ne 0) { throw "v2 run failed" }

Write-Output "ALL_RUNS_DONE"
