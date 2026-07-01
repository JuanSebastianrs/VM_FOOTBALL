# Run pipeline prerequisites (Phase 1 detection + Phase 8 team clustering + Phase 8.5 audit)
# for additional test sequences, enabling multi-sequence E2E jersey evaluation.
# Usage: powershell -File scripts/run_e2e_prereqs.ps1
$ErrorActionPreference = "Stop"
$seqs = @("SNMOT-116", "SNMOT-132", "SNMOT-190")
$base = "data/tracking/SoccerNet/tracking/test/test"

foreach ($seq in $seqs) {
    $outDir = "outputs/$seq"
    New-Item -ItemType Directory -Force $outDir | Out-Null
    $det = "$outDir/${seq}_detections.json"
    $team = "$outDir/${seq}_team_assignments.json"
    $audit = "$outDir/team_audit.json"

    if (-not (Test-Path $det)) {
        Write-Output "=== $seq Phase 1: detection ==="
        python core/detection/tactical_vision_extractor.py `
            --sequence_dir "$base/$seq" `
            --yolo_weights models/yolo26.pt `
            --rfdetr_weights models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth `
            --output_json $det
        if ($LASTEXITCODE -ne 0) { throw "Phase 1 failed for $seq" }
    }
    if (-not (Test-Path $team)) {
        Write-Output "=== $seq Phase 8: team clustering ==="
        python core/clustering/team_clustering_phase.py `
            --sequence_dir "$base/$seq" `
            --rfdetr_weights models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth `
            --detections_json $det `
            --output_json $team `
            --output_video "$outDir/${seq}_team_clustering.mp4"
        if ($LASTEXITCODE -ne 0) { throw "Phase 8 failed for $seq" }
    }
    if (-not (Test-Path $audit)) {
        Write-Output "=== $seq Phase 8.5: team audit ==="
        python scripts/audit_team_clustering.py `
            --sequence_dir "$base/$seq" `
            --team_json $team `
            --detections_json $det `
            --output_json $audit
        if ($LASTEXITCODE -ne 0) { throw "Audit failed for $seq" }
    }
    Write-Output "=== $seq prereqs DONE ==="
}
Write-Output "ALL_SEQUENCES_DONE"
