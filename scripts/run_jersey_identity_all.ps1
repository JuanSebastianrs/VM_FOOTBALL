# Run the jersey identity phase (v2 postprocessing config, no roster) on every
# test sequence with cached detections. Skips sequences whose output already
# exists, so the run is resumable. Produces:
#   outputs/<seq>/<seq>_jersey_identity_v2_all.json
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."
$tag = "v2_all"
$dataRoot = "data\tracking\SoccerNet\tracking\test\test"

$seqs = Get-ChildItem outputs -Directory -Filter 'SNMOT-*' |
    Where-Object { Test-Path (Join-Path $_.FullName ($_.Name + '_detections.json')) } |
    Select-Object -ExpandProperty Name | Sort-Object

$done = 0
foreach ($seq in $seqs) {
    $out = "outputs\$seq\${seq}_jersey_identity_$tag.json"
    if (Test-Path $out) { Write-Output "[$seq] already done"; $done++; continue }

    $args = @(
        "core/identity/jersey_identity_phase.py",
        "--sequence_dir", "$dataRoot\$seq",
        "--detections_json", "outputs\$seq\${seq}_detections.json",
        "--team_assignments_json", "outputs\$seq\${seq}_team_assignments.json",
        "--model_path", "runs/jersey_perframe_v2_mixed/best.pt",
        "--output_json", $out,
        "--inference_mode", "temporal",
        "--fusion_mode", "arithmetic",
        "--legibility_model", "runs/jersey_legibility_v1/best.pt",
        "--legibility_threshold", "0.70",
        "--p1_threshold", "0.75",
        "--margin_threshold", "0.15",
        "--link_fragments", "--split_on_switch", "--reassign_conflicts"
    )
    $audit = "outputs\$seq\team_audit.json"
    if (Test-Path $audit) { $args += @("--team_mapping", $audit) }

    Write-Output "[$seq] running jersey identity..."
    python @args
    if ($LASTEXITCODE -ne 0) { Write-Output "[$seq] FAILED (exit $LASTEXITCODE)" }
    else { $done++ }
}
Write-Output "JERSEY_ALL_DONE ($done/$($seqs.Count))"
