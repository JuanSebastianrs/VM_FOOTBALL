# Run the pipeline over every SoccerNet-GSR valid sequence, data phases only.
# Cache-incremental and resumable: sequences already processed are skipped per-phase.
# Produces per sequence:
#   outputs/<seq>/<seq>_tracking_2d.csv, _team_assignments.json, _jersey_identity.json
#
# --parseq_weight 1.0 makes the PARSeq OCR reader the sole voice on frames where it
# reads a number. On GSR the per-frame CNN classifier is confidently wrong (it hits
# p1 ~ 0.99 on the wrong digit, see results_final/EVALUATION_METHODOLOGY.md), and
# since the frame posterior is mixed geometrically as cls^(1-w) * ocr^w, any w < 1
# lets the classifier drag the OCR off the right answer. Measured on SNGS-021:
# OCR alone gets 8/9 tracklets right with zero wrong or spurious numbers, versus
# 1/7 for the roster-masked classifier. Do NOT re-add --no_parseq here.
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "."
$gsr = "data\SoccerNetGS\gamestate-2024\valid"

$seqs = Get-ChildItem $gsr -Directory -Filter 'SNGS-*' |
    Select-Object -ExpandProperty Name | Sort-Object

foreach ($seq in $seqs) {
    Write-Output "########## $seq ##########"
    python src\tactical_vision_pipeline.py `
        --sequences $seq `
        --data_root data/SoccerNetGS/gamestate-2024/valid `
        --only detect cmc track team jersey map2d `
        --jersey_model runs/jersey_perframe_v2_mixed/best.pt `
        --parseq_model parseq `
        --parseq_checkpoint runs/parseq_jersey_ft/best.pt `
        --parseq_weight 1.0 `
        --link_fragments --split_on_switch --reassign_conflicts `
        --no_scanning
}
Write-Output "GSR_PIPELINE_ALL_DONE"
