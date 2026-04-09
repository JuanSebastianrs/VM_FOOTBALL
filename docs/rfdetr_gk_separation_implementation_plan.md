# RF-DETR Role-Aware Separation - Implementation Plan

## Purpose
Create a generalized team-clustering and role-aware assignment pipeline that works across arbitrary clips by retraining RF-DETR with explicit role separation (`player`, `goalkeeper`, `referee`) and redesigning downstream logic to treat goalkeepers/referees as separate roles.

## High-Level Goals
- Use RF-DETR with 3 classes: player, goalkeeper, referee.
- Build Team A/Team B identity from outfield players only.
- Assign goalkeeper team with a robust multi-cue temporal strategy, not color-only logic.
- Avoid clip-specific heuristics and hardcoded jersey assumptions.
- Preserve or improve outfield team-clustering quality.

## Scope
### In scope
- Data mapping and retraining configuration for role-aware RF-DETR.
- Inference integration preserving class IDs.
- Outfield-only team clustering.
- Goalkeeper team assignment redesign (spatial + temporal + optional weak appearance).
- Evaluation protocol, diagnostics, and ablations.

### Out of scope (for first pass)
- Full player identity by jersey number OCR.
- End-to-end pipeline integration in core/pipeline.py unless explicitly requested.
- Full player identity by jersey number OCR.

## Success Criteria
- Works when only one goalkeeper is visible in a clip.
- Goalkeeper team label remains stable through temporary occlusion and re-entry.
- No hardcoded team colors or clip-specific if-statements.
- Lower goalkeeper misassignment rate than current baseline.
- No regression in outfield team clustering metrics.

## Current Status (2026-04-05)
Implemented already (no retraining required):
- Class-aware detector scaffolding in `core/detection/` with backward compatibility for legacy single-class RF-DETR.
- `eval_team_clustering.py` feature flags:
   - default role-aware (`use_gk_class=True`)
   - `--no-use-gk-class`
   - `--gk-assignment-mode {legacy,fused}`
   - `--cluster-referee` / `--ignore-referee`
- Outfield-only clustering path when goalkeeper class is available, with safe fallback when class separation is absent.
- Goalkeeper assignment modes:
   - `legacy`: edge/motion + neighbor voting
   - `fused`: class-seeded fused scoring (spatial + neighbor + weak appearance) with temporal fallback
- Extended diagnostics and post-report fields:
   - decision confidence
   - reason codes
   - fused component scores per team

Validated on clips:
- SNMOT-116, SNMOT-117, and SNMOT-143 all run successfully with the canonical 3-class checkpoint.
- Fused and legacy modes are stable on the tested clips.
- Default behavior is now the validated role-aware fused path.

Implemented already (training complete):
- 3-class RF-DETR training complete with taxonomy: `player`, `goalkeeper`, `referee` (ball ignored).
- Artifacts available in GCS:
   - `gs://vm-football-data/models/rfdetr_player_gk_ref/rfdetr/rfdetr_base_448_3class/checkpoint_best_ema.pth`
   - `gs://vm-football-data/models/rfdetr_player_gk_ref/rfdetr/rfdetr_base_448_3class/results.json`
- Smoke test passed end-to-end (dataset download, conversion, training, checkpoint verification, upload).

Pending:
- True class-separated ablation (Baseline B / Proposed) using `player` + `goalkeeper` + `referee` outputs.
- Final threshold tuning on validation-only clips.
- Unseen-clip generalization report with finalized defaults.

## Phase 0 - Kickoff Contract (Future Conversation Start)
1. Confirm objective: generalized behavior across arbitrary clips.
2. Confirm class taxonomy: player, goalkeeper, referee.
3. Confirm deliverables:
   - Training config and retraining-ready scripts.
   - Updated evaluation pipeline.
   - Diagnostics + ablation report.
4. Confirm acceptance metrics and minimum test set.

## Phase 1 - Data and Label Mapping Validation
1. Audit current labels and mapping in training scripts/config.
2. Define final mapping:
   - player_left -> player
   - player_right -> player
   - goalkeeper_left -> goalkeeper
   - goalkeeper_right -> goalkeeper
   - referee -> referee
   - ball -> ignore
3. Generate class distribution report per split (train/valid/test).
4. Validate goalkeeper sample adequacy and edge cases:
   - Far camera views
   - Partial visibility
   - Side-of-frame appearances
5. Freeze dataset version/hash for reproducibility.

### Artifacts
- Mapping table markdown.
- Class balance CSV/JSON.
- Dataset version note.

## Phase 2 - Retrain RF-DETR (Role-Aware)
1. Clone current best training recipe to minimize variance.
2. Update config for 3 classes and corresponding class mapping.
3. Add imbalance handling if needed:
   - Class weights, oversampling, or focal-like alternatives as available.
4. Train with checkpoint sync and metadata logging.
5. Export best checkpoint and full metrics.

### Required outputs
- best checkpoint path
- validation/test metrics with per-class AP (player, goalkeeper, referee)
- training log
- training curves
- model card section documenting mapping and caveats

## Phase 3 - Inference Integration
1. Update detector wrapper to preserve class_id in outputs.
2. Split detections by role each frame:
   - players set
   - goalkeepers set
3. Keep tracking stable per role:
   - Either class-aware tracking
   - Or dual trackers (players and goalkeepers)
4. Add fallback behavior when no goalkeeper detections are present.
5. Maintain backward-compatible mode for old single-class checkpoint.

### Validation checks
- Class IDs survive full detection -> tracking -> visualization path.
- Goalkeeper track IDs are stable enough for temporal logic.

### Phase 3 status
- Complete.
- `core/detection/rfdetr_detector.py` now supports class-name propagation for legacy and future checkpoints.
- `eval_team_clustering.py` now captures per-track class counts and infers goalkeeper IDs by taxonomy.
- `eval_team_clustering.py` also infers and tracks referee role separately (`REF`).

## Phase 4 - Team Clustering Refactor (Outfield Only)
1. Build team prototypes from player tracks only.
2. Exclude goalkeeper tracks from prototype construction and cluster fitting.
3. Keep current HSV/DBSCAN mode as baseline-compatible option.
4. Keep unknown policy configurable.
5. Normalize Team A/B identity consistently (left-right initialization or equivalent).

### Why
Removing goalkeeper samples from team prototype generation prevents atypical goalkeeper kits (gray/black/neon) from distorting team clusters.

### Phase 4 status
- Implemented with fallback.
- Outfield-only clustering is active by default (role-aware enabled) when goalkeeper classes are observed.
- If exclusion leaves fewer than `k` tracks, pipeline falls back to all eligible tracks and logs a warning.

## Phase 5 - Goalkeeper Team Assignment Redesign
Use role-aware assignment, independent from outfield clustering internals.

### Candidate identification
- Primary: use goalkeeper class detections/tracks from model.
- Secondary fallback (if needed): current edge/motion-based heuristic.

### Team affiliation model
Compute a fused score per team and select argmax:

S(team) = ws * spatial_affinity + wn * neighbor_affinity + wa * appearance_affinity

Recommended defaults:
- ws high
- wn medium
- wa low

### Cue definitions
1. spatial_affinity:
   - Distance from goalkeeper anchor to team centroid or robust team shape center.
   - Optionally on pitch-transformed coordinates when available.
2. neighbor_affinity:
   - Weighted vote from nearest outfield players (distance-decayed).
   - Optionally restricted to same half/lane to reduce contamination by opponents.
3. appearance_affinity:
   - Weak prior from goalkeeper crop similarity to team appearance prototypes.
   - Automatically downweighted when confidence is low (e.g., low saturation gray kits).

### Temporal consistency
- Hysteresis for team switching.
- Minimum sustained contrary evidence before switching team.
- Retain last stable team label during short missing intervals.
- Allow unknown when confidence stays below threshold.

### Phase 5 status
- Implemented in `eval_team_clustering.py` as two selectable modes:
   - `legacy`: candidate selection + neighbor vote assignment
   - `fused`: class-seeded fused score (`spatial`, `neighbor`, weak `appearance`) with temporal fallback if class separation is unavailable
- Hysteresis and temporal fallback preserved for non-class-separated runs.

## Phase 6 - Diagnostics and Explainability
Extend existing outputs with explicit goalkeeper decision traces.

### Add to diagnostics CSV
- per-window fused score by team
- component scores (spatial/neighbor/appearance)
- decision confidence
- switch triggers and reason codes

### Add to post-report
- final goalkeeper team assignment(s)
- number of switches
- confidence summary
- windows with low confidence

### Phase 6 status
- Partially implemented.
- Added to diagnostics CSV:
   - decision confidence
   - reason code
   - fused components (`fused_spatial_*`, `fused_neighbor_*`, `fused_appearance_*`, `fused_score_*`)
- Added to post-report:
   - assignment confidence
   - reason code
- Pending enhancement: explicit per-window switch trigger timeline.

## Phase 7 - Evaluation and Ablation Matrix
Run side-by-side comparisons:

1. Baseline A:
   - single-class detector + current GK logic
2. Baseline B:
   - role-aware detector (3-class) + current GK logic
3. Proposed:
   - role-aware detector (3-class) + fused GK assignment + hysteresis

### Metrics
- Goalkeeper team assignment accuracy
- Goalkeeper switch rate per minute
- Time-to-correct after re-entry
- Outfield team clustering quality (no regression)

### Test protocol
- First: SNMOT-116 and SNMOT-117
- Then: additional unseen clips for generalization
- Report mean and per-clip breakdown

### Phase 7 status
- Ablation harness is ready at CLI level.
- Can run now:
   - Baseline A: `--gk-assignment-mode legacy`
   - Proposed role-aware (default): `python eval_team_clustering.py --mode hsv --k 2`
- Baseline B and full Proposed now depend on evaluation runtime only (checkpoint ready).
- SNMOT-143 unseen-clip validation completed in the `dl` environment with the canonical 3-class GCS checkpoint.
- Legacy and fused runs produced the same goalkeeper team assignments on that clip.
- SNMOT-116 and SNMOT-117 validation also completed successfully on the local flat-folder clips.
- Phase 7 is effectively done for the three current validation clips.

## Phase 8 - Rollout Strategy
1. Introduce feature flags in eval script:
   - --no-use-gk-class
   - --gk-assignment-mode {legacy,fused}
   - --cluster-referee
2. Keep legacy path for regression tests.
3. Tune thresholds on validation clips only.
4. Freeze defaults once unseen-clip performance is acceptable.

### Phase 8 status
- Completed: the validated role-aware fused path is now the default evaluation behavior.
- Legacy and debug overrides remain available through flags (`--no-use-gk-class`, `--gk-assignment-mode legacy`, `--cluster-referee`).
- Phase 8 goal is satisfied; future changes should only happen if additional unseen-clip evidence justifies them.

## Phase 9 - Risks and Mitigation
1. Goalkeeper class underrepresented:
   - Mitigate with class balancing and targeted augmentation.
2. Track fragmentation under occlusion:
   - Tune tracker parameters for goalkeeper stream.
3. Camera motion/perspective instability:
   - Prefer normalized anchor and temporal windowing.
4. Appearance ambiguity for gray/black kits:
   - Keep appearance cue weak.
5. Overfitting to known clips:
   - Enforce unseen-clip evaluation before finalizing.

## Definition of Done
- Role-aware RF-DETR checkpoint available and documented.
- Inference path correctly separates player and goalkeeper detections.
- Inference path correctly tracks referee as separate role.
- Outfield team clustering excludes goalkeeper samples.
- Goalkeeper assignment uses fused spatial-temporal logic with hysteresis.
- Ablation report shows goalkeeper improvement without outfield regression.
- Documentation updated with architecture and known caveats.

## Interim Done (Post-Training)
- Backward-compatible scaffolding for role-aware integration completed.
- Role-aware logic and ablation flags wired in evaluation.
- Diagnostics extended for explainability and tuning.
- Core documentation synchronized to current post-training state.

## Suggested File Touchpoints (Future Implementation)
- cloud/rfdetr_player_detection/config.yaml
- cloud/rfdetr_player_detection/train_cloud.py
- core/detection/detector.py
- core/detection/rfdetr_detector.py
- core/clustering/team_classifier.py
- eval_team_clustering.py
- README.md
- MASTER_PROMPT.md
- docs/VM_FOOTBALL_Short_Paper_updated.md

## Copy-Paste Prompt For Next Conversation
Run the post-training ablation matrix for the role-aware RF-DETR checkpoint (`player`, `goalkeeper`, `referee`) and finalize default evaluation settings. Compare legacy vs fused goalkeeper assignment, keep referee excluded from team clustering, and export updated diagnostics/post-reports on unseen clips.
