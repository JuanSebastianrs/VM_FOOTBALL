# Jersey Number Tracking - Full Implementation Plan

Project: VM_FOOTBALL
Scope: Add robust jersey-number tracking on top of the current RF-DETR + ByteTrack + Team Clustering pipeline.

---

## 1. Executive Summary

The recommended approach is a hybrid identity stack:

1. Keep `track_id` as short-term temporal identity.
2. Keep `team_id` from clustering as team membership.
3. Add `jersey_number` as a track attribute estimated by OCR/classification.
4. Build a stable player key as `team_id + jersey_number` when confidence is high.
5. Use fallback to `team_id + track_id` when jersey number is unknown.

This avoids the main failure mode discussed earlier: replacing team clustering IDs directly with jersey numbers. Number alone is not globally unique (same number can appear on both teams) and single-frame OCR is noisy.

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. Assign a stable jersey number per tracked player over time.
2. Reduce frame-level OCR noise using temporal fusion.
3. Expose uncertainty explicitly (unknown/low-confidence states).
4. Integrate seamlessly with current outputs (video labels + diagnostics CSV + post report).
5. Keep compatibility with current role-aware logic (player, goalkeeper, referee).

### 2.2 Non-Goals (Phase 1)

1. Full roster-level identity mapping (name, position, club roster).
2. Perfect read rate in every frame (broadcast occlusion makes this unrealistic).
3. Training a large custom OCR model from scratch before integration.

---

## 2.3 Do We Need a New Model?

Short answer:

1. MVP and early validation: No.
2. Final robust deployment/publication-grade performance: Likely yes.

Recommended strategy:

1. Start without training a new model.
   - Use OCR/VLM backend plus temporal fusion and lock logic.
   - Validate end-to-end identity behavior and failure modes first.
2. Train only if metrics indicate a bottleneck.
   - Trigger training when wrong-lock rate, unknown rate, or switch rate remain high after threshold tuning.
3. If training is needed, keep it lightweight.
   - Train a compact jersey-number classifier (0-99) on extracted crops.
   - Keep OCR as auxiliary signal in an ensemble.

Decision gates:

1. If track-level number stability is acceptable on seen and unseen clips, do not train.
2. If instability persists in blur/occlusion/small-crop conditions, train a dedicated model.

Why this is best practice:

1. Minimizes initial complexity and accelerates integration.
2. Prevents premature training before observing real failure distribution.
3. Keeps compute and annotation cost aligned with measured gains.

---

## 3. Current Repo Integration Points

### 3.1 Existing code to leverage

1. `eval_team_clustering.py`
   - already computes `track_id`, `team_id`, role logic, diagnostics and visualization.
2. `core/identity/jersey_reader.py`
   - skeleton for prediction + temporal history.
3. `core/identity/crop_generator.py`
   - skeleton for dorsal/torso crop extraction.
4. `core/pipeline.py`
   - has `jersey_numbers` field and `use_jersey_reader` switch scaffold.

### 3.2 Principle

Do not modify team clustering semantics. Add a new identity layer on top:

- `team_id` remains unchanged.
- `jersey_number` is an orthogonal attribute.
- final display key can be `Team A #10` but internal IDs remain decomposed.

---

## 4. Recommended Architecture (Best Practical Approach)

## 4.1 Per-frame stages

1. Detect and track players (already done).
2. For each eligible track, extract a jersey crop.
3. Run OCR/classifier backend(s) on crop.
4. Store candidate predictions with confidence.
5. Fuse candidates temporally per track.
6. Emit stable number state (`locked`, `tentative`, `unknown`).

### 4.2 Dual backend strategy

Use two complementary predictors:

1. Fast digit classifier backend (ResNet/sequence classifier)
   - stable and efficient.
2. OCR backend (e.g., EasyOCR/Tesseract or VLM adapter)
   - stronger on uncommon font styles.

Fuse both with confidence weighting.

### 4.3 Team-aware identity key

Primary key after lock:

- `player_key = (team_id, jersey_number)`

Fallback when unknown:

- `player_key = (team_id, track_id)`

---

## 5. Data Model and State Machine

### 5.1 Track-level state object

For each `track_id`, maintain:

1. `team_id`: int
2. `role`: player/goalkeeper/referee
3. `obs_count`: total OCR observations
4. `votes[number]`: weighted vote accumulator
5. `top1_number`, `top1_score`
6. `top2_number`, `top2_score`
7. `status`: `unknown | tentative | locked`
8. `lock_streak`: consecutive windows supporting top1
9. `last_update_frame`

### 5.2 Status transitions

1. `unknown -> tentative`
   - min observations reached and top1 above initial threshold.
2. `tentative -> locked`
   - margin over top2 exceeds threshold for N windows.
3. `locked -> tentative`
   - only if sustained contradictory evidence exceeds unlock threshold.

Hysteresis is required to prevent oscillation.

---

## 6. Algorithm Details

### 6.1 Crop selection and quality gate

For each tracked bbox:

1. Extract torso/back crop using normalized ROI.
2. Reject crop if any quality gate fails:
   - too small,
   - excessive blur (variance of Laplacian below threshold),
   - too dark/too bright,
   - extreme aspect ratio.

Optional upgrade:

- Use pose/keypoint orientation to prefer back-facing crops.

### 6.2 OCR/classifier inference

For each accepted crop:

1. Run backend A (classifier) -> `(num_a, conf_a, topk_a)`.
2. Run backend B (OCR/VLM) -> `(num_b, conf_b)`.
3. Validate number range [0, 99].
4. Compute fused candidate score.

Suggested fusion:

- if `num_a == num_b`: confidence boost.
- else choose higher calibrated confidence, with conservative penalty.

### 6.3 Temporal fusion

For each track and frame window:

1. Update weighted votes: `votes[n] += w_conf * w_quality * w_recency`.
2. Recompute top1 and top2.
3. Compute margin: `margin = top1_score - top2_score`.
4. Apply state transition logic.

### 6.4 Suggested initial thresholds

These are starting values; tune with ablation.

1. `MIN_CROP_W = 24`, `MIN_CROP_H = 24`
2. `BLUR_MIN = 60.0`
3. `MIN_OBS_TENTATIVE = 4`
4. `MIN_OBS_LOCK = 8`
5. `LOCK_MIN_SCORE = 0.62`
6. `LOCK_MIN_MARGIN = 0.20`
7. `LOCK_STREAK_WINDOWS = 3`
8. `UNLOCK_MARGIN = 0.08`
9. `UNLOCK_STREAK_WINDOWS = 4`

---

## 7. Implementation Phases

## Phase 0 - Contract and interfaces

Deliverables:

1. Define canonical outputs for number tracking.
2. Freeze schema for diagnostics fields.
3. Add CLI flags in evaluator for reproducibility.

New CLI flags to add in `eval_team_clustering.py`:

1. `--use-jersey-number`
2. `--jersey-backend {resnet,ocr,ensemble}`
3. `--jersey-lock-min-score`
4. `--jersey-lock-min-margin`
5. `--jersey-min-obs-lock`
6. `--jersey-debug-csv`

## Phase 1 - Minimal viable integration (no heavy model dependency)

Goal: end-to-end wiring with a placeholder backend.

Tasks:

1. Implement `JerseyReader.predict` and temporal filter with pluggable backend interface.
2. Add a simple backend stub (rule-based or mock) for pipeline validation.
3. Integrate into `eval_team_clustering.py` loop.
4. Add visual label format: `Team A #10` when locked, `Team A #?` otherwise.

Acceptance:

1. Script runs end-to-end with jersey tracking enabled.
2. New CSV includes per-track number state fields.

## Phase 2 - Real OCR backend integration

Goal: usable real-world recognition.

Tasks:

1. Add OCR backend adapter in `core/identity/jersey_reader.py`.
2. Add preprocessing pipeline for OCR (denoise, contrast, binarize variants).
3. Implement range validation [0, 99] and parsing normalization.
4. Calibrate OCR confidence to common scale [0, 1].

Acceptance:

1. Stable reads on clear crops.
2. Unknowns are explicit instead of false confident numbers.

## Phase 3 - Classifier backend integration

Goal: speed and robustness.

Tasks:

1. Implement ResNet backend path in `JerseyReader`.
2. Support top-k output and calibrated confidence.
3. Add ensemble combiner function.

Acceptance:

1. Ensemble outperforms single backend on validation clips.

## Phase 4 - Crop quality and orientation improvements

Tasks:

1. Implement crop quality gate and quality score.
2. Optionally add front/back heuristic for goalkeeper and referee exclusions.
3. Use only role `player` by default (configurable for GK).

Acceptance:

1. Reduced noisy observations and fewer wrong locks.

## Phase 5 - Track fragmentation handling

Goal: preserve number identity when ByteTrack IDs break.

Tasks:

1. Build short-lived registry keyed by `(team_id, locked_number)` with spatial prior.
2. Re-associate new track IDs to existing player identity when evidence is strong.
3. Keep a cooldown and conflict resolution rule.

Acceptance:

1. Fewer identity resets after occlusions.

## Phase 6 - Diagnostics and reports

Tasks:

1. Add per-track fields in debug CSV:
   - `jersey_number_final`
   - `jersey_status`
   - `jersey_top1_score`
   - `jersey_top2_score`
   - `jersey_margin`
   - `jersey_obs_count`
   - `jersey_backend_agreement`
2. Add summary block in post report:
   - locked tracks,
   - unknown tracks,
   - conflict cases,
   - confidence stats by team.

Acceptance:

1. Explainability good enough for manual debugging clip-by-clip.

## Phase 7 - Ablation and threshold tuning

Matrix:

1. Backend: OCR vs classifier vs ensemble
2. Temporal filter: off vs on
3. Quality gate: off vs on
4. Lock threshold variants

Metrics:

1. Track-level number accuracy
2. Time-to-lock (frames)
3. Lock stability (switches per minute)
4. Unknown rate
5. False lock rate

Acceptance:

1. Chosen default has best stability/accuracy tradeoff.

## Phase 8 - Promote defaults and docs

Tasks:

1. Set default safe thresholds.
2. Update `README.md` and `MASTER_PROMPT.md` status sections.
3. Add quick commands for jersey tracking mode.

---

## 8. File-by-File Work Plan

### 8.1 `core/identity/jersey_reader.py`

Implement:

1. `load_models`
2. `predict`
3. `predict_batch`
4. `_predict_smolvlm` (or OCR adapter equivalent)
5. `_predict_resnet`
6. `_ensemble_predictions`
7. `_apply_temporal_filter`
8. `evaluate`

Add:

1. Backend interface abstraction.
2. Confidence calibration utility.
3. Number parser and validator.

### 8.2 `core/identity/crop_generator.py`

Implement:

1. `process_sequence`
2. `generate_metadata_csv`
3. `stream_crops`

Add:

1. quality score fields,
2. optional blur and brightness filtering.

### 8.3 `eval_team_clustering.py`

Integrate:

1. CLI flags for jersey tracking.
2. Jersey reader initialization.
3. Per-track jersey update inside frame loop.
4. Label rendering with locked number.
5. New CSV columns and post-report section.

### 8.4 Optional follow-up

1. `core/pipeline.py` integrate `use_jersey_reader` path once evaluator is stable.

---

## 9. Output Schema Extensions

### 9.1 Suggested per-frame export fields

1. `frame_idx`
2. `track_id`
3. `team_id`
4. `role`
5. `jersey_number`
6. `jersey_status`
7. `jersey_confidence`
8. `player_key`

### 9.2 Suggested per-track summary fields

1. `track_id`
2. `team_id`
3. `jersey_final`
4. `status_final`
5. `obs_count`
6. `lock_frame`
7. `switch_count`
8. `mean_confidence`

---

## 10. Validation Protocol

Clips:

1. Start with SNMOT-116, SNMOT-117, SNMOT-143.
2. Add at least 5 unseen clips for generalization.

Procedure:

1. Run baseline without jersey tracking.
2. Run with jersey tracking and same detector/tracker settings.
3. Compare diagnostics and visual output.

Manual QA checklist:

1. No frequent number flicker on stable tracks.
2. Unknown used when unreadable, not random wrong numbers.
3. No cross-team collision confusion for same jersey number.

---

## 11. Risks and Mitigations

1. Blur/occlusion causes noisy OCR
   - Mitigation: quality gate + temporal lock + unknown state.
2. Same number on both teams
   - Mitigation: key as `(team_id, jersey_number)`.
3. Track fragmentation
   - Mitigation: identity registry + re-association heuristic.
4. Referee/GK contamination
   - Mitigation: role-aware filtering defaults to outfield players.
5. Overfitting thresholds to known clips
   - Mitigation: unseen-clip validation mandatory before freezing defaults.

---

## 12. Definition of Done

1. Jersey number tracking works end-to-end in `eval_team_clustering.py`.
2. Visual labels show stable numbers with explicit unknown states.
3. Diagnostics include confidence, margin, and lock state.
4. Defaults are tuned with ablation on seen and unseen clips.
5. Documentation reflects final behavior and commands.

---

## 13. Immediate Next Actions (Practical Order)

1. Implement Phase 1 (MVP wiring) first in evaluator and `JerseyReader` temporal core.
2. Add OCR backend adapter (Phase 2) and validate on existing three clips.
3. Add ensemble and thresholds tuning (Phase 3 and Phase 7).
4. Freeze defaults and update docs (Phase 8).

This order minimizes risk and gets an end-to-end working baseline quickly while preserving the current stable team-clustering pipeline.
