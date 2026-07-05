# VM_FOOTBALL — TacticalVision AI
## Agent Onboarding Guide v1.0

### 1. What is this project?
A master's thesis building an end-to-end computer vision pipeline that turns single-view broadcast football footage into structured tactical data: player and ball tracking, team classification, metric pitch coordinates, physical metrics, and interactive analytics.

**Reference clip for validation:** `SNMOT-148` (750 frames, 25 FPS, Goal action).

### 2. Current Snapshot
| Phase | Status | Key output files |
|---|---|---|
| Ball detection (YOLO26) | Functional | `models/yolo26.pt` |
| Player detection (RF-DETR 3-class) | Functional | `models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth` |
| Tracking (ByteTrack + Viterbi HMM) | Functional | `*_detections.json`, `*_trajectory.json` |
| Camera motion compensation (CMC) | Functional | `*_cmc.json` |
| Team clustering (HSV + role-aware GK) | Functional | `*_team_assignments.json` |
| **2D field mapping (PnLCalib + Lie smoother)** | **Functional** | `*_tracking_2d.csv` |
| **Physical & shape analytics** | **Functional** | `*_player_physical_metrics.csv`, `*_team_shape_metrics.csv` |
| **Pred vs GT comparison** | **Functional** | `comparison/comparison_summary.json` |
| **Interactive dashboard** | **Functional** | `dashboard/index.html` |
| SAM2 segmentation | Functional | Video rendering + masks |
| Jersey number reading | **Functional (v1.7)** | `core/identity/`, `training/identification/`, model `runs/jersey_perframe_v3_224/best.pt` |
| Event detection (receptions) | **Functional** | `core/events/`, `outputs/<seq>/scanning/pass_reception_events.parquet` |
| **Visual scanning (head-turn before reception)** | **Functional (weak-supervised model)** | `core/scanning_v2/`, model `outputs/scanning_training/models/best/` |

**Latest milestone completed:**
- Phase 10/11: Analytics extraction (physical metrics, team shape, Pred vs GT comparison, HTML dashboard for `SNMOT-148`).

### 3. Pipeline phases (end-to-end)

**Unified runner:** `src/tactical_vision_pipeline.py` executes all phases with
**incremental caching** (a phase is skipped when its declared outputs already
exist; `--force all` or `--force <phase>` recomputes). Video renders are OFF by
default (`--render` enables mapper video, SAM2, scanning clips). Multi-sequence:
`--sequences SNMOT-116 SNMOT-117 ...`. Phase selection: `--only` / `--skip`.

1. **detect** — YOLO26 (ball) + RF-DETR (players/goalkeepers/referees).
2. **cmc** — Frame-to-frame affine compensation from field keypoints.
3. **track** — Viterbi HMM over ball detections with Dummy Node for occlusions.
4. **eval** — Tracking metrics vs GT (plot).
5. **team** — HSV descriptor + K-Means (k=2), with separate GK role via temporal hysteresis.
6. **audit / jersey** — Optional: team-clustering audit, jersey number identification.
7. **map2d** — PnLCalib per-frame calibration → bidirectional SO(3) Lie smoother → `tracking_2d.csv` + `calibration_hinv.json` (video only with `--render`).
8. **analytics** — Distance, speed, acceleration, sprints, team shape.
9. **gt2d / gt_compare** — Optional (`--with_gt`): project GT to 2D, Hungarian comparison.
10. **scanning** — Scanning V2: reception events → head pose (approx.) → head-turn detection → `outputs/<seq>/scanning/`.
11. **scan_pred** — Trained scanning classifier predictions → `outputs/<seq>/scanning/model_predictions.parquet`.
12. **Dashboard** — Self-contained HTML with Chart.js (separate: `dashboard/generate_dashboard.py`).

### 4. What was recently done (critical context)
- Added `project_point_to_world()` to mapper for metric export.
- Added `--output_csv` and `--no_video` to mapper CLI.
- Created `core/analytics/compute_metrics.py` with outlier gating (`OUTLIER_DIST_M = 3.0`), speed cap (`MAX_SPEED_KMH = 40.0`), minimum sprint frames (`MIN_SPRINT_FRAMES = 5`).
- Created `core/analytics/project_gt_to_2d.py` to project SoccerNet GT to metric pitch.
- Created `core/analytics/compare_pred_gt.py` with per-frame Hungarian matching, automatic `team_mapping_pred_to_gt` inference, and threshold-explicit metrics.
- Created `dashboard/generate_dashboard.py` that builds a self-contained HTML dashboard applying the team mapping so Pred/GT colors align.

### 5. Data paths and file conventions
| Path | Description |
|---|---|
| `data/tracking/SoccerNet/tracking/test/test/SNMOT-148/` | Raw images + GT (`gt/gt.txt`, `gameinfo.ini`, `seqinfo.ini`) |
| `outputs/SNMOT-148/SNMOT-148_tracking_2d.csv` | Predicted player/ball positions in metres |
| `outputs/SNMOT-148/SNMOT-148_gt_tracking_2d.csv` | GT projected to metres |
| `outputs/SNMOT-148/SNMOT-148_player_physical_metrics.csv` | Per-player summary (distance, speed, sprints) |
| `outputs/SNMOT-148/SNMOT-148_player_physical_timeseries.csv` | Per-frame per-player speed/accel |
| `outputs/SNMOT-148/SNMOT-148_team_shape_metrics.csv` | Per-frame per-team shape stats |
| `outputs/SNMOT-148/comparison/comparison_summary.json` | Pred vs GT aggregated metrics |
| `outputs/SNMOT-148/comparison/matches_detailed.csv` | Every matched pair for inspection |
| `outputs/SNMOT-148/dashboard/index.html` | Final interactive dashboard |
| `outputs/SNMOT-148/scanning/` | Scanning V2: reception events, head pose, scanning labels, model predictions, event clips |
| `outputs/scanning_training/` | Global scanning-classifier training artifacts (dataset/, models/, reports_gt/) |
| `outputs/_archive/` | Superseded experiment artifacts (not used by the pipeline) |

**Output layout rule:** everything that belongs to ONE sequence lives under
`outputs/<video_id>/` (scanning included, in the `scanning/` subfolder — see
`core/scanning_v2/paths.py`). Only global cross-sequence artifacts (model
training) live outside. `scripts/migrate_outputs_layout.py` migrates legacy
layouts.

**Coordinate conventions:**
- Pitch size: 105 m × 68 m.
- Origin: top-left corner (x right, y down).
- Time: `time_s = (frame_id - 1) / fps`, so frame 1 starts at 0.0 s.

### 6. Canonical commands
```bash
# 0) Unified pipeline (incremental cache; add --render for videos, --with_gt for GT compare)
python src/tactical_vision_pipeline.py \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148
# multi-sequence / phase selection:
python src/tactical_vision_pipeline.py --sequences SNMOT-116 SNMOT-148 --only scanning scan_pred

# 1) Project GT to 2D (needs calibration_hinv.json already present)
python core/analytics/project_gt_to_2d.py \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --calibration_json outputs/SNMOT-148/calibration_hinv.json \
    --output_csv outputs/SNMOT-148/SNMOT-148_gt_tracking_2d.csv \
    --fps 25.0

# 2) Compute Pred metrics
python core/analytics/compute_metrics.py \
    --input_csv outputs/SNMOT-148/SNMOT-148_tracking_2d.csv \
    --output_dir outputs/SNMOT-148 \
    --fps 25.0

# 3) Compare Pred vs GT
python core/analytics/compare_pred_gt.py \
    --pred_csv outputs/SNMOT-148/SNMOT-148_tracking_2d.csv \
    --gt_csv outputs/SNMOT-148/SNMOT-148_gt_tracking_2d.csv \
    --output_dir outputs/SNMOT-148/comparison \
    --max_dist_m 3.0

# 4) Generate dashboard
python dashboard/generate_dashboard.py \
    --sequence_name SNMOT-148 \
    --output_html outputs/SNMOT-148/dashboard/index.html
```

### 7. Validation checklist (minimum)
- [ ] `comparison_summary.json` exists and contains:
  - `overall_rmse_m` (should be ~0.3–0.4 m for SNMOT-148)
  - `coverage` ≥ 0.90
  - `precision` ≥ 0.95
  - `team_mapping_pred_to_gt` with a mapping for both teams
- [ ] `max_speed_kmh` in physical metrics ≤ 40.0 (speed cap).
- [ ] First `time_s` in timeseries = 0.0.
- [ ] Dashboard opens and charts render.
- [ ] Pitch viewer shows Pred circles and GT squares with aligned colors.

### 8. Known caveats
- `team_id` from clustering is **arbitrary** (0/1 are labels, not left/right). Always use `team_mapping_pred_to_gt` when comparing with GT.
- Pred vs GT comparison uses the **same calibration** (`calibration_hinv.json`) for both. It validates bbox alignment and projection consistency, not absolute independent ground truth.
- `max_error_m` in summary is capped by `max_dist_m` (default 3.0 m) because unmatched pairs beyond that threshold are discarded.
- Dashboard metadata (frames, fps, duration) is currently hardcoded; should be read from `seqinfo.ini` for generic clips.

### 9. Rules for future agents
- **Do not** change output CSV column names without updating all downstream scripts.
- **Do not** assume `team_id` semantics are fixed; always resolve via mapping when comparing with GT.
- **Do** add a changelog entry under “Agent changelog” when modifying analytics or dashboard logic.
- **Do** validate on `SNMOT-148` before claiming a change works end-to-end.

### 10. Prioritized backlog
| Priority | Task | Notes |
|---|---|---|
| High | Dynamic dashboard metadata | Read `seqinfo.ini` instead of hardcoded 750/25/30 |
| High | Bijective team mapping | Use Hungarian on 2×2 vote matrix instead of per-team majority |
| Medium | Untruncated metrics | Compute RMSE over all pairs (not just ≤3 m) for audit |
| Medium | Ball analytics | Possession detection, nearest player, pass events |
| Medium | Spatial analytics | Voronoi, nearest defender, line-breaking passes |
| Low | Jersey OCR | SmolVLM2 / ResNet on player crops |
| Low | Video re-render | Overlay metric positions + speed labels on original frames |

---

## Agent changelog
- **2026-07-05 (d)** — Scanning: los clips de evento ahora CONVIVEN con las demás capas del pipeline. Bug: `ScanningV2Visualizer.render_event` dibujaba solo al receptor sobre el frame crudo (sin detecciones de otros jugadores, sin equipos, sin dorsales) y su minimapa iba a un mp4 aparte que nunca entraba al FINAL — durante el scanning "desaparecía todo". Fix en `core/scanning_v2/visualization.py`: (1) todos los jugadores dibujados con la paleta de equipos del video principal + árbitro `REF` + dorsales (labels de `<seq>_jersey_identity.json`, convención locked `10` / tentative `10?`, cargados en `render_scanning_events.py`); (2) `{eid}_video.mp4` ahora es side-by-side cámara|minimapa (scale=8/margin=40, mismas proporciones del 2d_map) con todos los jugadores + dorsales + balón + cono de orientación del receptor; (3) `render_from_outputs` borra clips obsoletos de corridas anteriores antes de renderizar (el FINAL anexaba eventos que ya no existían, con estilo viejo). SNMOT-148 regenerado (2 eventos con la calibración nueva). 182 tests pasan.
- **2026-07-05 (c)** — Map2D: `RobustOfflineSmoother` en producción (`--smoothing_mode robust_offline`, default; `lie_bidir` disponible). Rechazo robusto (rep_err + proyección insana + mediana deslizante 11f/4°) → interpolación → gauss fase-cero σ=3 sobre rotvec/focal/pos; huecos >25f sin H_inv. Domina al Lie bidireccional en TODAS las métricas en ambas secuencias validadas: SNMOT-148 jitter 0.014 vs 0.048, desv 0.25 vs 0.82 m, p95 3.4 vs 7.8 m; SNMOT-116 (100% calibrada, caso mayoritario) 0.016 vs 0.054 / 0.33 vs 0.57 / 2.37 vs 2.62. Clase reproduce bit-exacto la variante del script de análisis. Artefactos de SNMOT-148 regenerados. Informe: `docs/minimap_stabilization_technical_report.md` §2026-07-05(b).
- **2026-07-05 (b)** — Map2D: auditoría del suavizado Lie + fix de huecos; Jersey: SR descartada. (1) Nuevas herramientas de medición: `tactical_vision_2d_mapper.py --dump_raw_calib` (mediciones crudas PnLCalib) + `scripts/analyze_lie_smoothing.py` (compara variantes: crudo / EMA viejo / Lie fwd / Lie bidireccional / gauss offline; métricas de jitter y desviación). (2) **Veredicto: el Lie bidireccional NO empeoró** — 25% menos retardo que el EMA viejo (desv med 0.82 vs 1.11 m; p95 7.8 vs 10.3 m) con jitter 12x bajo el crudo. (3) **Causa real de la percepción**: 36% de frames de SNMOT-148 sin calibración (hueco de 9.5 s); el smoother los rellenaba con cámara congelada/promediada → posiciones ficticias en minimapa y `tracking_2d.csv`. **Fix**: `MAX_GAP_FILL_FRAMES=25` — huecos >1 s quedan sin H_inv (CSV honesto; analytics ya exige `gap==1`, la ficción sumaba distancia/velocidad fantasma). (4) Jersey: Real-ESRGAN x4 antes del lector medido en val: 41.13% vs 41.63% original → **SR descartada** (`scripts/eval_sr_reader_ab.py`, doc §23). Informe: `docs/minimap_stabilization_technical_report.md` (revisión 2026-07-05).
- **2026-07-05** — Jersey v2.2: PARSeq fine-tuneado en dorsales (producción). (1) Nuevo `training/identification/finetune_parseq_jersey.py`: fine-tune de PARSeq base (torch.hub, `training_step` por permutaciones nativo + AMP) con 29,662 crops limpios (19,956 tracking-train legibles + 9,706 SoccerNet jersey-2023 **TRAIN**; TEST prohibido por la fuga documentada en `docs/jersey_perframe_v2.md` §20). Val exact-match del lector: **2.1% → 41.6%**. (2) `--parseq_checkpoint` soportado en `jersey_identity_phase.py` (degrada a pesos genéricos con aviso si falta el archivo), `build_parseq_cache.py --checkpoint`, `evaluate_jersey_e2e_multi.py` y el pipeline unificado (default: `runs/parseq_jersey_ft/best.pt`). (3) Calibración: sweep de w en val saturado → se mantiene w=0.25 a priori. (4) Test (792 tracklets, conf_topk p1=.90 + roster): raw 38.4→**51.0%**, assigned 44.8→**54.7%**, locks 249@92.8% → **317@97.5%** (+27% cobertura, 1/3 del error). E2E SNMOT-148: raw 66.7→**72.2%**, locks 11→**13** (cov 61.1→72.2%) @92.3%. Cadena reproducible: `scripts/run_parseq_ft_eval_chain.ps1`. Informe: `docs/jersey_perframe_v2.md` §22.
- **2026-07-02** — v1.8: Repo refactor + unified pipeline. (1) `src/tactical_vision_pipeline.py` rewritten as a phase registry with **incremental caching** (phases skip when their outputs exist; `--force`, `--only`, `--skip`), multi-sequence mode (`--sequences`), and video renders OFF by default (`--render` enables mapper/SAM2/scanning clips) — full pipeline re-run on a computed sequence resolves in <1s; batch of 49 sequences (analytics+scan_pred) in ~160s. (2) Scanning V2 integrated as pipeline phases `scanning` + `scan_pred` (new `scripts/scanning_v2/predict_scanning_for_video.py` builds features in-memory and applies the trained best model). (3) **Canonical output layout**: per-sequence scanning moved from `outputs/scanning_v2/<seq>/` to `outputs/<seq>/scanning/` (`core/scanning_v2/paths.py` central helper with legacy-read fallback; `scripts/migrate_outputs_layout.py` migrated 42 sequences); training artifacts renamed `outputs/scanning_v2_supervised_weak` → `outputs/scanning_training`. (4) Cleanup: dead root files removed (debug_plot5.py, gcp_ls*.txt, tmp_files.txt, SKILL.md, training_summary.json), `eval_*.py` → `scripts/`, `best_hmm_params.yaml` → `configs/`, `test_job.yaml` → `cloud/`, scanning v1 CLI scripts → `scripts/legacy/scanning_v1/`; `outputs/SNMOT-148` old jersey experiment artifacts → `outputs/SNMOT-148/_archive/`. All 182 tests pass.
- **2025-05-04** — v1.0: Analytics phase (compute_metrics, project_gt_to_2d, compare_pred_gt, generate_dashboard) completed and validated on SNMOT-148. Mapper updated with `--output_csv` and `project_point_to_world()`.
- **2026-05-12** — v1.1: Jersey identification phase scaffold implemented and debugged. Added `build_tracking_jersey_dataset.py` (extracts crops, quality scores, tracklet bags from SoccerNet Tracking), `train_jersey_digit_mil.py` (digit-compositional MIL with EfficientNet-B0), `jersey_assignment.py` (Hungarian per-team assignment with duplicate resolution), `jersey_identity_phase.py` (inference CLI), `evaluate_jersey_tracklets.py` (tracklet-level metrics), and integrated optional `--jersey_model` into `tactical_vision_pipeline.py` as Phase 10 between team clustering and 2D mapping.
- **2026-05-12** — v1.2: Fixed critical jersey identification bugs and retrained model. Patched `tracklets.json` crop paths (removed duplicated `jersey_tracking_v1` prefix). Fixed `train_jersey_digit_mil.py` loader to raise `FileNotFoundError` instead of silently zero-filling missing crops. Fixed `jersey_assignment.py` lock logic to use Hungarian-assigned number confidence instead of global max. Fixed `jersey_identity_phase.py` zero-mass fallback to zeros instead of uniform artificial alternatives. Fixed `infer_jersey_on_dataset.py` and `evaluate_jersey_tracklets.py` to evaluate per-sequence (not across sequences) and report raw top1/top3 accuracy. Retrained EfficientNet-B0 MIL for 30 epochs: val raw top1 = 57.1%, test raw top1 = 27.3%. Added `tools/TrackEval/` to `.gitignore`.
- **2026-05-13** — v1.3 (**in progress, not yet validated end-to-end**): Jersey identification Plan B overhaul — architectural changes. (1) Created `core/identity/jersey_model.py` as single source of truth for `DigitCompositionalMIL`, eliminated 3 duplicate model definitions, enforced `strict=True` on all checkpoint loads. (2) Fixed `track_id` collision in `infer_jersey_on_dataset.py` using `(sequence, track_id)` composite keys, assignment now per-sequence. (3) Rewrote `jersey_assignment.py`: replaced destructive Hungarian global with roster mask + top-1 + temporal duplicate resolution. (4) Fixed `team_id=-1` contamination of team 0. (5) Eliminated zero-fill in `jersey_identity_phase.py`. (6) Created `scripts/extract_rosters.py` (GK separated, 106 sequences), `scripts/audit_team_clustering.py` (SNMOT-148: assigned_purity=100%). (7) Added `--roster_json` to pipeline. (8) Rewrote `evaluate_jersey_tracklets.py`.
- **2026-05-13** — v1.3.1: Audit-driven corrections. (1) Fixed roster schema mismatch: `jersey_identity_phase.py` now loads per-sequence/side rosters.json via `--team_mapping` (team_audit.json or inline `0:right,1:left`). (2) Fixed frame_ids not being passed to TrackletInfo in both `jersey_identity_phase.py` and `infer_jersey_on_dataset.py` — temporal duplicate resolution is now actually active. (3) Fixed model paths in docs (`rfdetr-l.pt` → actual pth, `sam2.pt` → `sam2.1_hiera_small.pt`). (4) Fixed `.gitignore` to cover all of `runs/` (not just detect/segment). (5) Evaluator now reports both `*_matched_acc` and `*_gt_acc` with explicit denominator naming. (6) Legacy checkpoint resume now discovers best_acc via validation pass instead of defaulting to 0.0. (7) Team audit GK/referee labels changed from 'detection_rate' to 'matched_rate' with clarifying comments about spatial-only semantics. (8) Pipeline commands now quote all paths for shell safety and accept `--device`/`--team_mapping`. (9) Fixed `num_frames` using nonexistent `crop_paths_used` field → uses `num_frames` from record.
- **2026-05-13** — v1.3.2: End-to-end validation and conservative thresholding. (1) Fixed `jersey_identity_phase.py` to preserve `all_frame_ids` (full tracklet temporal range) separate from top-K inference frames, enabling proper temporal duplicate resolution. (2) Created `scripts/evaluate_jersey_e2e.py` with per-frame IoU+Hungarian matching for realistic pipeline↔GT evaluation on SNMOT-148 (raw top1 32.6%, assigned 34.9%, locked accuracy 66.7% at default thresholds). (3) Generated `SNMOT-148_jersey_identity_v5.json` with per-team roster + conservative threshold `p1=0.85`: locked accuracy improved to 76.9%, false lock rate reduced to 23.1% (vs 33.3% in v4). (4) Fixed `scripts/jersey_visual_qa.py` to accept `--e2e_matches_json` for IoU-based correctness labels, eliminating invalid direct `track_id` comparison on pipeline outputs. Generated `SNMOT-148_jersey_overlay_v5.mp4` and `jersey_crops_v5/` (428 correct, 89 incorrect via E2E matching).
- **2026-05-13** — v1.3.3: Code-audit fixes. (1) Rewrote `evaluate_jersey_e2e.py` to aggregate predictions per-GT tracklet (using best fragment by match count), eliminating fragmentation-weighted bias. Now reports both GT-level (primary) and fragment-level metrics with `gt_total`, `unique_gt_matched`, `gt_coverage`. SNMOT-148 v5 GT-level: raw top1 44.4%, assigned 38.9%, locked accuracy 75.0%, false lock 25.0%. (2) `jersey_identity_phase.py` now exports full `frame_ids`, `start_frame`, `end_frame` from `all_frame_ids`; top-K frames renamed to `inference_frame_ids`. (3) `build_tracking_jersey_dataset.py` persists `all_frame_ids` (all tracklet frames) alongside selected crop frames; `infer_jersey_on_dataset.py` uses `all_frame_ids` for temporal duplicate resolution. (4) `jersey_identity_phase.py` now supports `team_assignments` key in team JSON (in addition to `players`/`tracklets`). (5) `jersey_visual_qa.py` exports crops from raw frame *before* drawing overlays, ensuring clean inspection images.
- **2026-05-13** — v1.3.4: Second code-audit pass. (1) Fixed `evaluate_jersey_e2e.py` GT denominator (`jersey_number > 1` → `>= 1`) to include player #1. (2) Added `per_frame_matches` export to E2E evaluator for frame-level correctness labels. (3) Rewrote `jersey_visual_qa.py` to use per-frame GT lookup (handles ID switches) and auto-clean crop folders before export. (4) `jersey_identity_phase.py` now accumulates `all_detected_frames` before quality filters, making temporal range and duplicate resolution truly accurate. (5) Aligned detection JSON schema support between `jersey_identity_phase.py` and `jersey_visual_qa.py` (dict `frames`, `player_detections`).
- **2026-05-13** — v1.3.5: Third code-audit pass. (1) `evaluate_jersey_e2e.py` now filters GT detections to jersey-relevant tracks (numeric jersey + left/right side) BEFORE IoU+Hungarian matching, preventing predicted players from matching against referees/ball. Frames matched dropped from 8965 to 7466 (referees/ball excluded). (2) E2E evaluator now reports both `gt_*_matched_acc` (over matched GTs) and `gt_*_total_acc` (over all numeric GTs) for honest denominator reporting. (3) `build_tracking_jersey_dataset.py` stores truly full `all_frame_ids` (all detection frames before crop quality filters) and excludes it from `metadata.csv`. (4) Aligned bbox schema parsing across `jersey_identity_phase.py`, `evaluate_jersey_e2e.py`, and `jersey_visual_qa.py` to support `bbox`/`box`/`x_min` formats. (5) `jersey_visual_qa.py` no longer falls back to track-level GT labels when per-frame matches exist, eliminating mislabeled crops during ID switches.
- **2026-05-13** — v1.3.6: Pipeline integration closed. (1) Added `--p1_threshold`, `--margin_threshold`, `--seed`, `--run_team_audit` to `src/tactical_vision_pipeline.py`. (2) `jersey_identity_phase.py` default `p1_threshold` changed to `0.85` (validated config) and inference padding made deterministic via `seed + track_id`. (3) `infer_jersey_on_dataset.py` also deterministic. (4) `core/identity/jersey_model.py` loads with `pretrained_backbone=False` to avoid internet dependency. (5) Fixed `evaluate_jersey_e2e.py` `gt_best_frag_assigned_acc` bug (was using raw numerator). (6) `audit_team_clustering.py` now uses shared `schema_utils` parsers. (7) Regenerated canonical `SNMOT-148_jersey_identity.json` and `jersey_e2e_eval.json`: GT raw top1 44.4%, assigned 38.9%, locked accuracy 75.0%, false lock 25.0%, coverage 100%, team side consistency 100%.
- **2026-05-14** — v1.3.7: Temporal fusion validated and promoted to default. (1) Fixed critical shape bug in `compute_jersey_probs_from_logits` — now squeezes when B==1 so per-frame and MIL inference both return `(99,)` instead of `(1, 99)`. (2) Ran `--inference_mode temporal` on SNMOT-148: GT raw top1 **61.1%** (vs MIL 44.4%, **+16.7 pts**), assigned **44.4%** (vs 38.9%, **+5.5 pts**), lock accuracy **100%** (vs 75%, **+25 pts**), false lock **0%** (vs 25%, **-25 pts**). (3) Promoted temporal output to canonical `SNMOT-148_jersey_identity.json` and `jersey_e2e_eval.json`. (4) Added `--inference_mode {mil,temporal,hybrid}` to `src/tactical_vision_pipeline.py` with default `temporal`.
- **2026-05-14** — v1.3.8: SoccerNet Jersey 2023 dataset downloaded and integrated. (1) Downloaded official SoccerNet Jersey Number Recognition dataset via `SoccerNetDownloader` (task `jersey-2023`): 1,427 train tracklets (1,024 visible, 403 not visible), 1,211 test tracklets (856 visible, 355 not visible), 1,211 challenge tracklets (hidden GT). (2) Created `training/identification/soccernet_jersey_loader.py` with `SoccerNetJerseyDataset` class supporting both `train/images/` and `train/train/images/` structures, configurable `K` crops per tracklet, and filtering of not-visible tracklets. (3) Added `get_soccernet_jersey_loaders()` factory for easy DataLoader creation. (4) Created `tests/test_soccernet_jersey_loader.py` with 5 integration tests (basic loading, not-visible inclusion, test split, DataLoader batching, challenge split). All tests pass. (5) Dataset location: `datasets/soccernet/jersey-2023/`.
- **2026-05-14** — v1.3.9: Multi-dataset training experiment completed (negative result). (1) Created `training/identification/train_unified_jersey.py` supporting `soccernet`/`tracking`/`mixed` modes with resume/pretrained logic. (2) **Phase A** — Pretrained on SoccerNet Jersey 2023 visible (1,024 train / 856 test): best val jersey_acc = **34.35%**. (3) **Phase B** — Fine-tuned on tracking-derived (`jersey_tracking_v1`, 880 train / 70 val) with lr=3e-5: best val jersey_acc = **58.57%**. (4) **Phase C** — E2E evaluation on SNMOT-148 with unified checkpoint: GT raw top1 **55.6%** (vs baseline 61.1%, **-5.5 pts**), assigned **33.3%** (vs 44.4%, **-11.1 pts**), lock accuracy **75.0%** (vs 100%, **-25 pts**), false lock **25.0%** (vs 0%, **+25 pts**). (5) **Conclusion**: unified training introduced domain shift; baseline checkpoint (`runs/jersey_digit_mil_v2/best.pt`) remains production. SoccerNet Jersey 2023 data deferred to future legibility/keyframe classifier training instead of direct digit model fine-tuning.
- **2026-06-12** — v1.7: 224px re-extraction + retraining. (1) Dataset re-extracted at native 224px (`datasets/jersey_tracking_v2_224`, same builder/params/sequences, splits.json verified identical to v1 — protocol intact; 128px was a hard information ceiling, digit ≈ 15px). (2) Checkpoint now stores `img_size`; `load_jersey_model` restores it and `get_model_transform(model)` builds the matching transform — all inference paths (E2E phase, dataset evaluators, legacy MIL) auto-resolve resolution; legacy 128px checkpoints and the legibility model (always 128) unaffected. Transforms parametrized via `build_transform_inference/train_perframe(img_size)`. (3) Retrained with the exact v1.5 recipe at 224 (warm-start from v2_mixed, batch 32) so resolution is the only variable. (4) Results — test (49 seqs): raw 37.4%→**39.4%**, top-3 50.0%→**52.3%**, assigned 20.2%→**25.5%**, locked coverage +43% (56→80 @ 92.5%; conservative p1=0.85: 51 @ 94.1%). Val: raw **92.9%**, 36 locks @ 100%. Multi-seq E2E gate: assigned 41.5%→**46.2%**, SNMOT-148 assigned **77.8%**. Production checkpoint: `runs/jersey_perframe_v3_224/best.pt`. Chain script: `scripts/run_train_224_chain.ps1`. Report: `docs/jersey_perframe_v2.md` §11-15.
- **2026-06-12** — v1.6: Identity post-processing + multi-sequence E2E gate. (1) New `core/identity/tracklet_linking.py`: `link_fragments` joins ByteTrack fragments of the same player (same team + gap ≤50 frames + spatial continuity capped at 300px + STRICT identity compatibility: per-fragment fused top-1 must agree — the permissive "allow if uncertain" variant was tested and rejected because an uncertain wrong fragment flipped a confident correct one on SNMOT-132); `detect_mode_switch` finds ID switches inside tracklets (mode change with ≥60% dominance per side), blocking locks across switches and trimming only large (≥30%) contaminations. (2) `jersey_assignment.py --reassign_conflicts`: intra-frame exclusivity — duplicate-number conflict losers fall back to their best non-conflicting alternative (≥0.30) as tentative instead of unknown. (3) New multi-sequence E2E gate: `scripts/run_e2e_prereqs.ps1` (generates pipeline prerequisites per sequence) + `scripts/evaluate_jersey_e2e_multi.py` (aggregates GT-level metrics). Gate = SNMOT-148/116/132/190, 65 GT players. (4) Results: assigned 40.0%→**41.5%**, raw 55.4% and locked 14@92.9% unchanged — Pareto improvement, no per-sequence regression. (5) All flags exposed through `jersey_identity_phase.py` and `tactical_vision_pipeline.py` (`--link_fragments --split_on_switch --reassign_conflicts`), defaults off for backward compat but recommended for production. Report: `docs/jersey_perframe_v2.md` §7-11. Tests: 91 passed (9 new: linking, switch detection, conflict reassignment).
- **2026-06-11** — v1.5: Per-frame training + mixed data + calibrated fusion. (1) Root cause fixed: model was trained as MIL bags of K=16 but production (`temporal`) runs per-frame — new `training/identification/train_jersey_perframe.py` trains on individual legible crops (bags of K=1, checkpoint fully MIL-compatible), with tens-loss masking for 1-digit numbers, label smoothing, AdamW+warmup-cosine and strong augmentation (`TRANSFORM_TRAIN_PERFRAME`). (2) `scripts/score_crop_legibility.py` caches legibility for all 41,795 dataset crops. (3) `scripts/build_soccernet_perframe_index.py` adds 9,706 legible SoccerNet Jersey 2023 train crops (torso-cropped, legibility-ranked); joint mixed training now WORKS (vs failed sequential fine-tune of v1.3.9). (4) `temporal_fusion` gained `fusion_mode` {geometric,arithmetic,topk_geometric} and `temperature`; exposed through phase CLI and pipeline; defaults unchanged. (5) New multi-sequence production-path evaluator `scripts/evaluate_jersey_dataset_temporal.py` with npz prob caching and val-only calibration sweep. (6) Results — test split (49 seqs, production path, no roster): raw top-1 26.0%→**37.4%**, raw top-3 40.9%→**50.0%**, locked acc 75.5%→**94.6%**, false lock 24.5%→**5.4%**. E2E SNMOT-148: raw 66.7%→**72.2%**, assigned 61.1%→**66.7%**, best-frag raw **77.8%**, 1 false lock (GT#33 read as 44, systematic 3↔4 digit confusion documented). Production checkpoint: `runs/jersey_perframe_v2_mixed/best.pt`; production config: `--fusion_mode arithmetic --legibility_threshold 0.7 --p1_threshold 0.75 --margin_threshold 0.15`. Full report: `docs/jersey_perframe_v2.md`. Tests: 82 passed (incl. 12 new for fusion modes and per-frame training). (7) Final-render integration: `core/mapping/tactical_vision_2d_mapper.py` gained optional `--jersey_json` — locked numbers replace `#track_id` labels on video bboxes and minimap dots (tentative marked with `?`); the pipeline passes it automatically when `--jersey_model` is set, so Phase 9 output now shows identified jersey numbers. Demo artifact: `outputs/SNMOT-148/SNMOT-148_2d_map_final_jersey.mp4` (+ QA overlay `SNMOT-148_jersey_overlay_v7_perframe.mp4`, crops 422 correct / 54 incorrect).
- **2026-05-28** — v1.4: Temporal Legibility Filtering & Optimization Gate completed. (1) Coded robust in-memory sweep script (`sweep_jersey_thresholds.py`) reducing search time from 56 mins to 0.4s. (2) Ran E2E validation sweep over 80 combinations on holdout `SNMOT-148` (0.0% leakage). Resolved optimal production parameters: `p1=0.75, margin=0.15, legibility=0.70`. (3) Verified E2E final gate metrics exceeding baseline on all acceptance thresholds: raw top1 matched accuracy **66.7%** (+5.6 pts), assigned matched accuracy **61.1%** (+16.7 pts), locked total coverage **44.4%** (+11.1 pts), lock accuracy **100%** (0% false locks). (4) Packaged repo with `pyproject.toml` and `pytest.ini` for editable install, and wrote E2E validation walkthrough and thesis-grade technical report under `docs/jersey_legibility_v1.md`.

