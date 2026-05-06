# VM_FOOTBALL — TacticalVision AI
## Agent Onboarding Guide v1.0

### 1. What is this project?
A master's thesis building an end-to-end computer vision pipeline that turns single-view broadcast football footage into structured tactical data: player and ball tracking, team classification, metric pitch coordinates, physical metrics, and interactive analytics.

**Reference clip for validation:** `SNMOT-148` (750 frames, 25 FPS, Goal action).

### 2. Current Snapshot
| Phase | Status | Key output files |
|---|---|---|
| Ball detection (YOLO26) | Functional | `models/yolo26.pt` |
| Player detection (RF-DETR 3-class) | Functional | `models/rfdetr-l.pt` |
| Tracking (ByteTrack + Viterbi HMM) | Functional | `*_detections.json`, `*_trajectory.json` |
| Camera motion compensation (CMC) | Functional | `*_cmc.json` |
| Team clustering (HSV + role-aware GK) | Functional | `*_team_assignments.json` |
| **2D field mapping (PnLCalib + Lie smoother)** | **Functional** | `*_tracking_2d.csv` |
| **Physical & shape analytics** | **Functional** | `*_player_physical_metrics.csv`, `*_team_shape_metrics.csv` |
| **Pred vs GT comparison** | **Functional** | `comparison/comparison_summary.json` |
| **Interactive dashboard** | **Functional** | `dashboard/index.html` |
| SAM2 segmentation | Functional | Video rendering + masks |
| Jersey number reading | Planned | `core/identity/` |
| Event detection (passes, possessions) | Planned | Not yet implemented |

**Latest milestone completed:**
- Phase 10/11: Analytics extraction (physical metrics, team shape, Pred vs GT comparison, HTML dashboard for `SNMOT-148`).

### 3. Pipeline phases (end-to-end)
1. **Detection** — YOLO26 (ball) + RF-DETR (players/goalkeepers/referees).
2. **CMC** — Frame-to-frame affine compensation from field keypoints.
3. **Temporal tracking** — Viterbi HMM over ball detections with Dummy Node for occlusions.
4. **Team clustering** — HSV descriptor + K-Means (k=2), with separate GK role via temporal hysteresis.
5. **2D mapping** — PnLCalib per-frame calibration → bidirectional SO(3) Lie smoother → H_inv homography → world coords (metres).
6. **Analytics** — Distance, speed, acceleration, sprints, team shape (centroid, width, compactness, convex hull).
7. **GT validation** — Project GT bboxes to 2D using same calibration, compare with Hungarian matching.
8. **Dashboard** — Self-contained HTML with Chart.js: KPIs, tables, time-series, 2D pitch viewer with frame slider.

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

**Coordinate conventions:**
- Pitch size: 105 m × 68 m.
- Origin: top-left corner (x right, y down).
- Time: `time_s = (frame_id - 1) / fps`, so frame 1 starts at 0.0 s.

### 6. Canonical commands
```bash
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
- **2025-05-04** — v1.0: Analytics phase (compute_metrics, project_gt_to_2d, compare_pred_gt, generate_dashboard) completed and validated on SNMOT-148. Mapper updated with `--output_csv` and `project_point_to_world()`.
