"""
Minimal regression tests for jersey identity pipeline.
Run: python -m pytest tests/test_jersey_identity.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from core.identity.schema_utils import extract_bbox_xyxy, load_team_map
from core.identity.jersey_assignment import (
    TrackletInfo,
    apply_roster_mask,
    select_top1,
    compute_lock_state,
    frames_overlap,
    resolve_temporal_duplicates,
    assign_jersey_numbers_simple,
    assign_per_team,
)


class TestSchemaUtils:
    def test_extract_bbox_xyxy_variants(self):
        assert extract_bbox_xyxy({"bbox": [1, 2, 3, 4]}) == [1.0, 2.0, 3.0, 4.0]
        assert extract_bbox_xyxy({"box": [1, 2, 3, 4]}) == [1.0, 2.0, 3.0, 4.0]
        assert extract_bbox_xyxy({"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}) == [1.0, 2.0, 3.0, 4.0]
        assert extract_bbox_xyxy({"bbox": [3, 2, 1, 4]}) is None  # x2 <= x1
        assert extract_bbox_xyxy({}) is None

    def test_load_team_map_variants(self):
        assert load_team_map([{"track_id": 5, "team_id": 0}]) == {5: 0}
        assert load_team_map({"team_assignments": [{"track_id": 5, "team_id": 0}]}) == {5: 0}
        assert load_team_map({"players": [{"track_id": 5, "team_id": 1}]}) == {5: 1}
        assert load_team_map({"tracklets": [{"track_id": 5, "team_id": -1}]}) == {5: -1}


class TestJerseyAssignment:
    def test_roster_mask(self):
        probs = np.ones(99, dtype=np.float64)
        masked = apply_roster_mask(probs, {1, 10, 99})
        assert masked.sum() > 0.99
        assert masked[0] > 0  # number 1
        assert masked[9] > 0  # number 10
        assert masked[98] > 0  # number 99
        assert masked[1] == 0.0  # number 2 not in roster

    def test_select_top1(self):
        probs = np.zeros(99, dtype=np.float64)
        probs[7] = 0.8  # number 8
        number, conf = select_top1(probs)
        assert number == 8
        assert conf == 0.8

    def test_lock_state(self):
        probs = np.zeros(99, dtype=np.float64)
        probs[7] = 0.9
        probs[3] = 0.1
        state, conf, margin = compute_lock_state(probs, 8, p1_threshold=0.85, margin_threshold=0.20)
        assert state == "locked"
        assert margin == 0.8

    def test_frames_overlap(self):
        assert frames_overlap([1, 2, 3], [3, 4]) is True
        assert frames_overlap([1, 2], [3, 4]) is False
        assert frames_overlap([], [1]) is False

    def test_team_id_minus_one_not_contaminates(self):
        t_unknown = TrackletInfo(track_id=99, team_id=-1, jersey_probs=np.ones(99)/99, frame_ids=[1, 2])
        t0 = TrackletInfo(track_id=1, team_id=0, jersey_probs=np.ones(99)/99, frame_ids=[1, 2])
        assignments = assign_per_team([t_unknown, t0], team_rosters={})
        assert assignments[99]["team_id"] == -1
        assert assignments[99]["state"] == "unknown"
        assert assignments[1]["team_id"] == 0

    def test_temporal_duplicate_resolution(self):
        probs = np.zeros(99, dtype=np.float64)
        probs[7] = 0.9  # number 8
        t1 = TrackletInfo(track_id=1, team_id=0, jersey_probs=probs, frame_ids=[1, 2, 3])
        t2 = TrackletInfo(track_id=2, team_id=0, jersey_probs=probs, frame_ids=[3, 4, 5])
        raw = assign_jersey_numbers_simple([t1, t2], team_id=0)
        resolved = resolve_temporal_duplicates(raw, [t1, t2])
        # One of them should be unknown due to overlap on frame 3
        states = [resolved[t]["state"] for t in (1, 2)]
        assert "unknown" in states


class TestTemporalFusion:
    def _frame(self, num, conf, quality=1.0, legibility=1.0, frame_id=0):
        probs = np.full(99, (1.0 - conf) / 98, dtype=np.float64)
        probs[num - 1] = conf
        return {"frame_id": frame_id, "jersey_probs": probs,
                "quality_score": quality, "legibility_score": legibility}

    def test_fusion_modes_agree_on_clean_signal(self):
        from core.identity.jersey_identity_phase import temporal_fusion
        frames = [self._frame(10, 0.9, frame_id=i) for i in range(6)]
        for mode in ("geometric", "arithmetic", "topk_geometric"):
            fused = temporal_fusion(frames, fusion_mode=mode)
            assert int(np.argmax(fused["probs"])) + 1 == 10

    def test_arithmetic_robust_to_confident_wrong_frame(self):
        from core.identity.jersey_identity_phase import temporal_fusion
        # 5 frames agree on 10 with moderate confidence; 1 frame is
        # near-certain on 7 (prob ~1.0 elsewhere ~0). Geometric fusion is
        # dominated by the near-zero entries; arithmetic must survive.
        frames = [self._frame(10, 0.6, frame_id=i) for i in range(5)]
        bad = self._frame(7, 0.999999, frame_id=5)
        bad["jersey_probs"] = np.full(99, 1e-9, dtype=np.float64)
        bad["jersey_probs"][6] = 1.0 - 98e-9
        fused = temporal_fusion(frames + [bad], fusion_mode="arithmetic")
        assert int(np.argmax(fused["probs"])) + 1 == 10

    def test_temperature_softens_distribution(self):
        from core.identity.jersey_identity_phase import temporal_fusion
        frames = [self._frame(10, 0.9, frame_id=i) for i in range(4)]
        sharp = temporal_fusion(frames, temperature=1.0)
        soft = temporal_fusion(frames, temperature=3.0)
        assert soft["p1"] < sharp["p1"]
        assert int(np.argmax(soft["probs"])) + 1 == 10

    def test_default_matches_legacy_geometric(self):
        from core.identity.jersey_identity_phase import temporal_fusion
        frames = [self._frame(8, 0.7, quality=0.5, frame_id=i) for i in range(5)]
        default = temporal_fusion(frames)
        explicit = temporal_fusion(frames, fusion_mode="geometric", temperature=1.0)
        np.testing.assert_allclose(default["probs"], explicit["probs"])


class TestTrackletLinking:
    def _frag(self, tid, team, frames, box_start, box_end):
        return {"track_id": tid, "team_id": team, "all_frame_ids": frames,
                "all_bboxes": [box_start] + [box_start] * (len(frames) - 2) + [box_end]
                if len(frames) > 1 else [box_start]}

    def test_links_close_fragments_same_team(self):
        from core.identity.tracklet_linking import link_fragments
        a = self._frag(1, 0, [1, 2, 3, 4, 5], [100, 100, 130, 180], [200, 100, 230, 180])
        b = self._frag(2, 0, [10, 11, 12], [210, 105, 240, 185], [300, 105, 330, 185])
        groups = link_fragments([a, b])
        assert groups[1] == groups[2]

    def test_does_not_link_across_teams(self):
        from core.identity.tracklet_linking import link_fragments
        a = self._frag(1, 0, [1, 2, 3], [100, 100, 130, 180], [200, 100, 230, 180])
        b = self._frag(2, 1, [10, 11, 12], [210, 105, 240, 185], [300, 105, 330, 185])
        groups = link_fragments([a, b])
        assert groups[1] != groups[2]

    def test_does_not_link_overlapping_in_time(self):
        from core.identity.tracklet_linking import link_fragments
        a = self._frag(1, 0, [1, 2, 3, 10], [100, 100, 130, 180], [200, 100, 230, 180])
        b = self._frag(2, 0, [5, 6, 12], [210, 105, 240, 185], [300, 105, 330, 185])
        groups = link_fragments([a, b])
        assert groups[1] != groups[2]  # frames overlap -> two players on screen

    def test_does_not_link_far_apart(self):
        from core.identity.tracklet_linking import link_fragments
        a = self._frag(1, 0, [1, 2, 3], [100, 100, 130, 180], [100, 100, 130, 180])
        b = self._frag(2, 0, [5, 6, 7], [1500, 800, 1530, 880], [1500, 800, 1530, 880])
        groups = link_fragments([a, b])
        assert groups[1] != groups[2]


class TestModeSwitchDetection:
    def _frame(self, num, conf, fid):
        probs = np.full(99, (1.0 - conf) / 98, dtype=np.float64)
        probs[num - 1] = conf
        return {"frame_id": fid, "jersey_probs": probs,
                "quality_score": 1.0, "legibility_score": 1.0}

    def test_detects_clear_switch(self):
        from core.identity.tracklet_linking import detect_mode_switch
        frames = ([self._frame(14, 0.8, i) for i in range(12)] +
                  [self._frame(9, 0.8, 12 + i) for i in range(12)])
        split, lm, rm = detect_mode_switch(frames)
        assert split is not None
        assert {lm, rm} == {14, 9}

    def test_no_switch_on_consistent_tracklet(self):
        from core.identity.tracklet_linking import detect_mode_switch
        frames = [self._frame(14, 0.8, i) for i in range(24)]
        split, _, _ = detect_mode_switch(frames)
        assert split is None

    def test_noisy_but_single_mode_is_not_split(self):
        from core.identity.tracklet_linking import detect_mode_switch
        import itertools
        nums = itertools.cycle([14, 14, 14, 41])  # 25% noise, no temporal structure
        frames = [self._frame(next(nums), 0.7, i) for i in range(24)]
        split, _, _ = detect_mode_switch(frames)
        assert split is None


class TestConflictReassignment:
    def test_loser_falls_back_to_second_best(self):
        probs_a = np.zeros(99); probs_a[7] = 0.9                      # strong 8
        probs_b = np.zeros(99); probs_b[7] = 0.55; probs_b[20] = 0.40  # weak 8, alt 21
        a = TrackletInfo(track_id=1, team_id=0, jersey_probs=probs_a, frame_ids=[1, 2, 3])
        b = TrackletInfo(track_id=2, team_id=0, jersey_probs=probs_b, frame_ids=[2, 3, 4])
        asn = assign_per_team([a, b], p1_threshold=0.85, margin_threshold=0.2,
                              reassign_conflicts=True)
        assert asn[1]["predicted_number"] == 8
        assert asn[2]["predicted_number"] == 21
        assert asn[2]["state"] == "tentative"
        assert asn[2]["reassigned_from"] == 8

    def test_loser_unknown_when_no_good_alternative(self):
        probs_a = np.zeros(99); probs_a[7] = 0.9
        probs_b = np.zeros(99); probs_b[7] = 0.6; probs_b[20] = 0.1  # alt below 0.30
        a = TrackletInfo(track_id=1, team_id=0, jersey_probs=probs_a, frame_ids=[1, 2, 3])
        b = TrackletInfo(track_id=2, team_id=0, jersey_probs=probs_b, frame_ids=[2, 3, 4])
        asn = assign_per_team([a, b], reassign_conflicts=True)
        assert asn[2]["predicted_number"] is None
        assert asn[2]["state"] == "unknown"


class TestDeterminism:
    def test_predict_tracklets_padding_deterministic(self):
        import random
        from core.identity.jersey_model import DigitCompositionalMIL
        import torch
        model = DigitCompositionalMIL(pretrained_backbone=False)
        model.eval()
        # Fake tracklet with 2 crops
        crops = [torch.randn(3, 128, 128) for _ in range(2)]
        seed = 42
        tid = 7
        rng1 = random.Random(seed + tid)
        padded1 = crops + [crops[rng1.randrange(len(crops))] for _ in range(14)]
        rng2 = random.Random(seed + tid)
        padded2 = crops + [crops[rng2.randrange(len(crops))] for _ in range(14)]
        assert all(torch.equal(a, b) for a, b in zip(padded1, padded2))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
