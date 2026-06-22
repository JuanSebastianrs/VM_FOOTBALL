"""
CPU unit tests for the per-frame jersey training pipeline
(training/identification/train_jersey_perframe.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.identification.train_jersey_perframe import (
    jersey_to_targets,
    build_frame_samples,
    JerseyFrameDataset,
)


class TestJerseyTargets:
    def test_single_digit(self):
        assert jersey_to_targets(7) == (0, 0, 7)

    def test_two_digit(self):
        assert jersey_to_targets(25) == (1, 2, 5)
        assert jersey_to_targets(99) == (1, 9, 9)
        assert jersey_to_targets(10) == (1, 1, 0)


class TestBuildFrameSamples:
    def _records(self):
        return [
            {"split": "train", "sequence": "S1", "track_id": 1, "jersey_number": 10,
             "crop_paths": ["crops/S1/a.jpg", "crops/S1/b.jpg", "crops/S1/c.jpg"]},
            {"split": "train", "sequence": "S1", "track_id": 2, "jersey_number": -1,
             "crop_paths": ["crops/S1/d.jpg"]},  # non-numeric GT: excluded
            {"split": "test", "sequence": "S2", "track_id": 3, "jersey_number": 5,
             "crop_paths": ["crops/S2/e.jpg"]},  # wrong split: excluded
        ]

    def test_legibility_filter(self):
        scores = {"crops/S1/a.jpg": 0.9, "crops/S1/b.jpg": 0.2, "crops/S1/c.jpg": 0.7}
        samples, skipped = build_frame_samples(self._records(), scores, "train", 0.5)
        assert len(samples) == 2  # b.jpg dropped (0.2 < 0.5)
        assert all(s["jersey"] == 10 for s in samples)
        assert skipped == 0

    def test_all_illegible_tracklet_is_skipped(self):
        scores = {"crops/S1/a.jpg": 0.1, "crops/S1/b.jpg": 0.1, "crops/S1/c.jpg": 0.1}
        samples, skipped = build_frame_samples(self._records(), scores, "train", 0.5)
        assert len(samples) == 0
        assert skipped == 1

    def test_missing_score_treated_as_illegible(self):
        samples, skipped = build_frame_samples(self._records(), {}, "train", 0.5)
        assert len(samples) == 0


class TestJerseyFrameDataset:
    def _write_img(self, path, size=(60, 40)):
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=(120, 50, 200)).save(path)

    def test_bag_of_one_shape(self, tmp_path):
        self._write_img(tmp_path / "crops" / "x.jpg")
        from core.identity.jersey_model import TRANSFORM_INFERENCE
        ds = JerseyFrameDataset(
            [{"crop_path": "crops/x.jpg", "jersey": 25}], tmp_path, TRANSFORM_INFERENCE)
        x, length, tens, ones = ds[0]
        assert x.shape == (1, 3, 128, 128)  # bag of K=1, MIL-compatible
        assert (length.item(), tens.item(), ones.item()) == (1, 2, 5)

    def test_external_root_and_torso_crop(self, tmp_path):
        ext_root = tmp_path / "ext"
        self._write_img(ext_root / "img" / "y.jpg", size=(40, 160))
        from core.identity.jersey_model import TRANSFORM_INFERENCE
        ds = JerseyFrameDataset(
            [{"crop_path": "img/y.jpg", "jersey": 7, "root": str(ext_root), "torso_crop": True}],
            tmp_path / "other_root", TRANSFORM_INFERENCE)
        x, length, tens, ones = ds[0]
        assert x.shape == (1, 3, 128, 128)
        assert (length.item(), tens.item(), ones.item()) == (0, 0, 7)


class TestMILCheckpointCompatibility:
    def test_perframe_forward_matches_mil_interface(self):
        from core.identity.jersey_model import DigitCompositionalMIL
        model = DigitCompositionalMIL(pretrained_backbone=False).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, 3, 128, 128))  # per-frame: bags of K=1
        assert out[0].shape == (2, 2) and out[1].shape == (2, 10) and out[2].shape == (2, 10)
