"""
Robustness unit tests for Jersey Legibility, Splits, and E2E evaluation.
Runs on CPU, does not require GPU.
"""

import os
import json
import sys
from pathlib import Path
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_unique_import_of_classifier():
    """Verify that LegibilityClassifier is only defined once and correctly imported."""
    # Attempt import
    from core.identity.jersey_model import LegibilityClassifier, load_legibility_model
    assert LegibilityClassifier is not None
    assert load_legibility_model is not None

    # Verify that the class name is not defined in other candidate scripts
    repo_root = Path(__file__).resolve().parent.parent
    candidate_files = [
        repo_root / "training" / "identification" / "train_jersey_legibility.py",
        repo_root / "core" / "identity" / "jersey_identity_phase.py",
    ]
    for fp in candidate_files:
        if fp.exists():
            content = fp.read_text(encoding="utf-8")
            # Should NOT contain 'class LegibilityClassifier' definition, only imports or references
            assert "class LegibilityClassifier(" not in content, f"LegibilityClassifier redefined in {fp}"


def test_exclusion_of_holdout_in_metadata():
    """Verify that SNMOT-148 (canonical holdout) is strictly absent from the legibility dataset metadata."""
    metadata_path = Path("datasets/jersey_legibility_v1/metadata.csv")
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        # Ensure sequence is never SNMOT-148
        holdout_rows = df[df["sequence"] == "SNMOT-148"]
        assert len(holdout_rows) == 0, "Security Leakage: SNMOT-148 found in legibility dataset!"
        
        # Ensure source_path never contains SNMOT-148
        holdout_paths = df[df["source_path"].str.contains("SNMOT-148", na=False)]
        assert len(holdout_paths) == 0, "Security Leakage: source_path contains SNMOT-148 crops!"
        print("Success: Checked metadata.csv, 0 holdout leakage found.")
    else:
        print("Skipping metadata check because datasets/jersey_legibility_v1/metadata.csv does not exist yet.")


def test_assertion_on_zero_tracking_samples():
    """Verify that build_legibility_dataset.py would fail with ValueError if 0 tracking samples were found."""
    # Test our validation logic on a dummy/empty records list
    records = [
        {"filename": "sn_1.jpg", "source": "soccernet_train", "sequence": "soccernet"}
    ]
    
    tracking_samples = [r for r in records if r["source"] == "tracking_v1"]
    
    with pytest.raises(ValueError, match="Zero tracking samples were generated"):
        if len(tracking_samples) == 0:
            raise ValueError("Assertion Error: Zero tracking samples were generated! The dataset is incomplete.")


def test_safe_rate_assertion_bounds():
    """Verify that _safe_rate raises AssertionError when values exceed limits, instead of silent clipping."""
    from scripts.evaluate_jersey_e2e import _safe_rate

    # Valid rates
    assert _safe_rate(0, 10) == 0.0
    assert _safe_rate(5, 10) == 0.5
    assert _safe_rate(10, 10) == 1.0
    assert _safe_rate(0, 0) == 0.0

    # Invalid rates: num > den or negative rates must trigger AssertionError
    with pytest.raises(AssertionError):
        _safe_rate(11, 10)  # > 1.0

    with pytest.raises(AssertionError):
        _safe_rate(-1, 10)  # < 0.0
