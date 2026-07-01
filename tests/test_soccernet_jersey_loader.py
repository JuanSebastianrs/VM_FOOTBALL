"""
Integration test for SoccerNet Jersey 2023 dataset loader.
Validates:
1. Dataset loads correctly
2. Ground truth is properly mapped
3. DataLoader works
4. Sample shapes are correct
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from training.identification.soccernet_jersey_loader import (
    SoccerNetJerseyDataset, 
    get_soccernet_jersey_loaders
)


def test_dataset_basic():
    """Test basic dataset loading."""
    print("="*50)
    print("TEST 1: Basic Dataset Loading")
    print("="*50)
    
    dataset = SoccerNetJerseyDataset(
        root_dir="datasets/soccernet/jersey-2023",
        split="train",
        K=16,
        exclude_not_visible=True,
    )
    
    assert len(dataset) > 0, "Dataset is empty!"
    print(f"[OK] Train dataset loaded: {len(dataset)} tracklets")
    
    # Test single item
    crops, target, player_id = dataset[0]
    assert crops.shape == (16, 3, 128, 128), f"Expected (16, 3, 128, 128), got {crops.shape}"
    assert 0 <= target <= 98 or target == -1, f"Invalid target: {target}"
    print(f"[OK] Sample shape correct: {crops.shape}")
    print(f"[OK] Sample target: {target} (player_id: {player_id})")
    

def test_dataset_with_not_visible():
    """Test including not visible tracklets."""
    print("\n" + "="*50)
    print("TEST 2: Dataset with Not Visible Tracklets")
    print("="*50)
    
    dataset = SoccerNetJerseyDataset(
        root_dir="datasets/soccernet/jersey-2023",
        split="train",
        K=16,
        exclude_not_visible=False,
    )
    
    visible = sum(1 for t in dataset.tracklets if t["jersey_number"] != -1)
    not_visible = sum(1 for t in dataset.tracklets if t["jersey_number"] == -1)
    
    print(f"[OK] Total tracklets: {len(dataset)}")
    print(f"[OK] Visible: {visible}, Not visible: {not_visible}")
    
    # Should have some not visible
    assert not_visible > 0, "Expected some not visible tracklets"
    print("[OK] Not visible tracklets present")


def test_test_split():
    """Test test split loading."""
    print("\n" + "="*50)
    print("TEST 3: Test Split Loading")
    print("="*50)
    
    dataset = SoccerNetJerseyDataset(
        root_dir="datasets/soccernet/jersey-2023",
        split="test",
        K=16,
        exclude_not_visible=True,
    )
    
    assert len(dataset) > 0, "Test dataset is empty!"
    print(f"[OK] Test dataset loaded: {len(dataset)} tracklets")


def test_dataloader():
    """Test DataLoader integration."""
    print("\n" + "="*50)
    print("TEST 4: DataLoader Integration")
    print("="*50)
    
    train_loader, test_loader = get_soccernet_jersey_loaders(
        root_dir="datasets/soccernet/jersey-2023",
        K=16,
        batch_size=4,
        num_workers=0,  # Single process for testing
        exclude_not_visible=True,
    )
    
    # Test train loader
    batch = next(iter(train_loader))
    crops, targets, player_ids = batch
    
    assert crops.shape == (4, 16, 3, 128, 128), f"Expected (4, 16, 3, 128, 128), got {crops.shape}"
    print(f"[OK] Train batch shape: {crops.shape}")
    print(f"[OK] Train batch targets: {targets}")
    
    # Test test loader
    batch = next(iter(test_loader))
    crops, targets, player_ids = batch
    
    assert crops.shape[0] <= 4, f"Batch size should be <= 4, got {crops.shape[0]}"
    print(f"[OK] Test batch shape: {crops.shape}")


def test_challenge_split():
    """Test challenge split (no ground truth)."""
    print("\n" + "="*50)
    print("TEST 5: Challenge Split (No GT)")
    print("="*50)
    
    dataset = SoccerNetJerseyDataset(
        root_dir="datasets/soccernet/jersey-2023",
        split="challenge",
        K=16,
        exclude_not_visible=False,
    )
    
    print(f"[OK] Challenge dataset loaded: {len(dataset)} tracklets")
    print("[OK] Challenge split works without ground truth")


if __name__ == "__main__":
    print("Running SoccerNet Jersey Dataset Integration Tests\n")
    
    test_dataset_basic()
    test_dataset_with_not_visible()
    test_test_split()
    test_dataloader()
    test_challenge_split()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED")
    print("="*50)
