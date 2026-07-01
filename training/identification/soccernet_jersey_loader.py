"""
SoccerNet Jersey 2023 Dataset Loader
Integrates official SoccerNet Jersey dataset with existing training pipeline.

Dataset structure:
  datasets/soccernet/jersey-2023/
    train/train/images/{player_id}/{player_id}_{frame_num}.jpg
    train/train_gt.json -> {player_id: jersey_number}
    test/test/images/{player_id}/{player_id}_{frame_num}.jpg
    test/test_gt.json -> {player_id: jersey_number}
    challenge/challenge/images/{player_id}/... (no GT)

Usage:
    from training.identification.soccernet_jersey_loader import SoccerNetJerseyDataset
    dataset = SoccerNetJerseyDataset('datasets/soccernet/jersey-2023', split='train')
"""

import json
import random
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.identity.jersey_model import TRANSFORM_TRAIN, TRANSFORM_INFERENCE


class SoccerNetJerseyDataset(Dataset):
    """
    Dataset for SoccerNet Jersey Number Recognition.
    
    Each item is a tracklet (player) with K random crops.
    Returns: (K, 3, H, W) tensor and jersey number label.
    """
    
    def __init__(self, root_dir: str, split: str = "train", K: int = 16, 
                 transform=None, min_frames: int = 1, exclude_not_visible: bool = False):
        """
        Args:
            root_dir: Path to datasets/soccernet/jersey-2023
            split: 'train', 'test', or 'challenge'
            K: Number of crops per tracklet (bag size)
            transform: Optional transform to apply
            min_frames: Minimum frames required per tracklet
            exclude_not_visible: If True, exclude tracklets with jersey_number == -1
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.K = K
        self.transform = transform or TRANSFORM_TRAIN
        self.min_frames = min_frames
        self.exclude_not_visible = exclude_not_visible
        
        # Load ground truth if available
        self.ground_truth = {}
        # Try both structures: train/train_gt.json or train_gt.json
        gt_path = self.root_dir / split / f"{split}_gt.json"
        if not gt_path.exists():
            gt_path = self.root_dir / f"{split}_gt.json"
        if gt_path.exists():
            with open(gt_path) as f:
                self.ground_truth = json.load(f)
        
        # Build tracklet list
        # Try both structures: train/images/ or train/train/images/
        images_dir = self.root_dir / split / "images"
        if not images_dir.exists():
            images_dir = self.root_dir / split / split / "images"
        
        self.tracklets = []
        
        if images_dir.exists():
            for player_dir in sorted(images_dir.iterdir()):
                if not player_dir.is_dir():
                    continue
                player_id = player_dir.name
                image_files = sorted(player_dir.glob("*.jpg"))
                
                if len(image_files) < self.min_frames:
                    continue
                
                # Get jersey number
                jersey_number = self.ground_truth.get(player_id, -1)
                
                # Exclude not visible if requested
                if self.exclude_not_visible and jersey_number == -1:
                    continue
                
                self.tracklets.append({
                    "player_id": player_id,
                    "image_files": image_files,
                    "jersey_number": jersey_number,
                    "num_frames": len(image_files),
                })
        
        print(f"SoccerNet Jersey {split}: {len(self.tracklets)} tracklets loaded")
        if self.ground_truth:
            visible = sum(1 for t in self.tracklets if t["jersey_number"] != -1)
            print(f"  Visible jerseys: {visible}, Not visible: {len(self.tracklets) - visible}")
    
    def __len__(self):
        return len(self.tracklets)
    
    def __getitem__(self, idx):
        tracklet = self.tracklets[idx]
        image_files = tracklet["image_files"]
        jersey_number = tracklet["jersey_number"]
        
        # Sample K crops (with replacement if not enough)
        selected_files = random.choices(image_files, k=self.K) if len(image_files) < self.K else random.sample(image_files, k=self.K)
        
        crops = []
        for img_path in selected_files:
            img = cv2.imread(str(img_path))
            if img is None:
                # Fallback to blank image
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            crop_tensor = self.transform(img)
            crops.append(crop_tensor)
        
        # Stack to (K, 3, H, W)
        crops_tensor = torch.stack(crops)
        
        # Create target
        # jersey_number: -1 = not visible, 1-99 = valid numbers
        # For model: we use 0-98 for numbers 1-99, and -1 for not visible
        if jersey_number == -1:
            target = -1  # Will be handled in loss computation
        else:
            target = jersey_number - 1  # 0-indexed for numbers 1-99
        
        return crops_tensor, target, tracklet["player_id"]


def get_soccernet_jersey_loaders(root_dir: str, K: int = 16, batch_size: int = 8, 
                                  num_workers: int = 4, exclude_not_visible: bool = True):
    """
    Create DataLoaders for SoccerNet Jersey dataset.
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = SoccerNetJerseyDataset(
        root_dir=root_dir,
        split="train",
        K=K,
        transform=TRANSFORM_TRAIN,
        exclude_not_visible=exclude_not_visible,
    )
    
    test_dataset = SoccerNetJerseyDataset(
        root_dir=root_dir,
        split="test",
        K=K,
        transform=TRANSFORM_INFERENCE,
        exclude_not_visible=exclude_not_visible,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Quick test
    dataset = SoccerNetJerseyDataset(
        root_dir="datasets/soccernet/jersey-2023",
        split="train",
        K=16,
        exclude_not_visible=True,
    )
    
    if len(dataset) > 0:
        crops, target, player_id = dataset[0]
        print(f"Sample: player_id={player_id}, target={target}, crops_shape={crops.shape}")
    else:
        print("Dataset is empty!")
