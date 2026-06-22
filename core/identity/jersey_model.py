"""
Centralized jersey number recognition model and utilities.

This module is the SINGLE SOURCE OF TRUTH for:
  - DigitCompositionalMIL architecture
  - Inference and training transforms
  - Model loading with strict=True

All other scripts (train, infer, identity_phase) MUST import from here.
Do NOT redefine DigitCompositionalMIL elsewhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────

def build_transform_inference(img_size=128):
    """Inference transform for a given model input size."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_transform_train_perframe(img_size=128):
    """
    Strong augmentation for per-frame digit training (train_jersey_perframe.py).
    Per-frame samples are noisier than MIL bags, so heavier spatial/photometric
    jitter + random erasing act as regularizers against tracklet label noise.
    """
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        T.RandomRotation(15),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


TRANSFORM_TRAIN = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Legacy 128px instances (the LegibilityClassifier and all v<=1.6 digit
# checkpoints were trained at 128px and keep using these)
TRANSFORM_INFERENCE = build_transform_inference(128)
TRANSFORM_TRAIN_PERFRAME = build_transform_train_perframe(128)


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class DigitCompositionalMIL(nn.Module):
    """
    Digit-compositional Multi-Instance Learning model for jersey number recognition.

    Architecture:
        - EfficientNet-B0 backbone (shared across K crop instances)
        - Attention-based MIL pooling
        - Three classification heads:
            * length: 1-digit vs 2-digit (2 classes)
            * tens:   tens digit 0-9 (10 classes)
            * ones:   ones digit 0-9 (10 classes)

    Input:  (B, K, 3, 128, 128) — B bags of K crops
    Output: (out_length, out_tens, out_ones) — logits for each head
    """

    def __init__(self, backbone="efficientnet_b0", num_classes=100, pretrained_backbone=True):
        super().__init__()
        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
            base = efficientnet_b0(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            feat_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.head_length = nn.Linear(feat_dim, 2)
        self.head_tens = nn.Linear(feat_dim, 10)
        self.head_ones = nn.Linear(feat_dim, 10)

    def forward(self, x):
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        f = self.backbone(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        f = f.view(B, K, -1)

        att = self.attention(f)
        att = F.softmax(att, dim=1)
        f_pooled = (f * att).sum(dim=1)

        out_len = self.head_length(f_pooled)
        out_tens = self.head_tens(f_pooled)
        out_ones = self.head_ones(f_pooled)
        return out_len, out_tens, out_ones


# ──────────────────────────────────────────────────────────────────────
# Loading utilities
# ──────────────────────────────────────────────────────────────────────

def compute_jersey_probs_from_logits(out_len, out_tens, out_ones):
    """
    Convert model logits to jersey number probabilities (1..99).

    Args:
        out_len: logits for length head, shape (B, 2) or (2,)
        out_tens: logits for tens head, shape (B, 10) or (10,)
        out_ones: logits for ones head, shape (B, 10) or (10,)

    Returns:
        np.ndarray of shape (B, 99) or (99,) with probabilities for numbers 1..99.
    """
    import numpy as np
    import torch.nn.functional as F

    # Handle both batch and single-case
    squeeze = False
    if out_len.dim() == 1:
        out_len = out_len.unsqueeze(0)
        out_tens = out_tens.unsqueeze(0)
        out_ones = out_ones.unsqueeze(0)
        squeeze = True

    len_probs = F.softmax(out_len, dim=1).cpu().numpy()
    tens_probs = F.softmax(out_tens, dim=1).cpu().numpy()
    ones_probs = F.softmax(out_ones, dim=1).cpu().numpy()

    B = len_probs.shape[0]
    jersey_probs = np.zeros((B, 99), dtype=np.float64)

    for t_digit in range(10):
        for o_digit in range(10):
            num = t_digit * 10 + o_digit
            if num == 0:
                continue
            idx = num - 1
            if num <= 9:
                jersey_probs[:, idx] = len_probs[:, 0] * ones_probs[:, o_digit]
            else:
                jersey_probs[:, idx] = len_probs[:, 1] * tens_probs[:, t_digit] * ones_probs[:, o_digit]

    # Normalize
    valid_mass = jersey_probs.sum(axis=1, keepdims=True)
    mask = valid_mass > 1e-8
    jersey_probs = np.where(mask, jersey_probs / valid_mass, 0.0)

    if squeeze or B == 1:
        jersey_probs = jersey_probs.squeeze(0)

    return jersey_probs


def load_jersey_model(model_path, device, backbone="efficientnet_b0", strict=True):
    """
    Load a trained DigitCompositionalMIL model.

    Args:
        model_path: Path to the .pt checkpoint file.
        device: torch.device to load onto.
        backbone: Backbone architecture name.
        strict: If True (default), raises error on mismatched keys.
                NEVER use strict=False in production.

    Returns:
        Model in eval mode.
    """
    model = DigitCompositionalMIL(backbone=backbone, pretrained_backbone=False).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)

    # Support both raw state_dict and full checkpoint dict
    img_size = 128  # all legacy checkpoints were trained at 128px
    if isinstance(state, dict) and "model_state_dict" in state:
        img_size = int(state.get("img_size", 128))
        model.load_state_dict(state["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)

    model.img_size = img_size
    model.eval()
    return model


def get_model_transform(model):
    """Inference transform matching the input size the checkpoint was trained at."""
    return build_transform_inference(getattr(model, "img_size", 128))


def save_full_checkpoint(model, optimizer, scheduler, epoch, best_acc, path, rng_states=None,
                         img_size=None):
    """
    Save a full checkpoint with all training state for proper resume.

    Args:
        model: The model to save.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        epoch: Current epoch number.
        best_acc: Best validation accuracy so far.
        path: Output path for the checkpoint.
        rng_states: Optional dict of RNG states for reproducibility.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if rng_states:
        checkpoint["rng_states"] = rng_states
    if img_size is not None:
        checkpoint["img_size"] = int(img_size)
    torch.save(checkpoint, path)


def load_full_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    """
    Load a full checkpoint and restore training state.

    Args:
        path: Checkpoint path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Device to map to.

    Returns:
        dict with 'epoch' and 'best_acc'.
    """
    state = torch.load(path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in state and state["scheduler_state_dict"]:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        return {
            "epoch": state.get("epoch", 0),
            "best_acc": state.get("best_acc", 0.0),
            "rng_states": state.get("rng_states"),
        }
    else:
        # Legacy: raw state_dict only — best_acc unknown, caller MUST validate
        import warnings
        warnings.warn(
            f"Loading legacy checkpoint (raw state_dict) from {path}. "
            f"best_acc is unknown (set to -1.0). Run a validation pass before saving "
            f"to avoid overwriting a potentially better checkpoint.",
            UserWarning,
        )
        model.load_state_dict(state, strict=True)
        return {"epoch": 0, "best_acc": -1.0, "rng_states": None}


# ──────────────────────────────────────────────────────────────────────
# Legibility Classifier Model and Loading Utility
# ──────────────────────────────────────────────────────────────────────

class LegibilityClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        try:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = efficientnet_b0(weights=weights)
        except Exception:
            base = efficientnet_b0(weights=None)
            
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        logits = self.fc(f).squeeze(1)
        return logits


def load_legibility_model(model_path, device):
    """Load a trained LegibilityClassifier model."""
    model = LegibilityClassifier(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model

