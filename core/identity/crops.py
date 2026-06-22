"""
Centralized crop extraction and quality assessment utilities.
"""

import math
import cv2
import numpy as np


def extract_torso_crop_from_image(image, bbox):
    """Extract torso crop from image given bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    top = int(y1 + (y2 - y1) * 0.10)
    bottom = int(y1 + (y2 - y1) * 0.70)
    left = int(x1 + (x2 - x1) * 0.10)
    right = int(x1 + (x2 - x1) * 0.90)
    if top >= bottom or left >= right:
        return None
    
    h_img, w_img = image.shape[:2]
    top = max(0, min(top, h_img))
    bottom = max(0, min(bottom, h_img))
    left = max(0, min(left, w_img))
    right = max(0, min(right, w_img))
    
    if top >= bottom or left >= right:
        return None
        
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def extract_candidate_crops_from_image(image, bbox):
    """
    Extract 6 crop variants:
      1. Torso actual (Standard Torso)
      2. Torso superior (Upper Torso)
      3. Torso medio (Middle Torso)
      4. Bbox completo con padding (Padded Bbox)
      5. Crop ancho (Wide Crop)
      6. Crop bajo (Low Crop)
    Returns:
      List of dicts: [{"name": str, "crop": np.ndarray}]
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    w = x2 - x1
    h = y2 - y1
    
    variants = {}
    
    # 1. Torso actual
    t1, b1 = int(y1 + h * 0.10), int(y1 + h * 0.70)
    l1, r1 = int(x1 + w * 0.10), int(x1 + w * 0.90)
    variants["torso_actual"] = (t1, b1, l1, r1)
    
    # 2. Torso superior
    t2, b2 = int(y1 + h * 0.05), int(y1 + h * 0.50)
    l2, r2 = int(x1 + w * 0.15), int(x1 + w * 0.85)
    variants["torso_superior"] = (t2, b2, l2, r2)
    
    # 3. Torso medio
    t3, b3 = int(y1 + h * 0.15), int(y1 + h * 0.60)
    l3, r3 = int(x1 + w * 0.15), int(x1 + w * 0.85)
    variants["torso_medio"] = (t3, b3, l3, r3)
    
    # 4. Bbox completo con padding
    t4, b4 = int(y1 - h * 0.10), int(y2 + h * 0.10)
    l4, r4 = int(x1 - w * 0.10), int(x2 + w * 0.10)
    variants["padded_bbox"] = (t4, b4, l4, r4)
    
    # 5. Crop ancho
    t5, b5 = int(y1 + h * 0.10), int(y1 + h * 0.70)
    l5, r5 = int(x1), int(x2)
    variants["crop_ancho"] = (t5, b5, l5, r5)
    
    # 6. Crop bajo
    t6, b6 = int(y1 + h * 0.20), int(y1 + h * 0.80)
    l6, r6 = int(x1 + w * 0.10), int(x1 + w * 0.90)
    variants["crop_bajo"] = (t6, b6, l6, r6)
    
    results = []
    for name, (top, bottom, left, right) in variants.items():
        # Clamp to image boundaries
        top = max(0, min(top, h_img))
        bottom = max(0, min(bottom, h_img))
        left = max(0, min(left, w_img))
        right = max(0, min(right, w_img))
        
        if top >= bottom or left >= right:
            crop = None
        else:
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                crop = None
        
        results.append({"name": name, "crop": crop})
        
    return results


def compute_crop_quality_rgb(crop):
    """Compute crop quality after converting RGB to BGR to avoid color space issues."""
    if crop is None or crop.size == 0:
        return 0.0
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    return compute_crop_quality(bgr)


def compute_crop_quality(crop):
    """Compute quality score for a crop (higher = better for jersey reading)."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    size_score = min(1.0, crop.shape[0] * crop.shape[1] / (64 * 64))
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_score = min(1.0, math.log1p(laplacian_var) / 8.0)
    contrast = gray.std()
    contrast_score = min(1.0, contrast / 50.0)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (255 * edges.size)
    edge_score = min(1.0, edge_density * 20)
    brightness = gray.mean()
    brightness_penalty = max(0, abs(brightness - 128) / 128)
    border_r = min(crop.shape[1], crop.shape[0]) / max(crop.shape[1], crop.shape[0])
    border_penalty = max(0, 0.2 - border_r) / 0.2 if border_r < 0.2 else 0
    quality = max(0.0, size_score + lap_score + contrast_score + edge_score - brightness_penalty - border_penalty)
    return quality
