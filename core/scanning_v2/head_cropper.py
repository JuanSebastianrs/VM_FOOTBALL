# core/scanning_v2/head_cropper.py
"""
FASE 4 — Head crop del jugador receptor.

Estrategia (en orden):
  1. Si hay keypoints de cabeza (nariz/ojos/orejas) -> bbox de cabeza ajustada.
  2. Si no -> parte superior del bbox del jugador (top_bbox_ratio).
Se expande con margen, se recorta dentro de la imagen y se valida tamano/nitidez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

_HEAD_LM = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


@dataclass
class HeadCrop:
    crop: Optional[np.ndarray]
    bbox: Optional[np.ndarray]      # [x1,y1,x2,y2] en frame
    valid: bool
    width: int
    height: int
    quality: float                  # [0,1]
    source: str                     # "keypoints" | "bbox_top" | "invalid"


class HeadCropper:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.top_ratio = float(c.get("top_bbox_ratio", 0.35))
        self.expand = float(c.get("expand_ratio", 0.25))
        self.min_h = int(c.get("min_head_height_px", 32))

    def _quality(self, crop: np.ndarray) -> float:
        h = crop.shape[0]
        size_term = float(np.clip((h - self.min_h) / (3 * self.min_h), 0.0, 1.0))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_term = float(np.clip(blur / 120.0, 0.0, 1.0))   # >120 var = nitido
        return float(0.5 * size_term + 0.5 * blur_term)

    def crop(self, frame: np.ndarray, player_bbox: np.ndarray,
             head_points_img: Optional[Dict[str, tuple]] = None) -> HeadCrop:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in player_bbox]
        bw, bh = x2 - x1, y2 - y1
        source = "bbox_top"

        if head_points_img:
            xs = [p[0] for p in head_points_img.values()]
            ys = [p[1] for p in head_points_img.values()]
            if len(xs) >= 2:
                hx1, hx2 = min(xs), max(xs)
                hy1, hy2 = min(ys), max(ys)
                # margen vertical extra hacia arriba (frente/pelo)
                pad_x = self.expand * max(hx2 - hx1, 0.3 * bw)
                pad_y = self.expand * max(hy2 - hy1, 0.3 * bw)
                x1, y1, x2, y2 = hx1 - pad_x, hy1 - 1.4 * pad_y, hx2 + pad_x, hy2 + pad_y
                source = "keypoints"
            else:
                head_points_img = None

        if source == "bbox_top":
            hh = self.top_ratio * bh
            ex = self.expand * bw
            x1, y1, x2, y2 = x1 - ex, y1 - 0.1 * hh, x2 + ex, y1 + hh + 0.1 * hh

        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(W, int(x2)), min(H, int(y2))
        if ix2 - ix1 < 4 or iy2 - iy1 < 4:
            return HeadCrop(None, None, False, 0, 0, 0.0, "invalid")
        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return HeadCrop(None, None, False, 0, 0, 0.0, "invalid")
        h, w = crop.shape[:2]
        valid = h >= self.min_h
        q = self._quality(crop) if valid else 0.0
        return HeadCrop(crop, np.array([ix1, iy1, ix2, iy2], float),
                        valid, w, h, q, source if valid else "invalid")
