# core/scanning/player_cropper.py
"""
Recorte de jugadores a partir de un frame + bbox + track_id.

Expande la bbox (para capturar cabeza y hombros completos), la recorta dentro de
los limites de la imagen, la redimensiona a un tamano fijo y reporta metadatos de
calidad. No rompe el pipeline ante crops invalidos: los marca como tales.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CropResult:
    frame_id: int
    track_id: int
    bbox_original: np.ndarray         # [x1, y1, x2, y2]
    bbox_expanded: np.ndarray         # [x1, y1, x2, y2]
    crop: Optional[np.ndarray]        # BGR, redimensionado
    crop_path: Optional[str]
    valid_crop: bool
    low_quality: bool
    orig_height: float                # alto de la bbox original (px), proxy de lejania

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "bbox_original": self.bbox_original.tolist(),
            "bbox_expanded": self.bbox_expanded.tolist(),
            "crop_path": self.crop_path,
            "valid_crop": self.valid_crop,
            "low_quality": self.low_quality,
        }


class PlayerCropper:
    """
    Parameters
    ----------
    expand_ratio : float        fraccion de expansion de la bbox (0.15 - 0.30)
    crop_size : (w, h)          tamano de salida del crop
    min_crop_height : int       alto minimo (px) de la bbox original -> valido
    low_quality_height : int    por debajo de este alto -> low_quality
    save_dir : path | None      si se da, guarda los crops como debug
    """

    def __init__(
        self,
        expand_ratio: float = 0.25,
        crop_size: Tuple[int, int] = (384, 768),
        min_crop_height: int = 64,
        low_quality_height: int = 110,
        save_dir: Optional[str | Path] = None,
    ):
        self.expand_ratio = float(expand_ratio)
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.min_crop_height = int(min_crop_height)
        self.low_quality_height = int(low_quality_height)
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def crop(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        frame_id: int,
        track_id: int,
        save_debug: bool = False,
    ) -> CropResult:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        orig_height = bh

        # bbox invalida o demasiado pequena
        if bw <= 1 or bh <= 1 or bh < self.min_crop_height:
            return CropResult(
                frame_id, track_id, np.asarray(bbox, dtype=np.float64),
                np.asarray(bbox, dtype=np.float64), None, None,
                valid_crop=False, low_quality=True, orig_height=orig_height,
            )

        # Expansion (un poco mas hacia arriba para garantizar cabeza)
        ex, ey = self.expand_ratio * bw, self.expand_ratio * bh
        ex1 = max(0.0, x1 - ex)
        ey1 = max(0.0, y1 - ey * 1.3)
        ex2 = min(float(W), x2 + ex)
        ey2 = min(float(H), y2 + ey * 0.5)
        bbox_exp = np.array([ex1, ey1, ex2, ey2], dtype=np.float64)

        crop = frame[int(ey1):int(ey2), int(ex1):int(ex2)]
        if crop.size == 0:
            return CropResult(
                frame_id, track_id, np.asarray(bbox, dtype=np.float64), bbox_exp,
                None, None, valid_crop=False, low_quality=True,
                orig_height=orig_height,
            )

        crop_resized = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
        low_quality = orig_height < self.low_quality_height

        crop_path = None
        if save_debug and self.save_dir is not None:
            crop_path = str(self.save_dir / f"f{frame_id:06d}_t{track_id:04d}.jpg")
            cv2.imwrite(crop_path, crop_resized)

        return CropResult(
            frame_id, track_id, np.asarray(bbox, dtype=np.float64), bbox_exp,
            crop_resized, crop_path, valid_crop=True, low_quality=low_quality,
            orig_height=orig_height,
        )
