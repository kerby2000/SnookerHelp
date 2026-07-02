from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")


__all__ = ["read_bgr", "write_image"]
