from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorMeasurement:
    label: str
    hsv: tuple[int, int, int]
    lab: tuple[int, int, int]
    confidence: float


class BallColorClassifier:
    """Rules calibrated for the supplied fixed-light Sony image set."""

    def classify(
        self,
        hsv_pixels: np.ndarray,
        lab_pixels: np.ndarray,
        highlight_value_limit: int,
        minimum_non_highlight_pixels: int,
    ) -> ColorMeasurement:
        keep = hsv_pixels[:, 2] < highlight_value_limit
        if int(np.count_nonzero(keep)) < minimum_non_highlight_pixels:
            keep = np.ones(len(hsv_pixels), dtype=bool)

        h, s, v = (
            int(value)
            for value in np.median(hsv_pixels[keep], axis=0)
        )
        lightness, a_channel, b_channel = (
            int(value)
            for value in np.median(lab_pixels[keep], axis=0)
        )

        label, confidence = self._classify_values(
            h, s, v, lightness, a_channel, b_channel
        )
        return ColorMeasurement(
            label=label,
            hsv=(h, s, v),
            lab=(lightness, a_channel, b_channel),
            confidence=confidence,
        )

    @staticmethod
    def _classify_values(
        h: int,
        s: int,
        v: int,
        lightness: int,
        a_channel: int,
        b_channel: int,
    ) -> tuple[str, float]:
        del lightness

        if v < 105:
            return "black", min(0.99, 0.70 + (105 - v) / 120.0)

        if s < 75 and v > 180:
            if a_channel >= 131 and b_channel <= 136:
                return "pink", 0.82
            return "white", 0.88

        if s < 100:
            return "unknown", 0.35

        if (
            (h < 10 or h >= 160)
            and v > 225
            and s < 150
            and a_channel < 175
            and b_channel < 165
        ):
            return "pink", 0.76

        if 92 <= h <= 112:
            return "blue", 0.90
        if 75 <= h < 92 and a_channel <= 135:
            return "green", 0.90
        if 75 <= h < 92 and a_channel >= 160 and b_channel >= 145:
            return "red", 0.72
        if 20 <= h < 40:
            return "yellow", 0.90
        if 10 <= h < 22:
            return "brown", 0.84
        if h < 10 or h >= 160:
            return "red", 0.92
        return "unknown", 0.30

    @staticmethod
    def display_bgr(label: str) -> tuple[int, int, int]:
        return {
            "white": (245, 245, 245),
            "red": (40, 40, 235),
            "yellow": (0, 230, 255),
            "green": (50, 180, 30),
            "brown": (20, 90, 150),
            "blue": (220, 120, 30),
            "pink": (180, 150, 255),
            "black": (20, 20, 20),
            "unknown": (180, 180, 180),
        }.get(label, (180, 180, 180))


def convert_color_spaces(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV),
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB),
    )
