from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TableModel:
    name: str
    length_mm: float
    width_mm: float
    ball_radius_mm: float
    px_per_mm: float
    origin: str
    processing_margin_mm: float = 0.0

    @property
    def processing_margin_px(self) -> int:
        return int(round(self.processing_margin_mm * self.px_per_mm))

    @property
    def warp_width_px(self) -> int:
        return int(
            round(
                (self.length_mm + 2.0 * self.processing_margin_mm)
                * self.px_per_mm
            )
        )

    @property
    def warp_height_px(self) -> int:
        return int(
            round(
                (self.width_mm + 2.0 * self.processing_margin_mm)
                * self.px_per_mm
            )
        )

    @property
    def ball_radius_px(self) -> float:
        return self.ball_radius_mm * self.px_per_mm

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TableModel":
        surface = config["table"]["playing_surface"]
        return cls(
            name=str(config["table"]["name"]),
            length_mm=float(surface["length_mm"]),
            width_mm=float(surface["width_mm"]),
            ball_radius_mm=float(config["balls"]["radius_mm"]),
            px_per_mm=float(config["warp"]["px_per_mm"]),
            origin=str(config["coordinates"]["origin"]),
            processing_margin_mm=float(
                config["warp"].get("processing_margin_mm", 0.0)
            ),
        )


__all__ = ["TableModel"]
