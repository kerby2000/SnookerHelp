from __future__ import annotations

from typing import Protocol


class RoughDetector(Protocol):
    """Protocol for rough candidate detectors.

    v1 keeps the current classical detector behind this interface so final
    evidence/physics code does not depend on how candidates were proposed.
    """

    def detect(self, image, background):  # noqa: ANN001, ANN201
        ...


__all__ = ["RoughDetector"]
