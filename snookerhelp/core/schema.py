from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ConfidenceLevel = Literal["high", "medium", "low", "needs_review", "unknown"]
CoordinateSystem = Literal["source_px", "table_mm", "warped_px"]


@dataclass(slots=True)
class ImageModel:
    """2D image evidence extracted from source pixels."""

    model_type: str
    source: str
    center_px: list[float] | None = None
    major_axis_px: float | None = None
    minor_axis_px: float | None = None
    angle_deg: float | None = None
    axis_ratio: float | None = None
    residual_px: float | None = None
    point_count: int = 0
    quality: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class PhysicalModel:
    """Physical camera/table/ball model evidence for one estimate."""

    model_type: str
    camera_model: str | None = None
    approximate: bool = True
    status: str = "unknown"
    projection_mode: str = "forward"
    projected_center_px: list[float] | None = None
    projected_outline_px: list[list[float]] = field(default_factory=list)
    residual_px: float | None = None
    residual_grade: str = "unknown"
    observed_source: str | None = None
    z_mm: float | None = None
    optimization: dict[str, Any] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class Confidence:
    """Structured confidence for a final estimate."""

    score: float | None
    level: ConfidenceLevel
    reasons: list[str] = field(default_factory=list)
    method: str = "unknown"
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class BallEvidence:
    """Evidence for one detected or manually annotated ball."""

    ball_id: int
    label: str
    crop_uri: str | None = None
    crop_bounds_px: list[int] | None = None
    rough_center_px: list[float] | None = None
    boundary_points_px: list[list[float]] = field(default_factory=list)
    boundary_rejected_points_px: list[list[float]] = field(default_factory=list)
    boundary_filter: dict[str, Any] = field(default_factory=dict)
    boundary_source: str | None = None
    image_model: ImageModel | None = None
    physical_model: PhysicalModel | None = None
    color_confidence: float | None = None
    detection_confidence: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _clean(payload)


@dataclass(slots=True)
class BallEstimate:
    """Final v1 estimate for one ball."""

    ball_id: int
    label: str
    source_px: list[float] | None
    table_xy_mm: list[float] | None
    radius_px: float | None = None
    radius_mm: float | None = None
    table_xy_by_height_mm: dict[str, Any] = field(default_factory=dict)
    evidence: BallEvidence | None = None
    confidence: Confidence | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class TableState:
    """Canonical v1 output for one processed image."""

    schema_version: str
    image_name: str
    image_path: str | None
    source_image_uri: str | None
    source_size_px: dict[str, int] | None
    table_corners_px: list[list[float]]
    camera_model: dict[str, Any]
    balls: list[BallEstimate]
    summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class ManualCorrection:
    correction_type: str
    source_px: list[float] | None = None
    ellipse_px: dict[str, Any] | None = None
    cushion_line_px: list[list[float]] | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class ReviewBallFeedback:
    ball_id: int
    decision: str = "unreviewed"
    issue_tags: list[str] = field(default_factory=list)
    confidence: float | None = None
    comment: str = ""
    manual_correction: ManualCorrection | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class ReviewFeedback:
    schema_version: str
    image_name: str
    numbering_scheme: str | None = None
    balls: list[ReviewBallFeedback] = field(default_factory=list)
    missing_balls: list[dict[str, Any]] = field(default_factory=list)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class GroundTruthEllipse:
    """Human image-space ellipse annotation for one visible ball silhouette."""

    center_px: list[float]
    major_axis_px: float
    minor_axis_px: float
    angle_deg: float
    visible_arcs_deg: list[list[float]] = field(default_factory=list)
    occluded_arcs_deg: list[list[float]] = field(default_factory=list)
    source: str = "manual"
    uncertainty: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class GroundTruthBall:
    label: str
    coordinate_system: CoordinateSystem
    point: list[float]
    image_name: str | None = None
    ball_id: int | None = None
    ellipse_px: GroundTruthEllipse | None = None
    uncertainty: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(slots=True)
class GroundTruthImage:
    """Tracked human annotations for one source image."""

    schema_version: str
    image_name: str
    coordinate_system: CoordinateSystem
    balls: list[GroundTruthBall] = field(default_factory=list)
    image_path: str | None = None
    reviewer: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean(item)
            for key, item in value.items()
            if item is not None and item != {}
        }
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value
