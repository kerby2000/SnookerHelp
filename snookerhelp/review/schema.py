from __future__ import annotations

from typing import Any

from snookerhelp.core.schema import (
    ManualCorrection,
    ReviewBallFeedback,
    ReviewFeedback,
)


V1_REVIEW_SCHEMA = "snookerhelp.review_feedback.v1"


def default_review_feedback(
    *,
    image_name: str,
    ball_ids: list[int],
) -> ReviewFeedback:
    return ReviewFeedback(
        schema_version=V1_REVIEW_SCHEMA,
        image_name=image_name,
        balls=[ReviewBallFeedback(ball_id=ball_id) for ball_id in ball_ids],
    )


def review_feedback_from_dict(payload: dict[str, Any], *, image_name: str) -> ReviewFeedback:
    if payload.get("schema_version") == V1_REVIEW_SCHEMA:
        return _v1_review_from_dict(payload, image_name=image_name)
    return review_feedback_from_legacy(payload, image_name=image_name)


def review_feedback_from_legacy(
    payload: dict[str, Any],
    *,
    image_name: str,
) -> ReviewFeedback:
    balls: list[ReviewBallFeedback] = []
    for item in payload.get("balls", []):
        correction_payload = item.get("manual_correction") or {}
        correction = _manual_correction_from_legacy(correction_payload)
        balls.append(
            ReviewBallFeedback(
                ball_id=int(item["id"]),
                decision=str(item.get("decision") or "unreviewed"),
                issue_tags=list(item.get("issue_tags") or []),
                confidence=(
                    float(item["confidence"])
                    if item.get("confidence_source") == "human"
                    and item.get("confidence") is not None
                    else None
                ),
                comment=str(item.get("comment") or ""),
                manual_correction=correction,
            )
        )
    return ReviewFeedback(
        schema_version=V1_REVIEW_SCHEMA,
        image_name=image_name,
        balls=balls,
        missing_balls=list(payload.get("missing_ball_hints") or []),
        audit_trail=list(payload.get("audit_trail") or []),
    )


def _v1_review_from_dict(payload: dict[str, Any], *, image_name: str) -> ReviewFeedback:
    balls: list[ReviewBallFeedback] = []
    for item in payload.get("balls", []):
        correction_payload = item.get("manual_correction")
        balls.append(
            ReviewBallFeedback(
                ball_id=int(item["ball_id"]),
                decision=str(item.get("decision") or "unreviewed"),
                issue_tags=list(item.get("issue_tags") or []),
                confidence=(
                    float(item["confidence"])
                    if item.get("confidence") is not None
                    else None
                ),
                comment=str(item.get("comment") or ""),
                manual_correction=(
                    ManualCorrection(**correction_payload)
                    if correction_payload
                    else None
                ),
            )
        )
    return ReviewFeedback(
        schema_version=V1_REVIEW_SCHEMA,
        image_name=str(payload.get("image_name") or image_name),
        balls=balls,
        missing_balls=list(payload.get("missing_balls") or []),
        audit_trail=list(payload.get("audit_trail") or []),
    )


def _manual_correction_from_legacy(payload: dict[str, Any]) -> ManualCorrection | None:
    if not payload:
        return None
    model = str(payload.get("model") or "manual")
    ellipse = payload.get("ellipse")
    circle = payload.get("circle") or {}
    return ManualCorrection(
        correction_type=model,
        source_px=payload.get("center_px") or circle.get("center_px"),
        ellipse_px=ellipse,
        cushion_line_px=payload.get("cushion_line_px"),
        note=payload.get("note"),
    )
