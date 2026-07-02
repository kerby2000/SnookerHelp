from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from snookerhelp.core.schema import ManualCorrection, ReviewBallFeedback, ReviewFeedback
from snookerhelp.review.schema import review_feedback_from_dict


def load_review_feedback(path: str | Path, *, image_name: str | None = None) -> ReviewFeedback:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return review_feedback_from_dict(payload, image_name=image_name)


def save_review_feedback(feedback: ReviewFeedback, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(feedback.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl_review_feedback(feedback: ReviewFeedback, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(feedback.to_dict()) + "\n")


def load_legacy_feedback_jsonl(path: str | Path) -> list[ReviewFeedback]:
    """Read the existing exported JSONL review dataset as v1 feedback objects."""

    groups: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_name = Path(str(row.get("image") or "unknown")).stem
        group = groups.setdefault(
            image_name,
            {"schema_version": "snookerhelp.review_feedback.v1", "image_name": image_name, "balls": [], "missing_balls": []},
        )
        if row.get("row_type") == "missing_ball_hint":
            group["missing_balls"].append(
                {
                    "source_px": row.get("source_px"),
                    "label_guess": row.get("label_guess"),
                    "comment": row.get("comment", ""),
                }
            )
            continue
        if row.get("row_type") != "ball_review":
            continue
        group["balls"].append(_legacy_row_to_ball_feedback(row).to_dict())

    return [review_feedback_from_dict(group, image_name=group["image_name"]) for group in groups.values()]


def save_feedback_jsonl(feedback_items: list[ReviewFeedback], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for feedback in feedback_items:
            handle.write(json.dumps(feedback.to_dict(), ensure_ascii=False) + "\n")


def load_feedback_from_report_dir(report_dir: str | Path) -> ReviewFeedback | None:
    directory = Path(report_dir)
    stem = directory.name
    review_v1_path = directory / "review_v1.json"
    review_legacy_path = directory / "review.json"
    if review_v1_path.is_file():
        return load_review_feedback(review_v1_path, image_name=stem)
    if review_legacy_path.is_file():
        payload = json.loads(review_legacy_path.read_text(encoding="utf-8"))
        return review_feedback_from_dict(payload, image_name=stem)
    return None


def load_feedback_from_reports_root(reports_root: str | Path) -> list[ReviewFeedback]:
    feedback_items: list[ReviewFeedback] = []
    for report_path in sorted(Path(reports_root).glob("*/report.json")):
        feedback = load_feedback_from_report_dir(report_path.parent)
        if feedback is not None:
            feedback_items.append(feedback)
    return feedback_items


def _legacy_row_to_ball_feedback(row: dict[str, Any]) -> ReviewBallFeedback:
    manual_correction = None
    if row.get("manual_center_px") or row.get("manual_ellipse") or row.get("manual_cushion_line_px"):
        manual_correction = ManualCorrection(
            correction_type=str(row.get("manual_model") or "manual"),
            source_px=row.get("manual_center_px"),
            ellipse_px=row.get("manual_ellipse"),
            cushion_line_px=row.get("manual_cushion_line_px"),
            note=row.get("comment") or None,
        )
    return ReviewBallFeedback(
        ball_id=int(row["ball_id"]),
        decision=str(row.get("human_decision") or "unreviewed"),
        issue_tags=list(row.get("issue_tags") or []),
        confidence=row.get("human_confidence"),
        comment=str(row.get("comment") or ""),
        manual_correction=manual_correction,
    )


__all__ = [
    "append_jsonl_review_feedback",
    "load_feedback_from_report_dir",
    "load_feedback_from_reports_root",
    "load_legacy_feedback_jsonl",
    "load_review_feedback",
    "save_review_feedback",
    "save_feedback_jsonl",
]
