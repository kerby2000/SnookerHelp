from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from snookerhelp.core.schema import TableState
from snookerhelp.recognition import table_state_from_legacy_report
from snookerhelp.review.schema import ReviewFeedback, default_review_feedback


def load_legacy_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    return json.loads(report_path.read_text(encoding="utf-8"))


def table_state_from_report(path: str | Path) -> TableState:
    report_path = Path(path)
    return table_state_from_legacy_report(
        load_legacy_report(report_path),
        report_stem=report_path.parent.name,
    )


def default_feedback_for_table_state(table_state: TableState) -> ReviewFeedback:
    return default_review_feedback(
        image_name=table_state.image_name,
        ball_ids=[ball.ball_id for ball in table_state.balls],
    )


def review_payload_from_report(path: str | Path) -> dict[str, Any]:
    table_state = table_state_from_report(path)
    return {
        "table_state": table_state.to_dict(),
        "review_feedback": default_feedback_for_table_state(table_state).to_dict(),
    }


__all__ = [
    "default_feedback_for_table_state",
    "load_legacy_report",
    "review_payload_from_report",
    "table_state_from_report",
]
