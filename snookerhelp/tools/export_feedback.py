from __future__ import annotations

import argparse

from snookerhelp.review.feedback import (
    load_feedback_from_reports_root,
    save_feedback_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export v1 review feedback JSONL")
    parser.add_argument("--reports", default="outputs/reports")
    parser.add_argument("--output", default="data/review_feedback/dataset_feedback_v1.jsonl")
    args = parser.parse_args(argv)

    feedback_items = load_feedback_from_reports_root(args.reports)
    save_feedback_jsonl(feedback_items, args.output)
    total_balls = sum(len(item.balls) for item in feedback_items)
    total_missing = sum(len(item.missing_balls) for item in feedback_items)
    print(
        f"Exported {len(feedback_items)} image feedback records, "
        f"{total_balls} ball reviews, {total_missing} missing-ball hints: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
