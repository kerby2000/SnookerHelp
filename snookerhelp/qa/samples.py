from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from snookerhelp.core.config import resolve_path
from snookerhelp.recognition.estimator import StateEstimator


def evaluate_samples_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate supplied JPEGs")
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--media", default="Media")
    parser.add_argument("--output-dir", default="data/debug_outputs/evaluation")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save overlays for every sample instead of summary only",
    )
    args = parser.parse_args(argv)

    rows = evaluate_samples(
        config_path=args.config,
        media_root=args.media,
        output_root=args.output_dir,
        save_images=args.save_images,
    )
    exact = sum(row["count_error"] == 0 for row in rows)
    mean_absolute_error = (
        sum(abs(row["count_error"]) for row in rows) / len(rows) if rows else 0.0
    )
    summary_path = resolve_path(args.output_dir) / "sample_evaluation.json"
    print(
        f"Exact counts: {exact}/{len(rows)}; "
        f"mean absolute count error: {mean_absolute_error:.3f}; "
        f"summary: {summary_path}"
    )
    return 0


def evaluate_samples(
    *,
    config_path: str | Path = "configs/sony_dev.yaml",
    media_root: str | Path = "Media",
    output_root: str | Path = "data/debug_outputs/evaluation",
    save_images: bool = False,
) -> list[dict[str, object]]:
    estimator = StateEstimator.from_config(config_path)
    media_path = resolve_path(media_root)
    output_path = resolve_path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for image_path in sorted(media_path.rglob("*.JPG")):
        if save_images:
            frame, _ = estimator.process_and_save(image_path, output_path)
        else:
            frame = estimator.process(image_path)
        state = frame.state
        expected = 0 if image_path.parent.name == "01_empty_table" else 22
        classes = Counter(ball["class"] for ball in state["balls"])
        row = {
            "scenario": image_path.parent.name,
            "image": image_path.name,
            "expected_count": expected,
            "detected_count": state["detection"]["ball_count"],
            "count_error": state["detection"]["ball_count"] - expected,
            "classes": dict(sorted(classes.items())),
        }
        rows.append(row)
        print(
            f"{row['scenario']}/{row['image']}: "
            f"{row['detected_count']}/{expected} {row['classes']}"
        )

    summary_path = output_path / "sample_evaluation.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    return rows


__all__ = ["evaluate_samples", "evaluate_samples_command"]

