from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from snookerhelp.core.ground_truth import load_ground_truth
from snookerhelp.core.schema import GroundTruthImage
from snookerhelp.qa.ellipse_accuracy import compare_ellipses
from snookerhelp.recognition import table_state_from_legacy_report


ELLIPSE_BENCHMARK_SCHEMA = "snookerhelp.ellipse_benchmark.v1"


def evaluate_ellipse_benchmark(
    table_state: dict[str, Any],
    ground_truth: GroundTruthImage | dict[str, Any],
    *,
    default_tolerance_px: float = 3.0,
) -> dict[str, Any]:
    """Measure production and per-map ellipses against human annotations."""

    truth = (
        ground_truth.to_dict()
        if isinstance(ground_truth, GroundTruthImage)
        else dict(ground_truth)
    )
    truth_by_id = {
        int(item["ball_id"]): item
        for item in truth.get("balls", [])
        if item.get("ball_id") is not None and item.get("ellipse_px")
    }
    detected_by_id = {
        int(item["ball_id"]): item
        for item in table_state.get("balls", [])
        if item.get("ball_id") is not None
    }
    rows: list[dict[str, Any]] = []
    map_rows: dict[str, list[dict[str, Any]]] = {}
    matches, matched_detector_ids = _match_balls_by_label_and_position(
        list(truth_by_id.values()),
        list(detected_by_id.values()),
    )

    for ball_id, annotation in sorted(truth_by_id.items()):
        ball = matches.get(ball_id)
        if ball is None:
            rows.append(
                {
                    "ball_id": ball_id,
                    "label": annotation.get("label"),
                    "status": "missed",
                }
            )
            continue
        expected = annotation["ellipse_px"]
        evidence = ball.get("evidence") or {}
        diagnostics = evidence.get("diagnostics") or {}
        policy = diagnostics.get("final_image_evidence") or {}
        production_model = evidence.get("image_model")
        comparison = compare_ellipses(production_model, expected)
        center_error = _point_error(ball.get("source_px"), expected.get("center_px"))
        tolerance = _annotation_tolerance(annotation, default_tolerance_px)
        row = {
            "ball_id": ball_id,
            "matched_detector_ball_id": int(ball["ball_id"]),
            "label": ball.get("label") or annotation.get("label"),
            "status": "computed" if comparison.get("status") == "computed" else "unavailable",
            "selected_map": policy.get("selected_map"),
            "source_center_error_px": _round(center_error),
            "annotation_tolerance_px": _round(tolerance),
            "within_center_tolerance": (
                center_error is not None and center_error <= tolerance
            ),
            **comparison,
        }
        rows.append(row)

        variants = ((diagnostics.get("evidence_maps") or {}).get("boundary_variants") or {})
        for map_key, variant in variants.items():
            map_comparison = compare_ellipses(
                (variant or {}).get("ellipse_fit"),
                expected,
            )
            if map_comparison.get("status") != "computed":
                continue
            map_rows.setdefault(str(map_key), []).append(
                {
                    "ball_id": ball_id,
                    "label": row["label"],
                    **map_comparison,
                }
            )

    computed = [row for row in rows if row.get("status") == "computed"]
    extras = sorted(set(detected_by_id) - matched_detector_ids)
    misses = sorted(set(truth_by_id) - set(matches))
    worst = sorted(
        computed,
        key=lambda row: float(row.get("contour_rms_error_px") or -1.0),
        reverse=True,
    )
    return {
        "schema_version": ELLIPSE_BENCHMARK_SCHEMA,
        "image_name": truth.get("image_name") or table_state.get("image_name"),
        "annotation_schema_version": truth.get("schema_version"),
        "default_tolerance_px": float(default_tolerance_px),
        "summary": {
            "annotated_ball_count": len(truth_by_id),
            "computed_ball_count": len(computed),
            "missed_ball_ids": misses,
            "unannotated_detected_ball_ids": extras,
            "within_center_tolerance_count": sum(
                bool(row.get("within_center_tolerance")) for row in computed
            ),
            **_aggregate(computed),
        },
        "worst_balls": [
            {
                "ball_id": row["ball_id"],
                "label": row["label"],
                "selected_map": row.get("selected_map"),
                "source_center_error_px": row.get("source_center_error_px"),
                "contour_rms_error_px": row.get("contour_rms_error_px"),
                "annotation_score": row.get("annotation_score"),
            }
            for row in worst[:10]
        ],
        "by_evidence_map": {
            map_key: {
                "summary": _aggregate(values),
                "balls": values,
            }
            for map_key, values in sorted(map_rows.items())
        },
        "balls": rows,
        "metric_note": (
            "Detections are matched one-to-one to annotations by class and "
            "nearest source position. This prevents interchangeable red-ball "
            "display IDs from creating false geometric errors. Annotation "
            "score and ellipse errors are ground-truth based; the production "
            "evidence-view score is not used here."
        ),
    }


def evaluate_report_file(
    report_path: str | Path,
    annotation_path: str | Path,
    *,
    default_tolerance_px: float = 3.0,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    table_state = table_state_from_legacy_report(
        report,
        report_stem=Path(report_path).parent.name,
    ).to_dict()
    annotation = load_ground_truth(annotation_path)
    return evaluate_ellipse_benchmark(
        table_state,
        annotation,
        default_tolerance_px=default_tolerance_px,
    )


def write_ellipse_benchmark(
    result: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "ellipse_benchmark.json"
    csv_path = output / "ellipse_benchmark.csv"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "ball_id",
        "matched_detector_ball_id",
        "label",
        "status",
        "selected_map",
        "source_center_error_px",
        "contour_rms_error_px",
        "major_axis_error_px",
        "minor_axis_error_px",
        "angle_error_deg",
        "annotation_score",
        "annotation_tolerance_px",
        "within_center_tolerance",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.get("balls", []))
    return json_path, csv_path


def _match_balls_by_label_and_position(
    annotations: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    """Match same-class balls globally instead of trusting display IDs.

    Canonical IDs are useful UI slots, but red balls are physically
    interchangeable and two nearly level reds can swap slot order after a
    subpixel coordinate improvement.  Accuracy must measure geometry, not that
    incidental ordering.  Hungarian assignment also prevents one detection
    from satisfying more than one annotation.
    """

    matches: dict[int, dict[str, Any]] = {}
    matched_detector_ids: set[int] = set()
    labels = {
        _label(item) for item in annotations
    } | {
        _label(item) for item in detections
    }
    for label in sorted(labels):
        expected = sorted(
            [item for item in annotations if _label(item) == label],
            key=lambda item: int(item["ball_id"]),
        )
        observed = sorted(
            [item for item in detections if _label(item) == label],
            key=lambda item: int(item["ball_id"]),
        )
        if not expected or not observed:
            continue
        costs = np.full((len(expected), len(observed)), 1e9, dtype=np.float64)
        for expected_index, annotation in enumerate(expected):
            expected_center = (annotation.get("ellipse_px") or {}).get("center_px")
            for observed_index, detection in enumerate(observed):
                error = _point_error(detection.get("source_px"), expected_center)
                if error is not None:
                    costs[expected_index, observed_index] = error
        expected_rows, observed_columns = linear_sum_assignment(costs)
        for expected_index, observed_index in zip(expected_rows, observed_columns):
            if costs[expected_index, observed_index] >= 1e8:
                continue
            annotation_id = int(expected[expected_index]["ball_id"])
            detection = observed[observed_index]
            detector_id = int(detection["ball_id"])
            matches[annotation_id] = detection
            matched_detector_ids.add(detector_id)
    return matches, matched_detector_ids


def _label(item: dict[str, Any]) -> str:
    return str(item.get("label") or "unknown").strip().lower()


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    return {
        "count": len(values),
        **_metric_summary(values, "source_center_error_px"),
        **_metric_summary(values, "contour_rms_error_px"),
        **_metric_summary(values, "annotation_score"),
    }


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    prefix = key.removesuffix("_px")
    if not values:
        return {
            f"mean_{prefix}": None,
            f"median_{prefix}": None,
            f"p95_{prefix}": None,
            f"max_{prefix}": None,
        }
    return {
        f"mean_{prefix}": _round(mean(values)),
        f"median_{prefix}": _round(median(values)),
        f"p95_{prefix}": _round(float(np.percentile(values, 95))),
        f"max_{prefix}": _round(max(values)),
    }


def _annotation_tolerance(annotation: dict[str, Any], default: float) -> float:
    ellipse_uncertainty = (annotation.get("ellipse_px") or {}).get("uncertainty") or {}
    ball_uncertainty = annotation.get("uncertainty") or {}
    return float(
        ellipse_uncertainty.get("center_tolerance_px")
        or ball_uncertainty.get("center_tolerance_px")
        or default
    )


def _point_error(left: Any, right: Any) -> float | None:
    if not left or not right:
        return None
    return float(np.linalg.norm(np.asarray(left[:2], dtype=float) - np.asarray(right[:2], dtype=float)))


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


__all__ = [
    "ELLIPSE_BENCHMARK_SCHEMA",
    "evaluate_ellipse_benchmark",
    "evaluate_report_file",
    "write_ellipse_benchmark",
]
