from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from snookerhelp.qa.validation import add_region_to_row, summarize_by_region
from snookerhelp.calibration.homography_bootstrap import TableWarp


def annotation_points_to_warped(
    annotation: dict[str, Any],
    table_warp: TableWarp,
) -> list[dict[str, Any]]:
    coordinate_system = str(annotation["coordinate_system"])
    entries = annotation.get("balls", annotation.get("annotations", []))
    converted: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        x_value = float(entry["x"])
        y_value = float(entry["y"])
        if coordinate_system == "source_px":
            point = table_warp.source_to_warped(
                np.float32([[x_value, y_value]])
            )[0]
            x_px, y_px = float(point[0]), float(point[1])
        elif coordinate_system == "warped_px":
            x_px, y_px = x_value, y_value
        elif coordinate_system == "table_mm":
            x_px, y_px = table_warp.table_mm_to_warped_px(x_value, y_value)
        else:
            raise ValueError(
                "coordinate_system must be source_px, warped_px, or table_mm"
            )
        x_mm, y_mm = table_warp.warped_px_to_table_mm(x_px, y_px)
        converted.append(
            {
                "id": entry.get("id", index),
                "label": str(entry.get("label", entry.get("class", "unknown"))),
                "x_px": x_px,
                "y_px": y_px,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "notes": entry.get("notes"),
            }
        )
    return converted


def detector_points_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, ball in enumerate(state["balls"], start=1):
        center = ball.get("refined_center_px")
        if center is None:
            center = ball.get("debug", {}).get("warped_center_px")
        if center is None:
            raise ValueError(f"Detector ball {index} has no warped center")
        table_xy = ball.get("table_xy_mm", [ball["x_mm"], ball["y_mm"]])
        points.append(
            {
                "id": ball.get("id", index),
                "label": str(ball.get("color_label", ball.get("class", "unknown"))),
                "x_px": float(center[0]),
                "y_px": float(center[1]),
                "x_mm": float(table_xy[0]),
                "y_mm": float(table_xy[1]),
                "confidence": float(
                    ball.get("detection_confidence", ball.get("confidence", 0.0))
                ),
            }
        )
    return points


def match_points_by_class(
    detections: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    maximum_distance_px: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    detection_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        detection_groups[detection["label"]].append(detection)
    for annotation in annotations:
        annotation_groups[annotation["label"]].append(annotation)

    matches: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    labels = sorted(set(detection_groups) | set(annotation_groups))
    for label in labels:
        class_detections = detection_groups[label]
        class_annotations = annotation_groups[label]
        possible_pairs = sorted(
            (
                (
                    float(
                        np.hypot(
                            detection["x_px"] - annotation["x_px"],
                            detection["y_px"] - annotation["y_px"],
                        )
                    ),
                    detection_index,
                    annotation_index,
                )
                for detection_index, detection in enumerate(class_detections)
                for annotation_index, annotation in enumerate(class_annotations)
            ),
            key=lambda item: item[0],
        )
        used_detections: set[int] = set()
        used_annotations: set[int] = set()
        for distance_px, detection_index, annotation_index in possible_pairs:
            if (
                detection_index in used_detections
                or annotation_index in used_annotations
                or (
                    maximum_distance_px is not None
                    and distance_px > maximum_distance_px
                )
            ):
                continue
            detection = class_detections[detection_index]
            annotation = class_annotations[annotation_index]
            used_detections.add(detection_index)
            used_annotations.add(annotation_index)
            matches.append(
                {
                    "label": label,
                    "detection": detection,
                    "annotation": annotation,
                    "error_px": distance_px,
                }
            )
        missed.extend(
            annotation
            for index, annotation in enumerate(class_annotations)
            if index not in used_annotations
        )
        extras.extend(
            detection
            for index, detection in enumerate(class_detections)
            if index not in used_detections
        )
    return matches, missed, extras


def build_accuracy_report(
    detections: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    px_per_mm: float,
    maximum_distance_px: float | None = None,
    table_length_mm: float | None = None,
    table_width_mm: float | None = None,
    region_margin_mm: float | None = None,
) -> dict[str, Any]:
    matches, missed, extras = match_points_by_class(
        detections, annotations, maximum_distance_px
    )
    rows: list[dict[str, Any]] = []
    for match in matches:
        detection = match["detection"]
        annotation = match["annotation"]
        dx_px = detection["x_px"] - annotation["x_px"]
        dy_px = detection["y_px"] - annotation["y_px"]
        dx_mm = dx_px / px_per_mm
        dy_mm = -dy_px / px_per_mm
        error_mm = float(np.hypot(dx_mm, dy_mm))
        row = {
            "label": match["label"],
            "annotation_id": annotation["id"],
            "detection_id": detection["id"],
            "annotation_x_px": annotation["x_px"],
            "annotation_y_px": annotation["y_px"],
            "detection_x_px": detection["x_px"],
            "detection_y_px": detection["y_px"],
            "dx_px": dx_px,
            "dy_px": dy_px,
            "error_px": match["error_px"],
            "annotation_x_mm": annotation["x_mm"],
            "annotation_y_mm": annotation["y_mm"],
            "detection_x_mm": detection["x_mm"],
            "detection_y_mm": detection["y_mm"],
            "dx_mm": dx_mm,
            "dy_mm": dy_mm,
            "error_mm": error_mm,
        }
        if (
            table_length_mm is not None
            and table_width_mm is not None
            and region_margin_mm is not None
        ):
            add_region_to_row(
                row,
                x_mm=annotation["x_mm"],
                y_mm=annotation["y_mm"],
                table_length_mm=table_length_mm,
                table_width_mm=table_width_mm,
                edge_margin_mm=region_margin_mm,
            )
        rows.append(row)

    errors_px = np.array([row["error_px"] for row in rows], dtype=float)
    errors_mm = np.array([row["error_mm"] for row in rows], dtype=float)
    summary = {
        "matched_balls": len(rows),
        "missed_balls": len(missed),
        "extra_balls": len(extras),
        "mean_error_px": _stat(errors_px, np.mean),
        "median_error_px": _stat(errors_px, np.median),
        "p95_error_px": _percentile(errors_px, 95),
        "max_error_px": _stat(errors_px, np.max),
        "mean_error_mm": _stat(errors_mm, np.mean),
        "median_error_mm": _stat(errors_mm, np.median),
        "p95_error_mm": _percentile(errors_mm, 95),
        "max_error_mm": _stat(errors_mm, np.max),
    }
    if (
        table_length_mm is not None
        and table_width_mm is not None
        and region_margin_mm is not None
    ):
        summary["by_region"] = summarize_by_region(rows, "error_mm")
    return {
        "summary": summary,
        "matches": rows,
        "missed": missed,
        "extras": extras,
    }


def _stat(values: np.ndarray, function: Any) -> float | None:
    return float(function(values)) if values.size else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None

