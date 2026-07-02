from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from snookerhelp.qa.accuracy import (
    annotation_points_to_warped,
    build_accuracy_report,
    detector_points_from_state,
)
from snookerhelp.core.config import PROJECT_ROOT, resolve_path
from snookerhelp.recognition.estimator import StateEstimator
from snookerhelp.qa.validation import default_region_margin_mm


CSV_FIELDS = [
    "status",
    "label",
    "annotation_id",
    "detection_id",
    "annotation_x_px",
    "annotation_y_px",
    "detection_x_px",
    "detection_y_px",
    "dx_px",
    "dy_px",
    "error_px",
    "annotation_x_mm",
    "annotation_y_mm",
    "detection_x_mm",
    "detection_y_mm",
    "dx_mm",
    "dy_mm",
    "error_mm",
    "region",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare detected ball centers with manual ground truth"
    )
    parser.add_argument("--image", required=True, help="Source JPEG path")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--detector-output", default=None)
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--maximum-match-distance-mm",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--region-margin-mm",
        type=float,
        default=None,
        help="Distance from cushion used for region grouping. Default: 2 ball diameters.",
    )
    args = parser.parse_args(argv)

    image_path = resolve_path(args.image)
    annotation_path = (
        resolve_path(args.annotations)
        if args.annotations
        else PROJECT_ROOT / "data" / "annotations" / f"{image_path.stem}.json"
    )
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    estimator = StateEstimator.from_config(args.config)
    if args.detector_output:
        detector_output_path = resolve_path(args.detector_output)
        with detector_output_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    else:
        frame = estimator.process(image_path)
        state = frame.state

    detections = detector_points_from_state(state)
    annotations = annotation_points_to_warped(annotation, estimator.table_warp)
    maximum_distance_px = (
        args.maximum_match_distance_mm * estimator.table.px_per_mm
    )
    region_margin_mm = (
        args.region_margin_mm
        if args.region_margin_mm is not None
        else default_region_margin_mm(estimator.table.ball_radius_mm * 2.0)
    )
    report = build_accuracy_report(
        detections=detections,
        annotations=annotations,
        px_per_mm=estimator.table.px_per_mm,
        maximum_distance_px=maximum_distance_px,
        table_length_mm=estimator.table.length_mm,
        table_width_mm=estimator.table.width_mm,
        region_margin_mm=region_margin_mm,
    )
    report["source_image"] = str(image_path)
    report["annotation_file"] = str(annotation_path)
    report["coordinate_system"] = annotation["coordinate_system"]
    report["px_per_mm"] = estimator.table.px_per_mm
    report["region_margin_mm"] = region_margin_mm

    output_directory = (
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "accuracy_reports" / image_path.stem
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{image_path.stem}_accuracy.json"
    csv_path = output_directory / f"{image_path.stem}_accuracy.csv"
    overlay_path = output_directory / f"{image_path.stem}_accuracy_overlay.jpg"
    detector_state_path = output_directory / f"{image_path.stem}_detector_state.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with detector_state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
    _write_csv(csv_path, report)
    overlay = _draw_error_overlay(
        image_path=image_path,
        estimator=estimator,
        report=report,
    )
    cv2.imwrite(
        str(overlay_path),
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )

    summary = report["summary"]
    print(f"Matched: {summary['matched_balls']}")
    print(
        f"Mean error: {_format(summary['mean_error_px'])} px / "
        f"{_format(summary['mean_error_mm'])} mm"
    )
    print(
        f"Median error: {_format(summary['median_error_px'])} px / "
        f"{_format(summary['median_error_mm'])} mm"
    )
    print(
        f"95th percentile: {_format(summary['p95_error_px'])} px / "
        f"{_format(summary['p95_error_mm'])} mm"
    )
    print(
        f"Maximum error: {_format(summary['max_error_px'])} px / "
        f"{_format(summary['max_error_mm'])} mm"
    )
    print(
        f"Missed: {summary['missed_balls']}; "
        f"extra: {summary['extra_balls']}"
    )
    print(f"Reports: {output_directory}")
    return 0


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for match in report["matches"]:
            writer.writerow({"status": "matched", **match})
        for missed in report["missed"]:
            writer.writerow(
                {
                    "status": "missed",
                    "label": missed["label"],
                    "annotation_id": missed["id"],
                    "annotation_x_px": missed["x_px"],
                    "annotation_y_px": missed["y_px"],
                    "annotation_x_mm": missed["x_mm"],
                    "annotation_y_mm": missed["y_mm"],
                }
            )
        for extra in report["extras"]:
            writer.writerow(
                {
                    "status": "extra",
                    "label": extra["label"],
                    "detection_id": extra["id"],
                    "detection_x_px": extra["x_px"],
                    "detection_y_px": extra["y_px"],
                    "detection_x_mm": extra["x_mm"],
                    "detection_y_mm": extra["y_mm"],
                }
            )


def _draw_error_overlay(
    image_path: Path,
    estimator: StateEstimator,
    report: dict[str, Any],
) -> np.ndarray:
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    overlay = estimator.table_warp.warp_image(source)
    for match in report["matches"]:
        annotation = (
            int(round(match["annotation_x_px"])),
            int(round(match["annotation_y_px"])),
        )
        detection = (
            int(round(match["detection_x_px"])),
            int(round(match["detection_y_px"])),
        )
        cv2.arrowedLine(
            overlay,
            annotation,
            detection,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.drawMarker(
            overlay,
            annotation,
            (40, 220, 40),
            cv2.MARKER_CROSS,
            16,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, detection, 5, (20, 20, 240), -1, cv2.LINE_AA)
        text = f"{match['label']} {match['error_mm']:.2f} mm"
        cv2.putText(
            overlay,
            text,
            (detection[0] + 8, detection[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (detection[0] + 8, detection[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    for missed in report["missed"]:
        point = (int(round(missed["x_px"])), int(round(missed["y_px"])))
        cv2.drawMarker(
            overlay,
            point,
            (255, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            24,
            3,
            cv2.LINE_AA,
        )
    for extra in report["extras"]:
        point = (int(round(extra["x_px"])), int(round(extra["y_px"])))
        cv2.circle(overlay, point, 14, (0, 0, 255), 3, cv2.LINE_AA)

    summary = report["summary"]
    header = (
        f"mean={_format(summary['mean_error_mm'])} mm  "
        f"median={_format(summary['median_error_mm'])} mm  "
        f"p95={_format(summary['p95_error_mm'])} mm  "
        f"max={_format(summary['max_error_mm'])} mm  "
        f"missed={summary['missed_balls']} extra={summary['extra_balls']}"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        header,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())




