from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from snookerhelp.qa.accuracy import detector_points_from_state, match_points_by_class
from snookerhelp.core.config import PROJECT_ROOT, resolve_path
from snookerhelp.recognition.estimator import StateEstimator
from snookerhelp.qa.validation import (
    add_region_to_row,
    default_region_margin_mm,
    draw_ball_marker,
    draw_text_with_outline,
    max_pairwise_range_mm,
    summarize_by_region,
)


CSV_FIELDS = [
    "reference_id",
    "label",
    "frame_count",
    "mean_x_mm",
    "mean_y_mm",
    "std_x_mm",
    "std_y_mm",
    "radial_std_mm",
    "range_x_mm",
    "range_y_mm",
    "max_range_mm",
    "region",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure center repeatability across unchanged scenes"
    )
    parser.add_argument("images", nargs="*", help="Repeated source JPEGs")
    parser.add_argument("--folder", default=None)
    parser.add_argument("--pattern", default="*.JPG")
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
    parser.add_argument(
        "--layout-warning-min-match-fraction",
        type=float,
        default=0.8,
        help="Warn below this fraction of matched reference balls.",
    )
    parser.add_argument(
        "--layout-warning-max-median-displacement-mm",
        type=float,
        default=30.0,
        help="Warn when median frame-to-reference displacement exceeds this value.",
    )
    args = parser.parse_args(argv)

    image_paths = [resolve_path(path) for path in args.images]
    if args.folder:
        image_paths.extend(
            sorted(resolve_path(args.folder).glob(args.pattern))
        )
    image_paths = list(dict.fromkeys(path.resolve() for path in image_paths))
    if len(image_paths) < 2:
        raise ValueError("Provide at least two repeated images")

    estimator = StateEstimator.from_config(args.config)
    frame_points: list[list[dict[str, Any]]] = []
    frame_states: list[dict[str, Any]] = []
    for image_path in image_paths:
        state = estimator.process(image_path).state
        frame_states.append(state)
        frame_points.append(detector_points_from_state(state))
        print(f"{image_path.name}: {len(state['balls'])} balls")

    reference = frame_points[0]
    region_margin_mm = (
        args.region_margin_mm
        if args.region_margin_mm is not None
        else default_region_margin_mm(estimator.table.ball_radius_mm * 2.0)
    )
    tracks: dict[int, list[dict[str, Any]]] = {
        int(point["id"]): [
            {
                "frame_index": 0,
                "image": str(image_paths[0]),
                **point,
            }
        ]
        for point in reference
    }
    frame_diagnostics = [
        {
            "frame_index": 0,
            "image": str(image_paths[0]),
            "matched": len(reference),
            "missed": 0,
            "extras": 0,
            "matched_fraction": 1.0,
            "median_displacement_mm": 0.0,
            "max_displacement_mm": 0.0,
            "warnings": [],
        }
    ]
    maximum_distance_px = (
        args.maximum_match_distance_mm * estimator.table.px_per_mm
    )
    for frame_index, current in enumerate(frame_points[1:], start=1):
        matches, missed, extras = match_points_by_class(
            detections=current,
            annotations=reference,
            maximum_distance_px=maximum_distance_px,
        )
        displacements_mm = [
            float(
                np.hypot(
                    match["detection"]["x_mm"] - match["annotation"]["x_mm"],
                    match["detection"]["y_mm"] - match["annotation"]["y_mm"],
                )
            )
            for match in matches
        ]
        matched_fraction = len(matches) / len(reference) if reference else 0.0
        median_displacement_mm = (
            float(np.median(displacements_mm)) if displacements_mm else None
        )
        max_displacement_mm = (
            float(np.max(displacements_mm)) if displacements_mm else None
        )
        warnings: list[str] = []
        if matched_fraction < args.layout_warning_min_match_fraction:
            warnings.append(
                "low_match_fraction: repeated frames may not be the same layout"
            )
        if (
            median_displacement_mm is not None
            and median_displacement_mm
            > args.layout_warning_max_median_displacement_mm
        ):
            warnings.append(
                "large_median_displacement: repeated frames may not be the same layout"
            )
        for match in matches:
            reference_id = int(match["annotation"]["id"])
            tracks[reference_id].append(
                {
                    "frame_index": frame_index,
                    "image": str(image_paths[frame_index]),
                    **match["detection"],
                }
            )
        frame_diagnostics.append(
            {
                "frame_index": frame_index,
                "image": str(image_paths[frame_index]),
                "matched": len(matches),
                "missed": len(missed),
                "extras": len(extras),
                "matched_fraction": matched_fraction,
                "median_displacement_mm": median_displacement_mm,
                "max_displacement_mm": max_displacement_mm,
                "warnings": warnings,
            }
        )

    rows: list[dict[str, Any]] = []
    for reference_point in reference:
        reference_id = int(reference_point["id"])
        samples = tracks[reference_id]
        x_values = np.array([sample["x_mm"] for sample in samples], dtype=float)
        y_values = np.array([sample["y_mm"] for sample in samples], dtype=float)
        if len(samples) > 1:
            std_x: float | None = float(np.std(x_values, ddof=1))
            std_y: float | None = float(np.std(y_values, ddof=1))
            radial_std: float | None = float(np.hypot(std_x, std_y))
        else:
            std_x = None
            std_y = None
            radial_std = None
        row = {
            "reference_id": reference_id,
            "label": reference_point["label"],
            "frame_count": len(samples),
            "mean_x_mm": float(np.mean(x_values)),
            "mean_y_mm": float(np.mean(y_values)),
            "std_x_mm": std_x,
            "std_y_mm": std_y,
            "radial_std_mm": radial_std,
            "range_x_mm": float(np.ptp(x_values)),
            "range_y_mm": float(np.ptp(y_values)),
            "max_range_mm": max_pairwise_range_mm(samples),
            "samples": samples,
        }
        add_region_to_row(
            row,
            x_mm=reference_point["x_mm"],
            y_mm=reference_point["y_mm"],
            table_length_mm=estimator.table.length_mm,
            table_width_mm=estimator.table.width_mm,
            edge_margin_mm=region_margin_mm,
        )
        rows.append(row)

    complete_rows = [row for row in rows if row["frame_count"] == len(image_paths)]
    radial_values = np.array(
        [
            row["radial_std_mm"]
            for row in complete_rows
            if row["radial_std_mm"] is not None
        ],
        dtype=float,
    )
    summary = {
        "image_count": len(image_paths),
        "reference_ball_count": len(reference),
        "complete_tracks": len(complete_rows),
        "region_margin_mm": region_margin_mm,
        "warnings": [
            warning
            for frame in frame_diagnostics
            for warning in frame.get("warnings", [])
        ],
        "mean_radial_std_mm": (
            float(np.mean(radial_values)) if radial_values.size else None
        ),
        "median_radial_std_mm": (
            float(np.median(radial_values)) if radial_values.size else None
        ),
        "p95_radial_std_mm": (
            float(np.percentile(radial_values, 95))
            if radial_values.size
            else None
        ),
        "max_radial_std_mm": (
            float(np.max(radial_values)) if radial_values.size else None
        ),
        "by_region": summarize_by_region(rows, "radial_std_mm"),
    }
    report = {
        "summary": summary,
        "frames": frame_diagnostics,
        "balls": rows,
    }

    output_directory = (
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "data"
        / "repeatability_reports"
        / image_paths[0].stem
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "repeatability.json"
    csv_path = output_directory / "repeatability.csv"
    overlay_path = output_directory / "repeatability_overlay.jpg"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})
    overlay = _draw_repeatability_overlay(
        estimator=estimator,
        image_path=image_paths[0],
        rows=rows,
        summary=summary,
    )
    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

    print(
        "Complete tracks: "
        f"{summary['complete_tracks']}/{summary['reference_ball_count']}"
    )
    print(
        "Radial standard deviation, mean/median/p95/max: "
        f"{_format(summary['mean_radial_std_mm'])} / "
        f"{_format(summary['median_radial_std_mm'])} / "
        f"{_format(summary['p95_radial_std_mm'])} / "
        f"{_format(summary['max_radial_std_mm'])} mm"
    )
    if summary["warnings"]:
        print("Warnings:")
        for warning in sorted(set(summary["warnings"])):
            print(f"  - {warning}")
    print(f"Reports: {output_directory}")
    return 0


def _draw_repeatability_overlay(
    estimator: StateEstimator,
    image_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> np.ndarray:
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    overlay = estimator.table_warp.warp_image(source)
    for row in rows:
        samples = row["samples"]
        if not samples:
            continue
        points_px = np.array(
            [
                [int(round(sample["x_px"])), int(round(sample["y_px"]))]
                for sample in samples
            ],
            dtype=np.int32,
        )
        if len(points_px) > 1:
            cv2.polylines(
                overlay,
                [points_px.reshape(-1, 1, 2)],
                False,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        draw_ball_marker(overlay, samples[0])
        text = (
            f"{row['reference_id']} {row['label']} "
            f"std={_format(row['radial_std_mm'])} "
            f"range={row['max_range_mm']:.2f}"
        )
        draw_text_with_outline(
            overlay,
            text,
            (int(points_px[0][0]) + 8, int(points_px[0][1]) - 8),
            scale=0.45,
        )
    header = (
        f"repeatability: complete={summary['complete_tracks']}/"
        f"{summary['reference_ball_count']}  "
        f"median radial std={_format(summary['median_radial_std_mm'])} mm"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (20, 20, 20), -1)
    draw_text_with_outline(overlay, header, (12, 28), scale=0.65)
    return overlay


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())




