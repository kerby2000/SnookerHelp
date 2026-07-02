from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from snookerhelp.core.config import PROJECT_ROOT, resolve_path
from snookerhelp.qa.validation import (
    add_region_to_row,
    ball_by_id,
    ball_points_from_state,
    default_region_margin_mm,
    draw_ball_marker,
    draw_text_with_outline,
    load_json,
    load_yaml_or_json,
    state_display_name,
    state_matches_selector,
    summarize_by_region,
    summarize_values,
    table_dimensions_from_state,
    table_mm_to_warped_px_from_state,
    warped_overlay_base,
    write_csv,
    write_json,
)


CSV_FIELDS = [
    "state_file",
    "source_image",
    "spot",
    "ball_id",
    "ball_label",
    "expected_x_mm",
    "expected_y_mm",
    "detected_x_mm",
    "detected_y_mm",
    "dx_mm",
    "dy_mm",
    "error_mm",
    "region",
    "notes",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate detected ball centers against known table spot coordinates"
    )
    parser.add_argument("detector_outputs", nargs="+", help="Processed detector state JSON files")
    parser.add_argument("--spots", required=True, help="YAML/JSON known spot coordinates")
    parser.add_argument("--mappings", default=None, help="YAML/JSON spot-to-detection mapping")
    parser.add_argument(
        "--mapping",
        action="append",
        default=[],
        help="Inline mapping as spot:ball_id, e.g. blue:6. Applies to all states.",
    )
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-diameter-mm", type=float, default=52.5)
    parser.add_argument(
        "--region-margin-mm",
        type=float,
        default=None,
        help="Distance from cushion used for region grouping. Default: 2 ball diameters.",
    )
    args = parser.parse_args(argv)

    spots, embedded_mappings = _load_spots(args.spots)
    mappings = embedded_mappings
    if args.mappings:
        mappings.extend(_load_mappings(args.mappings))
    mappings.extend(_parse_inline_mapping(item) for item in args.mapping)
    if not mappings:
        raise ValueError(
            "Provide --mappings/--mapping, or include ball_id/class mappings in --spots"
        )

    state_paths = [resolve_path(path) for path in args.detector_outputs]
    states = [(path, load_json(path)) for path in state_paths]
    region_margin_mm = (
        args.region_margin_mm
        if args.region_margin_mm is not None
        else default_region_margin_mm(args.expected_diameter_mm)
    )

    rows: list[dict[str, Any]] = []
    for state_path, state in states:
        for mapping in mappings:
            selector = mapping.get("state") or mapping.get("file") or mapping.get("image")
            if not state_matches_selector(state, state_path, selector):
                continue
            spot_name = str(mapping["spot"])
            if spot_name not in spots:
                raise KeyError(f"Spot {spot_name!r} is not defined in {args.spots}")
            point = _mapped_detection(state, spots[spot_name], mapping)
            rows.append(
                _spot_row(
                    state_path=state_path,
                    state=state,
                    spot_name=spot_name,
                    expected=spots[spot_name],
                    point=point,
                    region_margin_mm=region_margin_mm,
                    notes=mapping.get("notes"),
                )
            )

    summary = {
        "spot_count": len(rows),
        "region_margin_mm": region_margin_mm,
        "error_mm": summarize_values(row["error_mm"] for row in rows),
        "abs_dx_mm": summarize_values(abs(row["dx_mm"]) for row in rows),
        "abs_dy_mm": summarize_values(abs(row["dy_mm"]) for row in rows),
        "by_region": summarize_by_region(rows, "error_mm"),
    }
    report = {"summary": summary, "spots": rows}

    output_directory = (
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "physical_validation" / "spot_positions"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "spot_positions.json", report)
    write_csv(output_directory / "spot_positions.csv", rows, CSV_FIELDS)

    for state_path, state in states:
        state_rows = [
            row for row in rows if Path(row["state_file"]).resolve() == state_path.resolve()
        ]
        if not state_rows:
            continue
        overlay = _draw_spot_overlay(state, state_rows, args.config)
        overlay_path = output_directory / f"{state_display_name(state, state_path)}_spot_positions_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

    print(f"Spots evaluated: {summary['spot_count']}")
    print(
        "Spot error mean/median/max: "
        f"{_fmt(summary['error_mm']['mean'])} / "
        f"{_fmt(summary['error_mm']['median'])} / "
        f"{_fmt(summary['error_mm']['max'])} mm"
    )
    print(f"Reports: {output_directory}")
    return 0


def _load_spots(path: str | Path) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    payload = load_yaml_or_json(path)
    raw_spots = payload.get("spots", payload) if isinstance(payload, dict) else {}
    spots: dict[str, dict[str, float]] = {}
    mappings: list[dict[str, Any]] = []
    for name, value in raw_spots.items():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            spots[str(name)] = {"x_mm": float(value[0]), "y_mm": float(value[1])}
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Unsupported spot specification for {name}: {value!r}")
        spots[str(name)] = {
            "x_mm": float(value.get("x_mm", value.get("x"))),
            "y_mm": float(value.get("y_mm", value.get("y"))),
        }
        if any(key in value for key in ("ball_id", "id", "ball", "class", "label")):
            mappings.append(
                {
                    "spot": str(name),
                    "ball_id": value.get("ball_id", value.get("id", value.get("ball"))),
                    "label": value.get("label", value.get("class")),
                    "state": value.get("state", value.get("file", value.get("image"))),
                    "notes": value.get("notes"),
                }
            )
    if isinstance(payload, dict):
        for item in payload.get("mappings", payload.get("spot_mappings", [])) or []:
            if not isinstance(item, dict):
                raise ValueError(f"Unsupported mapping specification: {item!r}")
            mappings.append(_normalize_mapping(item))
    return spots, mappings


def _load_mappings(path: str | Path) -> list[dict[str, Any]]:
    payload = load_yaml_or_json(path)
    if isinstance(payload, dict):
        raw_items = payload.get("mappings", payload.get("spot_mappings"))
        if raw_items is None:
            raw_items = [
                {"spot": spot, "ball_id": ball_id}
                for spot, ball_id in payload.items()
            ]
    else:
        raw_items = payload
    mappings: list[dict[str, Any]] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported mapping specification: {item!r}")
        mappings.append(_normalize_mapping(item))
    return mappings


def _normalize_mapping(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "spot": str(item["spot"]),
        "ball_id": item.get("ball_id", item.get("id", item.get("ball"))),
        "label": item.get("label", item.get("class")),
        "state": item.get("state", item.get("file", item.get("image"))),
        "notes": item.get("notes"),
    }


def _parse_inline_mapping(spec: str) -> dict[str, Any]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError("--mapping must be formatted as spot:ball_id")
    return {"spot": parts[0], "ball_id": int(parts[1])}


def _mapped_detection(
    state: dict[str, Any],
    expected: dict[str, float],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    if mapping.get("ball_id") is not None:
        return ball_by_id(state, int(mapping["ball_id"]))

    label = mapping.get("label", mapping["spot"])
    candidates = [
        point for point in ball_points_from_state(state) if point["label"] == label
    ]
    if not candidates:
        raise KeyError(f"No detected ball with class/label {label!r}")
    return min(
        candidates,
        key=lambda point: float(
            np.hypot(point["x_mm"] - expected["x_mm"], point["y_mm"] - expected["y_mm"])
        ),
    )


def _spot_row(
    state_path: Path,
    state: dict[str, Any],
    spot_name: str,
    expected: dict[str, float],
    point: dict[str, Any],
    region_margin_mm: float,
    notes: str | None,
) -> dict[str, Any]:
    length_mm, width_mm = table_dimensions_from_state(state)
    dx_mm = point["x_mm"] - expected["x_mm"]
    dy_mm = point["y_mm"] - expected["y_mm"]
    expected_px = table_mm_to_warped_px_from_state(
        state, expected["x_mm"], expected["y_mm"]
    )
    row = {
        "state_file": str(state_path.resolve()),
        "source_image": state.get("source_image"),
        "spot": spot_name,
        "ball_id": point["id"],
        "ball_label": point["label"],
        "expected_x_mm": expected["x_mm"],
        "expected_y_mm": expected["y_mm"],
        "expected_x_px": expected_px[0],
        "expected_y_px": expected_px[1],
        "detected_x_mm": point["x_mm"],
        "detected_y_mm": point["y_mm"],
        "detected_x_px": point["x_px"],
        "detected_y_px": point["y_px"],
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "error_mm": float(np.hypot(dx_mm, dy_mm)),
        "notes": notes,
    }
    add_region_to_row(
        row,
        x_mm=expected["x_mm"],
        y_mm=expected["y_mm"],
        table_length_mm=length_mm,
        table_width_mm=width_mm,
        edge_margin_mm=region_margin_mm,
    )
    return row


def _draw_spot_overlay(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    config_path: str | Path,
) -> Any:
    overlay = warped_overlay_base(state, config_path)
    for row in rows:
        expected = (
            int(round(row["expected_x_px"])),
            int(round(row["expected_y_px"])),
        )
        detected = (
            int(round(row["detected_x_px"])),
            int(round(row["detected_y_px"])),
        )
        cv2.arrowedLine(overlay, expected, detected, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(
            overlay,
            expected,
            (40, 220, 40),
            cv2.MARKER_CROSS,
            22,
            2,
            cv2.LINE_AA,
        )
        draw_ball_marker(
            overlay,
            {
                "x_px": row["detected_x_px"],
                "y_px": row["detected_y_px"],
                "label": row["ball_label"],
            },
        )
        draw_text_with_outline(
            overlay,
            f"{row['spot']} {row['error_mm']:.1f} mm",
            (detected[0] + 8, detected[1] - 8),
            scale=0.45,
        )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (20, 20, 20), -1)
    draw_text_with_outline(overlay, f"spot positions: {len(rows)} balls", (12, 28), scale=0.65)
    return overlay


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())




