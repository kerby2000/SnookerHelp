from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2


from snookerhelp.core.config import PROJECT_ROOT, resolve_path
from snookerhelp.qa.validation import (
    add_region_to_row,
    ball_by_id,
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
    "ball_id",
    "ball_label",
    "cushion",
    "distance_to_cushion_mm",
    "expected_radius_mm",
    "signed_error_mm",
    "abs_error_mm",
    "x_mm",
    "y_mm",
    "region",
    "notes",
]


VALID_CUSHIONS = {"left", "right", "top", "bottom"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate balls that are physically touching a cushion"
    )
    parser.add_argument("detector_outputs", nargs="+", help="Processed detector state JSON files")
    parser.add_argument("--touches", default=None, help="YAML/JSON touch specification file")
    parser.add_argument(
        "--touch",
        action="append",
        default=[],
        help="Inline touch spec as ball_id:cushion, e.g. 7:left. Applies to all input states.",
    )
    parser.add_argument("--expected-radius-mm", type=float, default=26.25)
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--region-margin-mm",
        type=float,
        default=None,
        help="Distance from cushion used for region grouping. Default: 2 ball diameters.",
    )
    args = parser.parse_args(argv)

    specs = []
    if args.touches:
        specs.extend(_load_touch_specs(args.touches))
    specs.extend(_parse_inline_touch(spec) for spec in args.touch)
    if not specs:
        raise ValueError("Provide --touches or at least one --touch ball_id:cushion")

    state_paths = [resolve_path(path) for path in args.detector_outputs]
    states = [(path, load_json(path)) for path in state_paths]
    region_margin_mm = (
        args.region_margin_mm
        if args.region_margin_mm is not None
        else default_region_margin_mm(args.expected_radius_mm * 2.0)
    )

    rows: list[dict[str, Any]] = []
    for state_path, state in states:
        for spec in specs:
            selector = spec.get("state") or spec.get("file") or spec.get("image")
            if not state_matches_selector(state, state_path, selector):
                continue
            point = ball_by_id(state, int(spec["ball_id"]))
            rows.append(
                _touch_row(
                    state_path=state_path,
                    state=state,
                    point=point,
                    cushion=spec["cushion"],
                    expected_radius_mm=args.expected_radius_mm,
                    region_margin_mm=region_margin_mm,
                    notes=spec.get("notes"),
                )
            )

    summary = {
        "touch_count": len(rows),
        "expected_radius_mm": args.expected_radius_mm,
        "region_margin_mm": region_margin_mm,
        "distance_to_cushion_mm": summarize_values(
            row["distance_to_cushion_mm"] for row in rows
        ),
        "signed_error_mm": summarize_values(row["signed_error_mm"] for row in rows),
        "abs_error_mm": summarize_values(row["abs_error_mm"] for row in rows),
        "by_region": summarize_by_region(rows, "abs_error_mm"),
    }
    report = {"summary": summary, "touches": rows}

    output_directory = (
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "physical_validation" / "cushion_touch"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "cushion_touch.json", report)
    write_csv(output_directory / "cushion_touch.csv", rows, CSV_FIELDS)

    for state_path, state in states:
        state_rows = [
            row for row in rows if Path(row["state_file"]).resolve() == state_path.resolve()
        ]
        if not state_rows:
            continue
        overlay = _draw_cushion_overlay(state, state_rows, args.config)
        overlay_path = output_directory / f"{state_display_name(state, state_path)}_cushion_touch_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

    print(f"Cushion touches evaluated: {summary['touch_count']}")
    print(
        "Absolute radius error mean/median/max: "
        f"{_fmt(summary['abs_error_mm']['mean'])} / "
        f"{_fmt(summary['abs_error_mm']['median'])} / "
        f"{_fmt(summary['abs_error_mm']['max'])} mm"
    )
    print(f"Reports: {output_directory}")
    return 0


def _touch_row(
    state_path: Path,
    state: dict[str, Any],
    point: dict[str, Any],
    cushion: str,
    expected_radius_mm: float,
    region_margin_mm: float,
    notes: str | None,
) -> dict[str, Any]:
    if cushion not in VALID_CUSHIONS:
        raise ValueError(f"cushion must be one of {sorted(VALID_CUSHIONS)}")
    length_mm, width_mm = table_dimensions_from_state(state)
    x_mm = float(point["x_mm"])
    y_mm = float(point["y_mm"])
    if cushion == "left":
        distance = x_mm
        contact = (0.0, y_mm)
    elif cushion == "right":
        distance = length_mm - x_mm
        contact = (length_mm, y_mm)
    elif cushion == "bottom":
        distance = y_mm
        contact = (x_mm, 0.0)
    else:
        distance = width_mm - y_mm
        contact = (x_mm, width_mm)

    row = {
        "state_file": str(state_path.resolve()),
        "source_image": state.get("source_image"),
        "ball_id": point["id"],
        "ball_label": point["label"],
        "ball_x_px": point["x_px"],
        "ball_y_px": point["y_px"],
        "cushion_contact_x_px": table_mm_to_warped_px_from_state(state, *contact)[0],
        "cushion_contact_y_px": table_mm_to_warped_px_from_state(state, *contact)[1],
        "cushion": cushion,
        "distance_to_cushion_mm": distance,
        "expected_radius_mm": expected_radius_mm,
        "signed_error_mm": distance - expected_radius_mm,
        "abs_error_mm": abs(distance - expected_radius_mm),
        "x_mm": x_mm,
        "y_mm": y_mm,
        "notes": notes,
    }
    add_region_to_row(
        row,
        x_mm=x_mm,
        y_mm=y_mm,
        table_length_mm=length_mm,
        table_width_mm=width_mm,
        edge_margin_mm=region_margin_mm,
    )
    return row


def _load_touch_specs(path: str | Path) -> list[dict[str, Any]]:
    payload = load_yaml_or_json(path)
    items = payload.get("touches", payload.get("cushion_touches", [])) if isinstance(payload, dict) else payload
    specs = []
    for item in items or []:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported touch specification: {item!r}")
        cushion = str(item.get("cushion", "")).lower()
        specs.append(
            {
                "ball_id": int(item.get("ball_id", item.get("id", item.get("ball")))),
                "cushion": cushion,
                "state": item.get("state", item.get("file", item.get("image"))),
                "notes": item.get("notes"),
            }
        )
    return specs


def _parse_inline_touch(spec: str) -> dict[str, Any]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError("--touch must be formatted as ball_id:cushion")
    return {"ball_id": int(parts[0]), "cushion": parts[1].lower()}


def _draw_cushion_overlay(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    config_path: str | Path,
) -> Any:
    overlay = warped_overlay_base(state, config_path)
    for row in rows:
        center = (int(round(row["ball_x_px"])), int(round(row["ball_y_px"])))
        contact = (
            int(round(row["cushion_contact_x_px"])),
            int(round(row["cushion_contact_y_px"])),
        )
        color = (40, 220, 40) if row["abs_error_mm"] <= 5.0 else (0, 140, 255)
        cv2.line(overlay, center, contact, color, 2, cv2.LINE_AA)
        draw_ball_marker(
            overlay,
            {
                "x_px": row["ball_x_px"],
                "y_px": row["ball_y_px"],
                "label": row["ball_label"],
            },
        )
        draw_text_with_outline(
            overlay,
            f"{row['ball_id']} {row['cushion']} {row['signed_error_mm']:+.1f} mm",
            (center[0] + 8, center[1] - 8),
            scale=0.45,
        )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (20, 20, 20), -1)
    draw_text_with_outline(overlay, f"cushion touch: {len(rows)} balls", (12, 28), scale=0.65)
    return overlay


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())




