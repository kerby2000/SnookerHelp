from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from snookerhelp.qa.validation import (
    add_region_to_row,
    available_source_z_center_modes,
    ball_by_id,
    ball_points_from_state,
    default_region_margin_mm,
    distance_mm,
    draw_ball_marker,
    draw_text_with_outline,
    load_json,
    load_yaml_or_json,
    state_display_name,
    state_matches_selector,
    summarize_by_region,
    summarize_values,
    table_dimensions_from_state,
    warped_overlay_base,
    write_csv,
    write_json,
)
from snookerhelp.calibration.camera import parse_z_center_method
from snookerhelp.core.config import PROJECT_ROOT, resolve_path


CSV_FIELDS = [
    "mode",
    "center_method",
    "z_mm",
    "state_file",
    "source_image",
    "pair_id",
    "pair_key",
    "ball_a_id",
    "ball_a_label",
    "ball_a_source_refinement_success",
    "ball_b_id",
    "ball_b_label",
    "ball_b_source_refinement_success",
    "distance_mm",
    "expected_distance_mm",
    "signed_error_mm",
    "abs_error_mm",
    "midpoint_x_mm",
    "midpoint_y_mm",
    "region",
    "notes",
]

Z_SUMMARY_FIELDS = [
    "region",
    "center_method",
    "z_mm",
    "pair_count",
    "mean_abs_error_mm",
    "median_abs_error_mm",
    "p95_abs_error_mm",
    "max_abs_error_mm",
    "rank_by_region_median",
    "is_best_by_region",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate physical center distances for touching balls or red-ball racks"
    )
    parser.add_argument(
        "detector_outputs",
        nargs="+",
        help="Processed detector state JSON files, e.g. *_state.json",
    )
    parser.add_argument("--pairs-file", default=None, help="YAML/JSON explicit ball-id pairs")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-find candidate touching pairs within the configured distance range.",
    )
    parser.add_argument(
        "--rack-reds",
        action="store_true",
        help="Evaluate nearest-neighbor distances among red balls instead of explicit/auto pairs.",
    )
    parser.add_argument("--expected-diameter-mm", type=float, default=52.5)
    parser.add_argument(
        "--center-mode",
        choices=("warped", "source-refined", "compare", "z-planes", "all"),
        default="warped",
        help=(
            "Distance source: old warped centers, source-refined centers mapped "
            "through the current approximate homography, Z-plane projections, "
            "or combinations."
        ),
    )
    parser.add_argument("--min-distance-mm", type=float, default=40.0)
    parser.add_argument("--max-distance-mm", type=float, default=70.0)
    parser.add_argument(
        "--class-relation",
        choices=("any", "same", "different"),
        default="any",
        help="Class filter for auto pair finding.",
    )
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--region-margin-mm",
        type=float,
        default=None,
        help="Distance from cushion used for region grouping. Default: 2 ball diameters.",
    )
    args = parser.parse_args(argv)

    if args.rack_reds and args.pairs_file:
        raise ValueError("--rack-reds and --pairs-file are separate modes")

    state_paths = [resolve_path(path) for path in args.detector_outputs]
    states = [(path, load_json(path)) for path in state_paths]
    region_margin_mm = (
        args.region_margin_mm
        if args.region_margin_mm is not None
        else default_region_margin_mm(args.expected_diameter_mm)
    )
    center_modes = _center_modes_for_args(args.center_mode, states)

    if args.rack_reds:
        rows = _evaluate_rack_reds(
            states=states,
            expected_diameter_mm=args.expected_diameter_mm,
            region_margin_mm=region_margin_mm,
            center_modes=center_modes,
        )
        mode = "rack_red_nearest_neighbor"
    else:
        pair_specs = _load_pair_specs(args.pairs_file) if args.pairs_file else None
        rows = _evaluate_touching_pairs(
            states=states,
            pair_specs=pair_specs,
            auto_find=args.auto or pair_specs is None,
            expected_diameter_mm=args.expected_diameter_mm,
            min_distance_mm=args.min_distance_mm,
            max_distance_mm=args.max_distance_mm,
            class_relation=args.class_relation,
            region_margin_mm=region_margin_mm,
            center_modes=center_modes,
        )
        mode = "touching_pairs"

    summary = {
        "mode": mode,
        "center_mode": args.center_mode,
        "state_count": len(states),
        "pair_count": len(rows),
        "expected_distance_mm": args.expected_diameter_mm,
        "region_margin_mm": region_margin_mm,
        "distance_mm": summarize_values(row["distance_mm"] for row in rows),
        "signed_error_mm": summarize_values(row["signed_error_mm"] for row in rows),
        "abs_error_mm": summarize_values(row["abs_error_mm"] for row in rows),
        "by_region": summarize_by_region(rows, "abs_error_mm"),
        "by_center_method": _summarize_by_center_method(rows),
        "z_plane_analysis": _summarize_z_planes(rows),
    }
    if args.center_mode == "compare":
        summary["source_refinement_comparison"] = _compare_center_methods(rows)
    if args.rack_reds:
        summary["nearest_neighbor_distance_spread_mm"] = {
            "std": summary["distance_mm"]["std"],
            "iqr": _iqr([row["distance_mm"] for row in rows]),
        }

    report = {
        "summary": summary,
        "settings": {
            "expected_diameter_mm": args.expected_diameter_mm,
            "min_distance_mm": args.min_distance_mm,
            "max_distance_mm": args.max_distance_mm,
            "class_relation": args.class_relation,
            "center_mode": args.center_mode,
        },
        "pairs": rows,
    }

    output_directory = (
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "physical_validation" / mode
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{mode}.json"
    csv_path = output_directory / f"{mode}.csv"
    z_summary_path = output_directory / f"{mode}_z_plane_summary.csv"
    z_heatmap_path = output_directory / f"{mode}_z_plane_region_heatmap.png"
    write_json(json_path, report)
    write_csv(csv_path, rows, CSV_FIELDS)
    z_summary_rows = _z_plane_summary_rows(summary["z_plane_analysis"])
    if z_summary_rows:
        write_csv(z_summary_path, z_summary_rows, Z_SUMMARY_FIELDS)
        cv2.imwrite(
            str(z_heatmap_path),
            _draw_z_plane_heatmap(z_summary_rows),
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

    for state_path, state in states:
        state_rows = [
            row for row in rows if Path(row["state_file"]).resolve() == state_path.resolve()
        ]
        if not state_rows:
            continue
        overlay = _draw_pair_overlay(
            state=state,
            rows=state_rows,
            config_path=args.config,
            title=mode,
        )
        overlay_path = output_directory / f"{state_display_name(state, state_path)}_{mode}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

    print(f"Mode: {mode}")
    print(f"Pairs evaluated: {summary['pair_count']}")
    print(
        "Distance median/spread: "
        f"{_fmt(summary['distance_mm']['median'])} mm / "
        f"std={_fmt(summary['distance_mm']['std'])} mm"
    )
    print(
        "Absolute error mean/median/max: "
        f"{_fmt(summary['abs_error_mm']['mean'])} / "
        f"{_fmt(summary['abs_error_mm']['median'])} / "
        f"{_fmt(summary['abs_error_mm']['max'])} mm"
    )
    if args.center_mode == "compare":
        comparison = summary["source_refinement_comparison"]
        print(
            "Source refinement improvement, mean/median: "
            f"{_fmt(comparison['mean_abs_error_improvement_mm'])} / "
            f"{_fmt(comparison['median_abs_error_improvement_mm'])} mm "
            f"({comparison['improved_pairs']}/{comparison['compared_pairs']} pairs improved)"
        )
    if summary["z_plane_analysis"]["best_z_by_region"]:
        print("Best Z by region, median abs error:")
        for region, details in summary["z_plane_analysis"]["best_z_by_region"].items():
            print(
                f"  - {region}: z={_fmt(details['z_mm'])} mm, "
                f"median={_fmt(details['median_abs_error_mm'])} mm"
            )
    print(f"Reports: {output_directory}")
    return 0


def _evaluate_touching_pairs(
    states: list[tuple[Path, dict[str, Any]]],
    pair_specs: list[dict[str, Any]] | None,
    auto_find: bool,
    expected_diameter_mm: float,
    min_distance_mm: float,
    max_distance_mm: float,
    class_relation: str,
    region_margin_mm: float,
    center_modes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_path, state in states:
        points = ball_points_from_state(state, center_mode="warped")
        if auto_find:
            pairs = _auto_pairs(
                points,
                min_distance_mm,
                max_distance_mm,
                class_relation,
                expected_diameter_mm,
            )
        else:
            pairs = []
            assert pair_specs is not None
            for spec in pair_specs:
                selector = spec.get("state") or spec.get("file") or spec.get("image")
                if not state_matches_selector(state, state_path, selector):
                    continue
                a = ball_by_id(state, int(spec["ball_a"]), center_mode="warped")
                b = ball_by_id(state, int(spec["ball_b"]), center_mode="warped")
                pairs.append((a, b, spec.get("notes")))
        for pair_index, (a, b, notes) in enumerate(pairs, start=1):
            for center_mode in center_modes:
                method_a = ball_by_id(state, int(a["id"]), center_mode=center_mode)
                method_b = ball_by_id(state, int(b["id"]), center_mode=center_mode)
                rows.append(
                    _distance_row(
                        state_path=state_path,
                        state=state,
                        mode="touching_pairs",
                        center_method=center_mode,
                        pair_id=pair_index,
                        a=method_a,
                        b=method_b,
                        expected_distance_mm=expected_diameter_mm,
                        region_margin_mm=region_margin_mm,
                        notes=notes,
                    )
                )
    return rows


def _evaluate_rack_reds(
    states: list[tuple[Path, dict[str, Any]]],
    expected_diameter_mm: float,
    region_margin_mm: float,
    center_modes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_path, state in states:
        reds = [
            point
            for point in ball_points_from_state(state, center_mode="warped")
            if point["label"] == "red"
        ]
        for pair_index, point in enumerate(reds, start=1):
            others = [other for other in reds if other["id"] != point["id"]]
            if not others:
                continue
            nearest = min(others, key=lambda other: distance_mm(point, other))
            for center_mode in center_modes:
                method_point = ball_by_id(
                    state, int(point["id"]), center_mode=center_mode
                )
                method_nearest = ball_by_id(
                    state, int(nearest["id"]), center_mode=center_mode
                )
                rows.append(
                    _distance_row(
                        state_path=state_path,
                        state=state,
                        mode="rack_red_nearest_neighbor",
                        center_method=center_mode,
                        pair_id=pair_index,
                        a=method_point,
                        b=method_nearest,
                        expected_distance_mm=expected_diameter_mm,
                        region_margin_mm=region_margin_mm,
                        notes="nearest red neighbor",
                    )
                )
    return rows


def _distance_row(
    state_path: Path,
    state: dict[str, Any],
    mode: str,
    center_method: str,
    pair_id: int,
    a: dict[str, Any],
    b: dict[str, Any],
    expected_distance_mm: float,
    region_margin_mm: float,
    notes: str | None,
) -> dict[str, Any]:
    length_mm, width_mm = table_dimensions_from_state(state)
    measured = distance_mm(a, b)
    midpoint_x_mm = 0.5 * (a["x_mm"] + b["x_mm"])
    midpoint_y_mm = 0.5 * (a["y_mm"] + b["y_mm"])
    row = {
        "mode": mode,
        "center_method": center_method,
        "z_mm": a.get("z_mm"),
        "state_file": str(state_path.resolve()),
        "source_image": state.get("source_image"),
        "pair_id": pair_id,
        "pair_key": _pair_key(state_path, pair_id, a, b),
        "ball_a_id": a["id"],
        "ball_a_label": a["label"],
        "ball_a_source_refinement_success": a.get(
            "source_refinement_success", False
        ),
        "ball_a_x_px": a["x_px"],
        "ball_a_y_px": a["y_px"],
        "ball_a_x_mm": a["x_mm"],
        "ball_a_y_mm": a["y_mm"],
        "ball_b_id": b["id"],
        "ball_b_label": b["label"],
        "ball_b_source_refinement_success": b.get(
            "source_refinement_success", False
        ),
        "ball_b_x_px": b["x_px"],
        "ball_b_y_px": b["y_px"],
        "ball_b_x_mm": b["x_mm"],
        "ball_b_y_mm": b["y_mm"],
        "distance_mm": measured,
        "expected_distance_mm": expected_distance_mm,
        "signed_error_mm": measured - expected_distance_mm,
        "abs_error_mm": abs(measured - expected_distance_mm),
        "midpoint_x_mm": midpoint_x_mm,
        "midpoint_y_mm": midpoint_y_mm,
        "notes": notes,
    }
    add_region_to_row(
        row,
        x_mm=midpoint_x_mm,
        y_mm=midpoint_y_mm,
        table_length_mm=length_mm,
        table_width_mm=width_mm,
        edge_margin_mm=region_margin_mm,
    )
    return row


def _pair_key(
    state_path: Path,
    pair_id: int,
    a: dict[str, Any],
    b: dict[str, Any],
) -> str:
    ids = sorted([int(a["id"]), int(b["id"])])
    return f"{state_path.resolve()}:{pair_id}:{ids[0]}-{ids[1]}"


def _auto_pairs(
    points: list[dict[str, Any]],
    min_distance_mm: float,
    max_distance_mm: float,
    class_relation: str,
    expected_distance_mm: float,
) -> list[tuple[dict[str, Any], dict[str, Any], str | None]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
    for a, b in combinations(points, 2):
        if class_relation == "same" and a["label"] != b["label"]:
            continue
        if class_relation == "different" and a["label"] == b["label"]:
            continue
        measured = distance_mm(a, b)
        if min_distance_mm <= measured <= max_distance_mm:
            pairs.append((a, b, "auto"))
    return sorted(
        pairs,
        key=lambda pair: abs(distance_mm(pair[0], pair[1]) - expected_distance_mm),
    )


def _center_modes_for_args(
    requested: str,
    states: list[tuple[Path, dict[str, Any]]],
) -> list[str]:
    z_modes = _available_z_modes(states)
    if requested == "warped":
        return ["warped"]
    if requested == "source-refined":
        return ["source_refined"]
    if requested == "compare":
        return ["warped", "source_refined"]
    if requested == "z-planes":
        if not z_modes:
            raise ValueError("No source_refined_table_xy_by_z_mm entries found")
        return z_modes
    if requested == "all":
        return ["warped", "source_refined", *z_modes]
    raise ValueError(f"Unsupported center mode: {requested}")


def _available_z_modes(states: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    modes: list[str] = []
    for _, state in states:
        for mode in available_source_z_center_modes(state):
            if mode not in modes:
                modes.append(mode)
    return modes


def _summarize_by_center_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["center_method"]) for row in rows})
    return {
        method: {
            "pair_count": len([row for row in rows if row["center_method"] == method]),
            "distance_mm": summarize_values(
                row["distance_mm"]
                for row in rows
                if row["center_method"] == method
            ),
            "abs_error_mm": summarize_values(
                row["abs_error_mm"]
                for row in rows
                if row["center_method"] == method
            ),
            "by_region": summarize_by_region(
                [row for row in rows if row["center_method"] == method],
                "abs_error_mm",
            ),
        }
        for method in methods
    }


def _compare_center_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_key"]), {})[str(row["center_method"])] = row
    improvements: list[float] = []
    compared_rows: list[dict[str, Any]] = []
    for pair_key, method_rows in grouped.items():
        warped = method_rows.get("warped")
        source = method_rows.get("source_refined")
        if warped is None or source is None:
            continue
        improvement = float(warped["abs_error_mm"]) - float(source["abs_error_mm"])
        improvements.append(improvement)
        compared_rows.append(
            {
                "pair_key": pair_key,
                "warped_abs_error_mm": warped["abs_error_mm"],
                "source_refined_abs_error_mm": source["abs_error_mm"],
                "abs_error_improvement_mm": improvement,
                "region": warped.get("region"),
            }
        )
    values = np.asarray(improvements, dtype=float)
    by_region: dict[str, Any] = {}
    for region in sorted({str(row.get("region")) for row in compared_rows}):
        region_values = [
            row["abs_error_improvement_mm"]
            for row in compared_rows
            if str(row.get("region")) == region
        ]
        by_region[region] = {
            **summarize_values(region_values),
            "improved_pairs": int(
                np.count_nonzero(np.asarray(region_values, dtype=float) > 0.0)
            ),
        }
    return {
        "compared_pairs": len(improvements),
        "improved_pairs": int(np.count_nonzero(values > 0.0)) if values.size else 0,
        "mean_abs_error_improvement_mm": (
            float(np.mean(values)) if values.size else None
        ),
        "median_abs_error_improvement_mm": (
            float(np.median(values)) if values.size else None
        ),
        "max_abs_error_improvement_mm": (
            float(np.max(values)) if values.size else None
        ),
        "worst_abs_error_regression_mm": (
            float(np.min(values)) if values.size else None
        ),
        "by_region": by_region,
        "pairs": compared_rows,
    }


def _summarize_z_planes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    z_rows = [
        row for row in rows if parse_z_center_method(str(row["center_method"])) is not None
    ]
    regions = sorted({str(row["region"]) for row in z_rows})
    methods = sorted(
        {str(row["center_method"]) for row in z_rows},
        key=lambda method: parse_z_center_method(method) or 0.0,
    )
    by_region: dict[str, dict[str, Any]] = {}
    best_z_by_region: dict[str, dict[str, Any]] = {}
    for region in regions:
        region_summary: dict[str, Any] = {}
        for method in methods:
            method_rows = [
                row
                for row in z_rows
                if row["region"] == region and row["center_method"] == method
            ]
            if not method_rows:
                continue
            stats = summarize_values(row["abs_error_mm"] for row in method_rows)
            z_mm = parse_z_center_method(method)
            region_summary[method] = {
                "z_mm": z_mm,
                "pair_count": len(method_rows),
                "abs_error_mm": stats,
            }
        if region_summary:
            best_method, best_summary = min(
                region_summary.items(),
                key=lambda item: (
                    float("inf")
                    if item[1]["abs_error_mm"]["median"] is None
                    else item[1]["abs_error_mm"]["median"]
                ),
            )
            best_z_by_region[region] = {
                "center_method": best_method,
                "z_mm": best_summary["z_mm"],
                "median_abs_error_mm": best_summary["abs_error_mm"]["median"],
                "mean_abs_error_mm": best_summary["abs_error_mm"]["mean"],
                "pair_count": best_summary["pair_count"],
            }
        by_region[region] = region_summary
    return {
        "z_plane_pair_rows": len(z_rows),
        "by_region": by_region,
        "best_z_by_region": best_z_by_region,
    }


def _z_plane_summary_rows(z_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region, region_summary in z_analysis["by_region"].items():
        ordered = sorted(
            region_summary.items(),
            key=lambda item: (
                float("inf")
                if item[1]["abs_error_mm"]["median"] is None
                else item[1]["abs_error_mm"]["median"]
            ),
        )
        for rank, (method, summary) in enumerate(ordered, start=1):
            stats = summary["abs_error_mm"]
            rows.append(
                {
                    "region": region,
                    "center_method": method,
                    "z_mm": summary["z_mm"],
                    "pair_count": summary["pair_count"],
                    "mean_abs_error_mm": stats["mean"],
                    "median_abs_error_mm": stats["median"],
                    "p95_abs_error_mm": stats["p95"],
                    "max_abs_error_mm": stats["max"],
                    "rank_by_region_median": rank,
                    "is_best_by_region": rank == 1,
                }
            )
    return rows


def _draw_z_plane_heatmap(rows: list[dict[str, Any]]) -> np.ndarray:
    regions = sorted({str(row["region"]) for row in rows})
    methods = sorted(
        {str(row["center_method"]) for row in rows},
        key=lambda method: parse_z_center_method(method) or 0.0,
    )
    cell_w = 150
    cell_h = 54
    left_w = 145
    top_h = 80
    width = left_w + cell_w * max(1, len(methods)) + 30
    height = top_h + cell_h * max(1, len(regions)) + 40
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    values = [
        float(row["median_abs_error_mm"])
        for row in rows
        if row["median_abs_error_mm"] is not None
    ]
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    draw_text_with_outline(
        image,
        "Median touching-pair abs error by Z plane and region",
        (14, 30),
        scale=0.58,
        color=(20, 20, 20),
        thickness=1,
    )
    for column, method in enumerate(methods):
        z_mm = parse_z_center_method(method)
        x = left_w + column * cell_w + 8
        draw_text_with_outline(
            image,
            f"Z={_fmt(z_mm)}",
            (x, top_h - 18),
            scale=0.45,
            color=(20, 20, 20),
            thickness=1,
        )
    row_by_key = {
        (str(row["region"]), str(row["center_method"])): row
        for row in rows
    }
    for r_index, region in enumerate(regions):
        y = top_h + r_index * cell_h
        draw_text_with_outline(
            image,
            region,
            (10, y + 34),
            scale=0.45,
            color=(20, 20, 20),
            thickness=1,
        )
        for c_index, method in enumerate(methods):
            x = left_w + c_index * cell_w
            row = row_by_key.get((region, method))
            if row is None or row["median_abs_error_mm"] is None:
                color = (220, 220, 220)
                text = "n/a"
                best = False
            else:
                value = float(row["median_abs_error_mm"])
                normalized = min(1.0, value / max_value)
                color = (
                    int(70 + 150 * normalized),
                    int(220 - 150 * normalized),
                    70,
                )
                text = f"{value:.2f} mm"
                best = bool(row["is_best_by_region"])
            cv2.rectangle(
                image,
                (x + 4, y + 4),
                (x + cell_w - 4, y + cell_h - 4),
                color,
                -1,
            )
            cv2.rectangle(
                image,
                (x + 4, y + 4),
                (x + cell_w - 4, y + cell_h - 4),
                (20, 20, 20) if best else (170, 170, 170),
                3 if best else 1,
            )
            draw_text_with_outline(
                image,
                text,
                (x + 14, y + 34),
                scale=0.45,
                color=(20, 20, 20),
                thickness=1,
            )
    return image


def _load_pair_specs(path: str | Path) -> list[dict[str, Any]]:
    payload = load_yaml_or_json(path)
    raw_items = payload.get("pairs", payload.get("touching_pairs", [])) if isinstance(payload, dict) else payload
    specs: list[dict[str, Any]] = []
    for item in raw_items or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            specs.append({"ball_a": int(item[0]), "ball_b": int(item[1])})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported pair specification: {item!r}")
        specs.append(
            {
                "ball_a": int(
                    item.get("ball_a", item.get("a", item.get("id_a")))
                ),
                "ball_b": int(
                    item.get("ball_b", item.get("b", item.get("id_b")))
                ),
                "state": item.get("state", item.get("file", item.get("image"))),
                "notes": item.get("notes"),
            }
        )
    return specs


def _draw_pair_overlay(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    config_path: str | Path,
    title: str,
) -> np.ndarray:
    overlay = warped_overlay_base(state, config_path)
    for row in rows:
        a = {
            "x_px": row["ball_a_x_px"],
            "y_px": row["ball_a_y_px"],
            "label": row["ball_a_label"],
        }
        b = {
            "x_px": row["ball_b_x_px"],
            "y_px": row["ball_b_y_px"],
            "label": row["ball_b_label"],
        }
        p1 = (int(round(a["x_px"])), int(round(a["y_px"])))
        p2 = (int(round(b["x_px"])), int(round(b["y_px"])))
        color = (40, 220, 40) if row["abs_error_mm"] <= 5.0 else (0, 140, 255)
        cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)
        draw_ball_marker(overlay, a)
        draw_ball_marker(overlay, b)
        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        draw_text_with_outline(
            overlay,
            (
                f"{row['center_method']} "
                f"{row['distance_mm']:.1f} mm "
                f"({row['signed_error_mm']:+.1f})"
            ),
            (midpoint[0] + 6, midpoint[1] - 6),
            scale=0.42,
        )
    header = f"{title}: {len(rows)} pairs"
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (20, 20, 20), -1)
    draw_text_with_outline(overlay, header, (12, 28), scale=0.65)
    return overlay


def _iqr(values: list[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, 75) - np.percentile(array, 25))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())





