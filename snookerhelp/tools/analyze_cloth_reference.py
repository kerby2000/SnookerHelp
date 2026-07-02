from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from snookerhelp.core.config import resolve_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze active/global/local cloth references in v1 reports.",
    )
    parser.add_argument("--reports", default="outputs/reports_v1_evidence_maps")
    parser.add_argument("--output", default="outputs/cloth_reference_analysis")
    args = parser.parse_args(argv)

    reports_root = resolve_path(args.reports)
    output_root = resolve_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _collect_rows(reports_root)
    if not rows:
        raise FileNotFoundError(f"No report.json files with balls found under {reports_root}")

    csv_path = output_root / "cloth_reference_by_ball.csv"
    json_path = output_root / "cloth_reference_summary.json"
    _write_csv(csv_path, rows)
    summary = _summary(rows, reports_root)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(
        "Mean local-vs-global Delta-E: "
        f"{summary['mean_local_to_global_delta_e']:.2f}"
    )
    print(
        "Low-contrast active/local rows: "
        f"{summary['active_low_contrast_count']} / {summary['local_low_contrast_count']}"
    )
    return 0


def _collect_rows(reports_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(reports_root.glob("*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        image = str(report.get("image") or report_path.parent.name)
        review_balls = {
            int(ball["id"]): ball
            for ball in ((report.get("review_evidence") or {}).get("balls") or [])
            if "id" in ball
        }
        for state_ball in (report.get("state") or {}).get("balls", []):
            ball_id = int(state_ball["id"])
            review_ball = review_balls.get(ball_id, {})
            maps = (
                review_ball.get("evidence_maps")
                or state_ball.get("source_evidence_maps")
                or {}
            )
            if not maps:
                continue
            active = maps.get("active_color_model") or maps.get("local_color_model") or {}
            local = maps.get("local_color_model") or {}
            global_model = maps.get("global_cloth_model") or {}
            variants = maps.get("boundary_variants") or {}
            row = {
                "image": Path(image).name,
                "stem": report_path.parent.name,
                "ball_id": ball_id,
                "label": review_ball.get("label")
                or state_ball.get("color_label")
                or state_ball.get("class"),
                "active_mode": active.get("cloth_reference_mode"),
                "active_cloth_lab": _lab_text(active.get("cloth_lab")),
                "local_cloth_lab": _lab_text(local.get("cloth_lab")),
                "global_cloth_lab": _lab_text(global_model.get("cloth_lab")),
                "ball_lab": _lab_text(active.get("ball_lab") or local.get("ball_lab")),
                "active_separation_lab": _number(active.get("separation_lab")),
                "active_separation_chroma": _number(active.get("separation_chroma")),
                "local_separation_lab": _number(local.get("separation_lab")),
                "local_separation_chroma": _number(local.get("separation_chroma")),
                "local_to_global_delta_e": _lab_distance(
                    local.get("cloth_lab"),
                    global_model.get("cloth_lab"),
                ),
                "active_low_contrast": bool(active.get("low_contrast", False)),
                "local_low_contrast": bool(local.get("low_contrast", False)),
                "active_cloth_sample_count": active.get("cloth_sample_count"),
                "local_cloth_sample_count": local.get("cloth_sample_count"),
                "global_cloth_sample_count": global_model.get("sample_count"),
                "source_score": _score(review_ball.get("boundary_view_score")),
                "lab_delta_e_score": _score(variants.get("lab_delta_e", {}).get("view_score")),
                "chroma_difference_score": _score(
                    variants.get("chroma_difference", {}).get("view_score"),
                ),
                "ball_vs_cloth_probability_score": _score(
                    variants.get("ball_vs_cloth_probability", {}).get("view_score"),
                ),
                "combined_boundary_score": _score(
                    variants.get("combined_boundary_score", {}).get("view_score"),
                ),
                "final_map": (
                    review_ball.get("final_image_evidence")
                    or state_ball.get("source_final_center_policy")
                    or {}
                ).get("selected_map"),
                "confidence": _number(review_ball.get("review_confidence")),
            }
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]], reports_root: Path) -> dict[str, Any]:
    distances = [
        float(row["local_to_global_delta_e"])
        for row in rows
        if row["local_to_global_delta_e"] is not None
    ]
    return {
        "schema_version": "snookerhelp.cloth_reference_analysis.v1",
        "reports_root": str(reports_root),
        "row_count": len(rows),
        "image_count": len({row["stem"] for row in rows}),
        "mean_local_to_global_delta_e": mean(distances) if distances else 0.0,
        "max_local_to_global_delta_e": max(distances) if distances else 0.0,
        "active_low_contrast_count": sum(1 for row in rows if row["active_low_contrast"]),
        "local_low_contrast_count": sum(1 for row in rows if row["local_low_contrast"]),
        "worst_local_to_global_delta_e": sorted(
            [
                {
                    "image": row["image"],
                    "ball_id": row["ball_id"],
                    "label": row["label"],
                    "local_to_global_delta_e": row["local_to_global_delta_e"],
                    "local_cloth_lab": row["local_cloth_lab"],
                    "global_cloth_lab": row["global_cloth_lab"],
                    "active_separation_lab": row["active_separation_lab"],
                    "local_separation_lab": row["local_separation_lab"],
                }
                for row in rows
                if row["local_to_global_delta_e"] is not None
            ],
            key=lambda item: float(item["local_to_global_delta_e"]),
            reverse=True,
        )[:25],
    }


def _lab_text(values: Any) -> str | None:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    return ",".join(f"{float(value):.2f}" for value in values[:3])


def _lab_distance(a: Any, b: Any) -> float | None:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return None
    if len(a) < 3 or len(b) < 3:
        return None
    return round(
        sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5,
        4,
    )


def _score(payload: Any) -> float | None:
    if not isinstance(payload, dict) or payload.get("score") is None:
        return None
    return round(float(payload["score"]), 4)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
