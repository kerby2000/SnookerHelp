from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

from snookerhelp.qa.reports import load_dataset_table_states
from snookerhelp.core.config import resolve_path


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    report_count: int
    ball_count: int
    mean_score: float | None
    level_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_count": self.report_count,
            "ball_count": self.ball_count,
            "mean_score": self.mean_score,
            "level_counts": self.level_counts,
        }


def summarize_v1_confidence(reports_root: str | Path = "outputs/reports") -> ConfidenceSummary:
    states = load_dataset_table_states(reports_root)
    scores: list[float] = []
    levels: dict[str, int] = {}
    ball_count = 0
    for state in states:
        for ball in state.balls:
            ball_count += 1
            if ball.confidence:
                if ball.confidence.score is not None:
                    scores.append(float(ball.confidence.score))
                level = ball.confidence.level
            else:
                level = "unknown"
            levels[level] = levels.get(level, 0) + 1
    return ConfidenceSummary(
        report_count=len(states),
        ball_count=ball_count,
        mean_score=round(mean(scores), 4) if scores else None,
        level_counts=dict(sorted(levels.items())),
    )


def benchmark_model_scoring_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy circle-first review confidence with experimental "
            "physical-model-first confidence."
        )
    )
    parser.add_argument("--reports", default="outputs/reports")
    parser.add_argument("--output", default="outputs/model_scoring_benchmark")
    args = parser.parse_args(argv)

    reports_root = resolve_path(args.reports)
    output_root = resolve_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = collect_model_scoring_rows(reports_root)
    if not rows:
        raise FileNotFoundError(f"No report.json files with review evidence under {reports_root}")

    summary = summarize_model_scoring_rows(rows)
    csv_path = output_root / "model_scoring_benchmark.csv"
    json_path = output_root / "model_scoring_benchmark.json"
    write_model_scoring_csv(csv_path, rows)
    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    print(f"Rows: {len(rows)}")
    print(f"Images: {summary['image_count']}")
    print(f"Legacy mean confidence: {summary['legacy_mean_confidence']:.3f}")
    print(f"Physics-first mean confidence: {summary['physics_mean_confidence']:.3f}")
    print(
        "Physical model + observed ellipse mean confidence: "
        f"{summary['physics_c_only_mean_confidence']:.3f}"
    )
    print(f"Displayed mean confidence: {summary['displayed_mean_confidence']:.3f}")
    print(f"Confidence improved by >=10 points: {summary['improved_by_10_points']}")
    print(f"Confidence reduced by >=10 points: {summary['reduced_by_10_points']}")
    print(f"Physics decisions: {summary['physics_status_counts']}")
    print(
        "Physical model + observed ellipse decisions: "
        f"{summary['physics_c_only_status_counts']}"
    )
    print(f"Sphere grades: {summary['sphere_grade_counts']}")
    print(f"Observed-ellipse sphere grades: {summary['observed_ellipse_sphere_grade_counts']}")
    print(f"Observed ellipse grades: {summary['observed_ellipse_grade_counts']}")
    print(f"Object evidence grades: {summary['object_grade_counts']}")
    print(
        "Mean accepted/rejected boundary points: "
        f"{summary['mean_accepted_boundary_points']:.1f} / "
        f"{summary['mean_rejected_boundary_points']:.1f}"
    )
    print(f"Evidence map statuses: {summary['evidence_map_status_counts']}")
    print(f"Mean evidence-map assets per ball: {summary['mean_evidence_map_asset_count']:.1f}")
    print(
        "Physical optimization statuses: "
        f"{summary['physical_optimization_status_counts']}"
    )
    print(f"Physical projection modes: {summary['physical_projection_mode_counts']}")
    print(f"Joint cluster statuses: {summary['joint_cluster_status_counts']}")
    print(
        "Mean joint cluster pair-distance improvement: "
        f"{summary['mean_joint_cluster_improvement_mm']:.3f} mm"
    )
    print(
        "Duplicate-warning false-positive proxy count: "
        f"{summary['duplicate_warning_false_positive_proxy_count']}"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


def collect_model_scoring_rows(reports_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(Path(reports_root).glob("*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        image = str(report.get("image") or report_path.parent.name)
        for ball in (report.get("review_evidence") or {}).get("balls", []):
            physics = ball.get("physics_first_model_decision") or {}
            physics_c_only = ball.get("physics_c_only_model_decision") or {}
            sphere_projection = ball.get("sphere_projection") or {}
            forward_projection = sphere_projection.get("forward_projection") or {}
            optimization = ball.get("physical_optimization") or sphere_projection.get("optimization") or {}
            joint_cluster = ball.get("joint_cluster_optimization") or optimization.get("joint_cluster") or {}
            evidence_maps = ball.get("evidence_maps") or {}
            local_color = evidence_maps.get("local_color_model") or {}
            sphere_grade = physics.get("sphere_grade") or {}
            observed_ellipse_sphere_grade = physics_c_only.get("sphere_grade") or {}
            object_grade = physics.get("object_evidence_grade") or {}
            observed_ellipse_grade = physics_c_only.get("candidate_c_grade") or {}
            legacy_confidence = _float_or_none(ball.get("legacy_review_confidence"))
            physics_confidence = _float_or_none(ball.get("physics_first_review_confidence"))
            c_only_confidence = _float_or_none(ball.get("physics_c_only_review_confidence"))
            displayed_confidence = _float_or_none(ball.get("review_confidence"))
            rows.append(
                {
                    "image": image,
                    "report": str(report_path),
                    "ball_id": int(ball.get("id", -1)),
                    "label": ball.get("label"),
                    "legacy_status": (ball.get("model_decision") or {}).get("status"),
                    "legacy_model": (ball.get("model_decision") or {}).get(
                        "selected_model",
                    ),
                    "physics_status": physics.get("status"),
                    "physics_model": physics.get("selected_model"),
                    "physics_c_only_status": physics_c_only.get("status"),
                    "physics_c_only_model": physics_c_only.get("selected_model"),
                    "legacy_confidence": legacy_confidence,
                    "physics_confidence": physics_confidence,
                    "physics_c_only_confidence": c_only_confidence,
                    "displayed_confidence": displayed_confidence,
                    "displayed_delta_vs_legacy": _delta(
                        displayed_confidence,
                        legacy_confidence,
                    ),
                    "physics_delta_vs_legacy": _delta(
                        physics_confidence,
                        legacy_confidence,
                    ),
                    "sphere_grade": sphere_grade.get("level"),
                    "sphere_rms_error_px": sphere_grade.get("rms_error_px"),
                    "sphere_normalized_rms": sphere_grade.get("normalized_rms"),
                    "observed_ellipse_sphere_grade": observed_ellipse_sphere_grade.get("level"),
                    "observed_ellipse_sphere_rms_error_px": observed_ellipse_sphere_grade.get(
                        "rms_error_px"
                    ),
                    "observed_ellipse_grade": observed_ellipse_grade.get("level"),
                    "observed_ellipse_source": observed_ellipse_grade.get("source"),
                    "observed_ellipse_point_count": observed_ellipse_grade.get("point_count"),
                    "accepted_boundary_point_count": len(ball.get("boundary_points_px") or []),
                    "rejected_boundary_point_count": len(
                        ball.get("boundary_rejected_points_px") or []
                    ),
                    "evidence_map_status": evidence_maps.get("status"),
                    "evidence_map_asset_count": len(evidence_maps.get("assets") or []),
                    "local_color_separation_lab": local_color.get("separation_lab"),
                    "local_color_separation_chroma": local_color.get("separation_chroma"),
                    "local_color_low_contrast": local_color.get("low_contrast"),
                    "physical_projection_mode": sphere_projection.get("projection_mode", "forward"),
                    "physical_optimization_status": optimization.get("status"),
                    "physical_optimization_success": optimization.get("success"),
                    "physical_optimization_movement_mm": optimization.get(
                        "movement_from_initial_mm"
                    ),
                    "joint_cluster_status": joint_cluster.get("cluster_status"),
                    "joint_cluster_size": joint_cluster.get("component_size"),
                    "joint_cluster_initial_pair_rms_mm": joint_cluster.get(
                        "initial_pair_rms_mm"
                    ),
                    "joint_cluster_pair_rms_mm": joint_cluster.get(
                        "joint_pair_rms_mm"
                    ),
                    "joint_cluster_improvement_mm": joint_cluster.get("improvement_mm"),
                    "sphere_forward_rms_error_px": (
                        (forward_projection.get("observed_fit_score") or {}).get("rms_error_px")
                        or (sphere_projection.get("observed_fit_score") or {}).get("rms_error_px")
                    ),
                    "sphere_after_optimization_rms_error_px": (
                        (sphere_projection.get("observed_fit_score") or {}).get("rms_error_px")
                    ),
                    "object_evidence_grade": object_grade.get("level"),
                    "agreement_status": object_grade.get("agreement_status"),
                    "radial_point_count": object_grade.get("radial_point_count"),
                    "mask_point_count": object_grade.get("mask_point_count"),
                    "duplicate_warning_false_positive_proxy": "duplicate_detection"
                    in (ball.get("warnings") or []),
                    "warnings": ",".join(ball.get("warnings") or []),
                }
            )
    return rows


def summarize_model_scoring_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    images = {row["image"] for row in rows}
    legacy_values = [
        row["legacy_confidence"] for row in rows if row["legacy_confidence"] is not None
    ]
    physics_values = [
        row["physics_confidence"] for row in rows if row["physics_confidence"] is not None
    ]
    c_only_values = [
        row["physics_c_only_confidence"]
        for row in rows
        if row["physics_c_only_confidence"] is not None
    ]
    displayed_values = [
        row["displayed_confidence"]
        for row in rows
        if row["displayed_confidence"] is not None
    ]
    improved = sum(
        1
        for row in rows
        if row["displayed_delta_vs_legacy"] is not None
        and float(row["displayed_delta_vs_legacy"]) >= 0.10
    )
    reduced = sum(
        1
        for row in rows
        if row["displayed_delta_vs_legacy"] is not None
        and float(row["displayed_delta_vs_legacy"]) <= -0.10
    )
    return {
        "image_count": len(images),
        "ball_count": len(rows),
        "legacy_mean_confidence": mean(legacy_values) if legacy_values else 0.0,
        "physics_mean_confidence": mean(physics_values) if physics_values else 0.0,
        "physics_c_only_mean_confidence": mean(c_only_values) if c_only_values else 0.0,
        "displayed_mean_confidence": mean(displayed_values) if displayed_values else 0.0,
        "improved_by_10_points": improved,
        "reduced_by_10_points": reduced,
        "legacy_status_counts": dict(Counter(row["legacy_status"] for row in rows)),
        "physics_status_counts": dict(Counter(row["physics_status"] for row in rows)),
        "physics_c_only_status_counts": dict(
            Counter(row["physics_c_only_status"] for row in rows),
        ),
        "sphere_grade_counts": dict(Counter(row["sphere_grade"] for row in rows)),
        "observed_ellipse_sphere_grade_counts": dict(
            Counter(row["observed_ellipse_sphere_grade"] for row in rows),
        ),
        "observed_ellipse_grade_counts": dict(
            Counter(row["observed_ellipse_grade"] for row in rows),
        ),
        "object_grade_counts": dict(
            Counter(row["object_evidence_grade"] for row in rows),
        ),
        "mean_accepted_boundary_points": mean(
            row["accepted_boundary_point_count"] for row in rows
        ),
        "mean_rejected_boundary_points": mean(
            row["rejected_boundary_point_count"] for row in rows
        ),
        "evidence_map_status_counts": dict(
            Counter(row["evidence_map_status"] for row in rows),
        ),
        "mean_evidence_map_asset_count": mean(
            row["evidence_map_asset_count"] for row in rows
        ),
        "physical_optimization_status_counts": dict(
            Counter(row["physical_optimization_status"] for row in rows),
        ),
        "physical_projection_mode_counts": dict(
            Counter(row["physical_projection_mode"] for row in rows),
        ),
        "joint_cluster_status_counts": dict(
            Counter(row["joint_cluster_status"] or "not_in_cluster" for row in rows),
        ),
        "mean_joint_cluster_improvement_mm": _mean_optional(
            row["joint_cluster_improvement_mm"] for row in rows
        ),
        "duplicate_warning_false_positive_proxy_count": sum(
            1 for row in rows if row["duplicate_warning_false_positive_proxy"]
        ),
    }


def write_model_scoring_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def _mean_optional(values: Any) -> float:
    numeric = [float(value) for value in values if value is not None]
    return mean(numeric) if numeric else 0.0


__all__ = [
    "ConfidenceSummary",
    "benchmark_model_scoring_command",
    "collect_model_scoring_rows",
    "summarize_model_scoring_rows",
    "summarize_v1_confidence",
    "write_model_scoring_csv",
]

