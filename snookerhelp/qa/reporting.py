from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from snookerhelp.core.config import PROJECT_ROOT, resolve_path
from snookerhelp.qa.report_html import PANEL_FILES, write_report_html
from snookerhelp.qa.report_metrics import (
    build_ball_review_rows,
    build_coordinate_rows,
    build_physical_validation,
    choose_zoom_balls,
    load_scenario,
    select_report_ball,
    summarize_detection,
)
from snookerhelp.qa.report_views import (
    error_comparison_panel,
    geometry_panel,
    physical_validation_panel,
    pipeline_summary_panel,
    source_detection_panel,
    source_zoom_tile_panel,
    source_zoom_grid_panel,
    warped_detection_panel,
    write_panel,
)
from snookerhelp.review.evidence_builder import build_review_evidence
from snookerhelp.recognition.estimator import StateEstimator


def generate_image_report(
    image_path: str | Path,
    output_root: str | Path = "outputs/reports",
    config_path: str | Path = "configs/sony_dev.yaml",
    scenario_path: str | Path | None = None,
    selected_ball: str = "auto",
    ball_id: int | None = None,
) -> tuple[dict[str, Any], Path]:
    source_path = resolve_path(image_path)
    output_directory = resolve_path(output_root) / source_path.stem
    output_directory.mkdir(parents=True, exist_ok=True)

    estimator = StateEstimator.from_config(config_path)
    frame = estimator.process(source_path)
    state = frame.state
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")
    warped_image = estimator.table_warp.warp_image(source_image)

    scenario = load_scenario(scenario_path)
    selected_id = ball_id
    if selected_ball != "auto" and selected_id is None:
        selected_id = int(selected_ball)
    if state.get("balls"):
        selected = select_report_ball(state, selected_id)
        zoom_balls = choose_zoom_balls(state, selected, limit=None)
    else:
        selected = None
        zoom_balls = []
    validation = build_physical_validation(
        state,
        scenario,
        center_mode="source_refined",
    )
    coordinate_rows = build_coordinate_rows(state)
    summary = summarize_detection(state)

    panel_images = {
        "01_source_detection.png": source_detection_panel(source_image, state),
        "02_warped_detection.png": warped_detection_panel(warped_image, state),
        "03_source_zoom_grid.png": source_zoom_grid_panel(source_image, state, zoom_balls),
        "04_geometry_selected_ball.png": geometry_panel(source_image, state, selected),
        "05_error_comparison.png": error_comparison_panel(state),
        "06_physical_validation.png": physical_validation_panel(state, validation),
        "07_pipeline_summary.png": pipeline_summary_panel(state, validation),
    }
    for file_name, _ in PANEL_FILES:
        write_panel(output_directory / file_name, panel_images[file_name])
    zoom_directory = output_directory / "zoom_tiles"
    for ball in state.get("balls", []):
        write_panel(
            zoom_directory / f"ball_{int(ball['id']):02d}.png",
            source_zoom_tile_panel(source_image, state, ball),
        )
    review_evidence = build_review_evidence(
        state=state,
        source_image=source_image,
        warped_image=warped_image,
        output_directory=output_directory,
        evidence_map_settings=estimator.detector.config.get("evidence_maps", {}),
    )

    report = {
        "image": _display_path(source_path),
        "output_directory": str(output_directory),
        "scenario": _display_path(resolve_path(scenario_path)) if scenario_path else None,
        "summary": summary,
        "camera_model": state.get("camera_model", {}),
        "selected_ball": _selected_ball_summary(selected),
        "zoom_ball_ids": [int(ball["id"]) for ball in zoom_balls],
        "ball_review_rows": build_ball_review_rows(state),
        "coordinate_rows": coordinate_rows,
        "physical_validation": validation,
        "review_evidence": review_evidence,
        "panels": [file_name for file_name, _ in PANEL_FILES],
        "state": state,
    }
    with (output_directory / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_report_html(output_directory / "report.html", report)
    return report, output_directory


def _selected_ball_summary(ball: dict[str, Any] | None) -> dict[str, Any]:
    if ball is None:
        return {
            "id": None,
            "label": None,
            "source_refinement_success": None,
            "source_fit_residual_px": None,
            "table_xy_mm": None,
            "source_refined_table_xy_mm": None,
        }
    return {
        "id": int(ball["id"]),
        "label": ball.get("color_label", ball.get("class")),
        "source_refinement_success": ball.get("source_refinement_success"),
        "source_fit_residual_px": ball.get("source_fit_residual_px"),
        "table_xy_mm": ball.get("table_xy_mm"),
        "source_refined_table_xy_mm": ball.get("source_refined_table_xy_mm"),
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

