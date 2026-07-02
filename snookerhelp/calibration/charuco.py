from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from snookerhelp.calibration.charuco_core import (
    CharucoBoardSpec,
    CharucoDetection,
    calibrate_intrinsics_from_detections,
    calibration_flags_from_config,
    compose_world_to_camera_from_board_pose,
    create_charuco_board,
    detect_charuco_image,
    estimate_board_pose,
)
from snookerhelp.core.config import load_yaml, resolve_path, save_yaml


def calibrate_intrinsics_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from ChArUco board images"
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        help="Image paths or glob patterns, e.g. calibration/charuco/*.JPG",
    )
    parser.add_argument(
        "--board",
        default="configs/charuco_calitar_cali100020tar5.yaml",
        help="ChArUco board YAML",
    )
    parser.add_argument(
        "--output",
        default="configs/camera_intrinsics_charuco.yaml",
        help="Output intrinsics YAML",
    )
    parser.add_argument("--minimum-corners", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args(argv)

    board_config = load_yaml(args.board)
    spec = CharucoBoardSpec.from_config(board_config)
    board = create_charuco_board(spec)
    image_paths = expand_images(args.images)
    if not image_paths:
        raise FileNotFoundError(f"No images matched: {args.images}")

    minimum_corners = int(
        args.minimum_corners
        if args.minimum_corners is not None
        else (board_config.get("calibration") or {}).get("minimum_charuco_corners", 24)
    )
    subpixel = bool(
        (board_config.get("calibration") or {}).get("subpixel_refinement", True)
    )
    detections = [
        detect_charuco_image(path, board, spec, subpixel_refinement=subpixel)
        for path in image_paths
    ]
    calibration = calibrate_intrinsics_from_detections(
        detections,
        board,
        minimum_corners=minimum_corners,
        flags=calibration_flags_from_config(board_config),
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "calibration_type": "charuco_intrinsics",
        "board": {
            "name": spec.name,
            "squares_x": spec.squares_x,
            "squares_y": spec.squares_y,
            "square_length_mm": spec.square_length_mm,
            "marker_length_mm": spec.marker_length_mm,
            "dictionary": spec.dictionary,
        },
        "camera": calibration,
        "all_detections": [
            {
                "image": detection.image_path,
                "image_size_px": list(detection.image_size_px),
                "marker_count": detection.marker_count,
                "charuco_corner_count": detection.charuco_corner_count,
                "usable": bool(
                    detection.usable
                    and detection.charuco_corner_count >= minimum_corners
                ),
            }
            for detection in detections
        ],
        "next_step": (
            "Use tools/estimate_table_pose_charuco.py with this intrinsics file "
            "after placing the board on the cloth at a measured table position."
        ),
    }
    save_yaml(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            "Saved ChArUco intrinsics: "
            f"{resolve_path(args.output)} "
            f"(usable frames: {calibration['usable_frame_count']}, "
            f"RMS: {calibration['rms_reprojection_error_px']:.4f} px)"
        )
    return 0


def estimate_table_pose_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate calibrated camera pose in table/world coordinates from one "
            "ChArUco image and a measured board placement on the cloth"
        )
    )
    parser.add_argument("--image", required=True, help="ChArUco image path")
    parser.add_argument(
        "--intrinsics",
        default="configs/camera_intrinsics_charuco.yaml",
        help="Output from tools/calibrate_camera_charuco.py",
    )
    parser.add_argument(
        "--board",
        default="configs/charuco_calitar_cali100020tar5.yaml",
        help="ChArUco board YAML",
    )
    parser.add_argument(
        "--board-origin-table-mm",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help=(
            "Board coordinate origin in table/world mm. For a board lying on "
            "the cloth, Z is usually 0."
        ),
    )
    parser.add_argument(
        "--board-x-axis-table",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Unit direction of board +X in table/world coordinates",
    )
    parser.add_argument(
        "--board-y-axis-table",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Unit direction of board +Y in table/world coordinates",
    )
    parser.add_argument(
        "--output",
        default="configs/camera_model_charuco_table.yaml",
        help="Output camera model YAML usable from configs/sony_dev.yaml",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args(argv)

    board_config = load_yaml(args.board)
    intrinsics = load_yaml(args.intrinsics)
    camera = intrinsics["camera"]
    spec = CharucoBoardSpec.from_config(board_config)
    board = create_charuco_board(spec)
    detection = detect_charuco_image(args.image, board, spec)
    pose = estimate_board_pose(
        detection,
        board,
        np.asarray(camera["camera_matrix"], dtype=np.float64),
        np.asarray(camera["distortion_coefficients"], dtype=np.float64),
    )
    world_pose = compose_world_to_camera_from_board_pose(
        pose["rotation_board_to_camera"],
        pose["translation_board_to_camera"],
        args.board_origin_table_mm,
        args.board_x_axis_table,
        args.board_y_axis_table,
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "camera_model": {
            "mode": "calibrated_pinhole",
            "camera_matrix": camera["camera_matrix"],
            "distortion_coefficients": camera["distortion_coefficients"],
            "rotation_world_to_camera": world_pose["rotation_world_to_camera"],
            "rotation_vector_world_to_camera": world_pose[
                "rotation_vector_world_to_camera"
            ],
            "translation_world_to_camera": world_pose[
                "translation_world_to_camera"
            ],
            "projection_z_planes_mm": [0.0, 13.1, 26.25, 39.4, 52.5],
        },
        "calibration_metadata": {
            "intrinsics_file": str(resolve_path(args.intrinsics)),
            "pose_image": str(resolve_path(args.image)),
            "board": {
                "name": spec.name,
                "squares_x": spec.squares_x,
                "squares_y": spec.squares_y,
                "square_length_mm": spec.square_length_mm,
                "marker_length_mm": spec.marker_length_mm,
                "dictionary": spec.dictionary,
            },
            "detection": {
                "marker_count": detection.marker_count,
                "charuco_corner_count": detection.charuco_corner_count,
            },
            "board_pose": {
                "rotation_vector_board_to_camera": [
                    round(float(value), 10)
                    for value in pose["rotation_vector_board_to_camera"]
                ],
                "translation_board_to_camera_mm": [
                    round(float(value), 10)
                    for value in pose["translation_board_to_camera"]
                ],
                "board_origin_table_mm": world_pose["board_origin_world_mm"],
                "board_x_axis_table": world_pose["board_x_axis_world"],
                "board_y_axis_table": world_pose["board_y_axis_world"],
            },
        },
        "usage": {
            "sony_dev_yaml": (
                "Set pipeline.camera_model.config_file to this YAML path, "
                "or paste the camera_model mapping into configs/sony_dev.yaml."
            )
        },
    }
    save_yaml(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Saved calibrated table camera model: {resolve_path(args.output)}")
    return 0


def expand_images(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(str(resolve_path(pattern)), recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            candidate = resolve_path(pattern)
            if candidate.exists():
                paths.append(candidate)
    return sorted({path.resolve() for path in paths})

__all__ = [
    "CharucoBoardSpec",
    "CharucoDetection",
    "calibrate_intrinsics_command",
    "calibrate_intrinsics_from_detections",
    "calibration_flags_from_config",
    "compose_world_to_camera_from_board_pose",
    "create_charuco_board",
    "detect_charuco_image",
    "estimate_table_pose_command",
    "estimate_board_pose",
    "expand_images",
]

