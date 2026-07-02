from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CharucoBoardSpec:
    name: str
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    dictionary: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CharucoBoardSpec":
        board = config.get("board", config)
        return cls(
            name=str(board.get("name", "charuco_board")),
            squares_x=int(board["squares_x"]),
            squares_y=int(board["squares_y"]),
            square_length_mm=float(board["square_length_mm"]),
            marker_length_mm=float(board["marker_length_mm"]),
            dictionary=str(board.get("dictionary", "DICT_5X5_1000")),
        )


@dataclass(frozen=True)
class CharucoDetection:
    image_path: str
    image_size_px: tuple[int, int]
    marker_count: int
    charuco_corner_count: int
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray | None

    @property
    def usable(self) -> bool:
        return (
            self.charuco_corners is not None
            and self.charuco_ids is not None
            and int(len(self.charuco_ids)) > 0
        )


def require_aruco() -> Any:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "OpenCV ArUco/ChArUco support is unavailable. Install opencv-contrib-python."
        )
    required = [
        "CharucoBoard",
        "calibrateCameraCharuco",
        "estimatePoseCharucoBoard",
        "getPredefinedDictionary",
        "interpolateCornersCharuco",
    ]
    missing = [name for name in required if not hasattr(aruco, name)]
    if missing:
        raise RuntimeError(
            "OpenCV ArUco/ChArUco support is incomplete. Missing: "
            + ", ".join(missing)
        )
    return aruco


def dictionary_from_name(dictionary_name: str) -> Any:
    aruco = require_aruco()
    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    return aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))


def create_charuco_board(spec: CharucoBoardSpec) -> Any:
    aruco = require_aruco()
    dictionary = dictionary_from_name(spec.dictionary)
    return aruco.CharucoBoard(
        (int(spec.squares_x), int(spec.squares_y)),
        float(spec.square_length_mm),
        float(spec.marker_length_mm),
        dictionary,
    )


def detect_charuco_image(
    image_path: str | Path,
    board: Any,
    spec: CharucoBoardSpec,
    *,
    subpixel_refinement: bool = True,
) -> CharucoDetection:
    aruco = require_aruco()
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary_from_name(spec.dictionary), parameters)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    marker_tuple = tuple(marker_corners or ())
    if marker_ids is None or len(marker_tuple) == 0:
        return CharucoDetection(
            image_path=str(path),
            image_size_px=(int(image.shape[1]), int(image.shape[0])),
            marker_count=0,
            charuco_corner_count=0,
            charuco_corners=None,
            charuco_ids=None,
            marker_corners=(),
            marker_ids=None,
        )
    if subpixel_refinement:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        )
        for corners in marker_tuple:
            cv2.cornerSubPix(
                gray,
                corners,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=criteria,
            )
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        list(marker_tuple),
        marker_ids,
        gray,
        board,
    )
    count = 0 if charuco_ids is None else int(len(charuco_ids))
    return CharucoDetection(
        image_path=str(path),
        image_size_px=(int(image.shape[1]), int(image.shape[0])),
        marker_count=int(len(marker_tuple)),
        charuco_corner_count=count,
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        marker_corners=marker_tuple,
        marker_ids=marker_ids,
    )


def calibrate_intrinsics_from_detections(
    detections: list[CharucoDetection],
    board: Any,
    *,
    minimum_corners: int = 24,
    flags: int = 0,
) -> dict[str, Any]:
    usable = [
        detection
        for detection in detections
        if detection.usable
        and detection.charuco_corner_count >= int(minimum_corners)
    ]
    if not usable:
        raise ValueError(
            f"No usable ChArUco detections with at least {minimum_corners} corners"
        )
    image_sizes = {detection.image_size_px for detection in usable}
    if len(image_sizes) != 1:
        raise ValueError(f"All calibration images must share size, got: {image_sizes}")
    image_size = usable[0].image_size_px
    all_corners = [detection.charuco_corners for detection in usable]
    all_ids = [detection.charuco_ids for detection in usable]
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        None,
        None,
        flags=int(flags),
    )
    per_frame = []
    for detection, rvec, tvec in zip(usable, rvecs, tvecs):
        per_frame.append(
            {
                "image": detection.image_path,
                "marker_count": detection.marker_count,
                "charuco_corner_count": detection.charuco_corner_count,
                "rotation_vector_board_to_camera": _round_array(rvec.reshape(3)),
                "translation_board_to_camera_mm": _round_array(tvec.reshape(3)),
            }
        )
    return {
        "image_size_px": [int(image_size[0]), int(image_size[1])],
        "rms_reprojection_error_px": float(rms),
        "camera_matrix": _round_matrix(camera_matrix),
        "distortion_coefficients": _round_array(distortion.reshape(-1)),
        "usable_frame_count": len(usable),
        "frames": per_frame,
    }


def calibration_flags_from_config(config: dict[str, Any]) -> int:
    flags_config = (config.get("calibration") or {}).get("flags", {})
    flags = 0
    if bool(flags_config.get("fix_aspect_ratio", False)):
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if bool(flags_config.get("zero_tangent_dist", False)):
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if bool(flags_config.get("rational_model", False)):
        flags |= cv2.CALIB_RATIONAL_MODEL
    return int(flags)


def estimate_board_pose(
    detection: CharucoDetection,
    board: Any,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> dict[str, Any]:
    if not detection.usable:
        raise ValueError(f"No ChArUco corners detected in {detection.image_path}")
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        detection.charuco_corners,
        detection.charuco_ids,
        board,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion_coefficients, dtype=np.float64),
        rvec,
        tvec,
    )
    if not ok:
        raise RuntimeError(f"Could not estimate ChArUco pose: {detection.image_path}")
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "rotation_board_to_camera": rotation,
        "translation_board_to_camera": tvec.reshape(3),
        "rotation_vector_board_to_camera": rvec.reshape(3),
    }


def board_to_world_rotation(
    x_axis_world: list[float] | tuple[float, float, float],
    y_axis_world: list[float] | tuple[float, float, float],
) -> np.ndarray:
    x_axis = _unit(np.asarray(x_axis_world, dtype=np.float64))
    y_raw = np.asarray(y_axis_world, dtype=np.float64)
    y_axis = _unit(y_raw - x_axis * float(np.dot(x_axis, y_raw)))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def compose_world_to_camera_from_board_pose(
    rotation_board_to_camera: np.ndarray,
    translation_board_to_camera: np.ndarray,
    board_origin_world_mm: list[float] | tuple[float, float, float],
    board_x_axis_world: list[float] | tuple[float, float, float],
    board_y_axis_world: list[float] | tuple[float, float, float],
) -> dict[str, Any]:
    rotation_board_to_world = board_to_world_rotation(
        board_x_axis_world,
        board_y_axis_world,
    )
    translation_board_to_world = np.asarray(
        board_origin_world_mm,
        dtype=np.float64,
    ).reshape(3)
    rotation_world_to_board = rotation_board_to_world.T
    rotation_world_to_camera = (
        np.asarray(rotation_board_to_camera, dtype=np.float64).reshape(3, 3)
        @ rotation_world_to_board
    )
    translation_world_to_camera = (
        np.asarray(translation_board_to_camera, dtype=np.float64).reshape(3)
        - rotation_world_to_camera @ translation_board_to_world
    )
    rotation_vector_world_to_camera, _ = cv2.Rodrigues(rotation_world_to_camera)
    return {
        "rotation_world_to_camera": _round_matrix(rotation_world_to_camera),
        "rotation_vector_world_to_camera": _round_array(
            rotation_vector_world_to_camera.reshape(3)
        ),
        "translation_world_to_camera": _round_array(translation_world_to_camera),
        "board_origin_world_mm": _round_array(translation_board_to_world),
        "board_x_axis_world": _round_array(board_to_world_rotation(
            board_x_axis_world,
            board_y_axis_world,
        )[:, 0]),
        "board_y_axis_world": _round_array(board_to_world_rotation(
            board_x_axis_world,
            board_y_axis_world,
        )[:, 1]),
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Axis vector must be non-zero")
    return vector / norm


def _round_array(values: np.ndarray) -> list[float]:
    return [round(float(value), 10) for value in np.asarray(values).reshape(-1)]


def _round_matrix(values: np.ndarray) -> list[list[float]]:
    matrix = np.asarray(values, dtype=np.float64)
    return [[round(float(value), 10) for value in row] for row in matrix]
