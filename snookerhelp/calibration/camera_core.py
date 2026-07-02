from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from snookerhelp.calibration.homography_bootstrap import TableWarp


DEFAULT_PROJECTION_Z_PLANES_MM = (0.0, 13.1, 26.25, 39.4, 52.5)


class CameraModel(Protocol):
    """Image/table geometry interface.

    Implementations map source-image pixels to world/table coordinates in the
    project coordinate system:

    - X/Y are table coordinates in millimeters.
    - Z is height above the cloth plane in millimeters.
    - The cloth plane is Z=0.
    """

    model_name: str
    is_calibrated: bool

    def undistort_points(self, points_px: np.ndarray) -> np.ndarray:
        ...

    def image_point_to_world_plane(
        self,
        point_px: np.ndarray | tuple[float, float] | list[float],
        z_mm: float,
    ) -> np.ndarray:
        ...

    def world_point_to_image(
        self,
        point_xyz_mm: np.ndarray | tuple[float, float, float] | list[float],
    ) -> np.ndarray:
        ...

    def project_image_point_to_z_planes(
        self,
        point_px: np.ndarray | tuple[float, float] | list[float],
        z_planes_mm: list[float] | tuple[float, ...],
    ) -> dict[str, dict[str, Any]]:
        ...


@dataclass(frozen=True)
class HomographyCameraModel:
    """Manual homography-backed approximation.

    This mode uses the existing table homography for XY and deliberately cannot
    model height parallax. All Z planes therefore return the same XY. Keeping
    this behavior explicit is useful: reports can distinguish "schema is wired"
    from "real camera calibration is available".
    """

    table_warp: TableWarp
    model_name: str = "manual_homography"
    is_calibrated: bool = False

    def undistort_points(self, points_px: np.ndarray) -> np.ndarray:
        return np.asarray(points_px, dtype=np.float32).copy()

    def image_point_to_world_plane(
        self,
        point_px: np.ndarray | tuple[float, float] | list[float],
        z_mm: float,
    ) -> np.ndarray:
        point = np.asarray(point_px, dtype=np.float32).reshape(1, 2)
        undistorted = self.undistort_points(point)
        warped = self.table_warp.source_to_warped(undistorted)[0]
        x_mm, y_mm = self.table_warp.warped_px_to_table_mm(
            float(warped[0]), float(warped[1])
        )
        return np.array([x_mm, y_mm, float(z_mm)], dtype=np.float64)

    def world_point_to_image(
        self,
        point_xyz_mm: np.ndarray | tuple[float, float, float] | list[float],
    ) -> np.ndarray:
        point = np.asarray(point_xyz_mm, dtype=np.float64).reshape(-1)
        if point.size < 2:
            raise ValueError("world_point_to_image requires at least x_mm and y_mm")
        warped = self.table_warp.table_mm_to_warped_px(float(point[0]), float(point[1]))
        source = self.table_warp.warped_to_source(np.float32([warped]))[0]
        return source.astype(np.float64)

    def project_image_point_to_z_planes(
        self,
        point_px: np.ndarray | tuple[float, float] | list[float],
        z_planes_mm: list[float] | tuple[float, ...],
    ) -> dict[str, dict[str, Any]]:
        return {
            z_plane_key(z_mm): _projection_payload(
                self.image_point_to_world_plane(point_px, z_mm),
                approximate=True,
                model_name=self.model_name,
            )
            for z_mm in z_planes_mm
        }


@dataclass(frozen=True)
class PinholeCameraModel:
    """Calibrated pinhole camera with optional distortion.

    `rotation_world_to_camera` and `translation_world_to_camera` define:

    ```
    X_camera = R_world_to_camera @ X_world + t_world_to_camera
    ```

    Image-to-world projection undistorts an image point to a camera ray, then
    intersects that ray with the requested constant-Z table plane.
    """

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    model_name: str = "calibrated_pinhole"
    is_calibrated: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_matrix",
            np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "distortion_coefficients",
            _distortion_array(self.distortion_coefficients),
        )
        object.__setattr__(
            self,
            "rotation_world_to_camera",
            np.asarray(self.rotation_world_to_camera, dtype=np.float64).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "translation_world_to_camera",
            np.asarray(self.translation_world_to_camera, dtype=np.float64).reshape(3),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PinholeCameraModel":
        if "rotation_vector_world_to_camera" in config:
            rotation, _ = cv2.Rodrigues(
                np.asarray(config["rotation_vector_world_to_camera"], dtype=np.float64)
            )
        else:
            rotation = np.asarray(
                config["rotation_world_to_camera"],
                dtype=np.float64,
            )
        return cls(
            camera_matrix=np.asarray(config["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(
                config.get("distortion_coefficients", []),
                dtype=np.float64,
            ),
            rotation_world_to_camera=rotation,
            translation_world_to_camera=np.asarray(
                config["translation_world_to_camera"],
                dtype=np.float64,
            ),
        )

    @classmethod
    def from_table_corners(
        cls,
        table_warp: TableWarp,
        config: dict[str, Any],
    ) -> "PinholeCameraModel":
        """Build an approximate pinhole model from manual table corners.

        This is not a replacement for ChArUco calibration. It is useful while
        waiting for the board because it gives the review UI a physically
        meaningful first-order model:

        - intrinsics come from approximate lens/sensor/image metadata;
        - extrinsics are solved from the four manually clicked table corners.
        """
        camera_matrix = _camera_matrix_from_config(config)
        distortion = _distortion_array(
            np.asarray(config.get("distortion_coefficients", []), dtype=np.float64)
        )
        object_points = _table_corner_world_points(table_warp)
        image_points = np.asarray(table_warp.corner_points_px, dtype=np.float64)
        flag_name = str(config.get("solve_pnp_flag", "ITERATIVE")).upper()
        flag = _solve_pnp_flag(flag_name)
        ok, rotation_vector, translation = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=flag,
        )
        if not ok:
            raise RuntimeError("Could not estimate approximate camera pose from table corners")
        rotation, _ = cv2.Rodrigues(rotation_vector)
        model = cls(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            rotation_world_to_camera=rotation,
            translation_world_to_camera=translation.reshape(3),
            model_name="approximate_pinhole_from_corners",
            is_calibrated=False,
        )
        min_height_mm = float(config.get("minimum_camera_height_mm", 100.0))
        if float(model.camera_center_world_mm[2]) < min_height_mm:
            raise RuntimeError(
                "Approximate camera pose is invalid: camera center is not above "
                f"the table by at least {min_height_mm:g} mm"
            )
        return model

    @property
    def camera_center_world_mm(self) -> np.ndarray:
        return -self.rotation_world_to_camera.T @ self.translation_world_to_camera

    def undistort_points(self, points_px: np.ndarray) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(
            points,
            self.camera_matrix,
            self.distortion_coefficients,
            P=self.camera_matrix,
        )
        return undistorted.reshape(-1, 2)

    def image_point_to_world_plane(
        self,
        point_px: np.ndarray | tuple[float, float] | list[float],
        z_mm: float,
    ) -> np.ndarray:
        point = np.asarray(point_px, dtype=np.float64).reshape(1, 1, 2)
        normalized = cv2.undistortPoints(
            point,
            self.camera_matrix,
            self.distortion_coefficients,
        ).reshape(2)
        ray_camera = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
        ray_world = self.rotation_world_to_camera.T @ ray_camera
        camera_center = self.camera_center_world_mm
        if abs(ray_world[2]) < 1e-9:
            raise ValueError("Image ray is parallel to requested Z plane")
        scale = (float(z_mm) - camera_center[2]) / ray_world[2]
        world = camera_center + scale * ray_world
        return np.array([world[0], world[1], float(z_mm)], dtype=np.float64)

    def world_point_to_image(
        self,
        point_xyz_mm: np.ndarray | tuple[float, float, float] | list[float],
    ) -> np.ndarray:
        point = np.asarray(point_xyz_mm, dtype=np.float64).reshape(1, 1, 3)
        rotation_vector, _ = cv2.Rodrigues(self.rotation_world_to_camera)
        image_points, _ = cv2.projectPoints(
            point,
            rotation_vector,
            self.translation_world_to_camera.reshape(3, 1),
            self.camera_matrix,
            self.distortion_coefficients,
        )
        return image_points.reshape(2).astype(np.float64)

    def project_image_point_to_z_planes(
        self,
        point_px: np.ndarray | tuple[float, float] | list[float],
        z_planes_mm: list[float] | tuple[float, ...],
    ) -> dict[str, dict[str, Any]]:
        return {
            z_plane_key(z_mm): _projection_payload(
                self.image_point_to_world_plane(point_px, z_mm),
                approximate=not bool(self.is_calibrated),
                model_name=self.model_name,
            )
            for z_mm in z_planes_mm
        }


def build_camera_model(
    table_warp: TableWarp,
    config: dict[str, Any] | None = None,
) -> CameraModel:
    settings = config or {}
    mode = str(settings.get("mode", "manual_homography"))
    if mode in {"manual_homography", "homography"}:
        return HomographyCameraModel(table_warp=table_warp)
    if mode in {"calibrated_pinhole", "pinhole"}:
        return PinholeCameraModel.from_config(settings)
    if mode in {
        "approximate_pinhole",
        "approximate_pinhole_from_corners",
        "pinhole_from_corners",
    }:
        return PinholeCameraModel.from_table_corners(table_warp, settings)
    raise ValueError(
        "camera model mode must be manual_homography, calibrated_pinhole, "
        "or approximate_pinhole_from_corners"
    )


def z_plane_key(z_mm: float) -> str:
    value = f"{float(z_mm):.2f}".replace("-", "m").replace(".", "_")
    return f"z_{value}"


def z_center_method(z_mm: float) -> str:
    return f"source_{z_plane_key(z_mm)}"


def parse_z_center_method(center_method: str) -> float | None:
    if not center_method.startswith("source_z_"):
        return None
    value = center_method[len("source_z_") :].replace("m", "-").replace("_", ".")
    try:
        return float(value)
    except ValueError:
        return None


def configured_z_planes(config: dict[str, Any] | None = None) -> list[float]:
    if not config:
        return list(DEFAULT_PROJECTION_Z_PLANES_MM)
    values = config.get("projection_z_planes_mm", DEFAULT_PROJECTION_Z_PLANES_MM)
    return [float(value) for value in values]


def _projection_payload(
    world_xyz_mm: np.ndarray,
    approximate: bool,
    model_name: str,
) -> dict[str, Any]:
    return {
        "z_mm": float(world_xyz_mm[2]),
        "xy_mm": [float(world_xyz_mm[0]), float(world_xyz_mm[1])],
        "xyz_mm": [
            float(world_xyz_mm[0]),
            float(world_xyz_mm[1]),
            float(world_xyz_mm[2]),
        ],
        "approximate": bool(approximate),
        "camera_model": model_name,
    }


def _distortion_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if array.size == 0:
        return np.zeros((5, 1), dtype=np.float64)
    return array


def _camera_matrix_from_config(config: dict[str, Any]) -> np.ndarray:
    if "camera_matrix" in config:
        return np.asarray(config["camera_matrix"], dtype=np.float64).reshape(3, 3)
    width_px = float(config.get("image_width_px", config.get("image_width", 0.0)))
    height_px = float(config.get("image_height_px", config.get("image_height", 0.0)))
    if width_px <= 0 or height_px <= 0:
        raise ValueError(
            "approximate pinhole mode requires image_width_px and image_height_px"
        )
    if "focal_length_px" in config:
        focal = config["focal_length_px"]
        if isinstance(focal, (list, tuple)):
            fx_px, fy_px = float(focal[0]), float(focal[1])
        else:
            fx_px = fy_px = float(focal)
    else:
        focal_length_mm = float(config["focal_length_mm"])
        sensor_width_mm = float(config["sensor_width_mm"])
        sensor_height_mm = float(config["sensor_height_mm"])
        if sensor_width_mm <= 0 or sensor_height_mm <= 0:
            raise ValueError("sensor dimensions must be positive")
        fx_px = focal_length_mm / sensor_width_mm * width_px
        fy_px = focal_length_mm / sensor_height_mm * height_px
    principal = config.get("principal_point_px")
    if principal is None:
        cx_px = width_px * 0.5
        cy_px = height_px * 0.5
    else:
        cx_px = float(principal[0])
        cy_px = float(principal[1])
    return np.asarray(
        [
            [fx_px, 0.0, cx_px],
            [0.0, fy_px, cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _table_corner_world_points(table_warp: TableWarp) -> np.ndarray:
    table = table_warp.table
    length = float(table.length_mm)
    width = float(table.width_mm)
    if table.origin.startswith("bottom_left"):
        points = [
            [0.0, width, 0.0],
            [length, width, 0.0],
            [length, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    else:
        points = [
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [length, width, 0.0],
            [0.0, width, 0.0],
        ]
    return np.asarray(points, dtype=np.float64)


def _solve_pnp_flag(flag_name: str) -> int:
    mapping = {
        "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
        "IPPE": cv2.SOLVEPNP_IPPE,
        "IPPE_SQUARE": cv2.SOLVEPNP_IPPE_SQUARE,
        "EPNP": cv2.SOLVEPNP_EPNP,
    }
    if flag_name not in mapping:
        raise ValueError(
            "solve_pnp_flag must be one of: " + ", ".join(sorted(mapping))
        )
    return int(mapping[flag_name])
