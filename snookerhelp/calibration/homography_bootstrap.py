from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from snookerhelp.core.table import TableModel
from snookerhelp.core.config import save_yaml


POINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]


@dataclass(frozen=True)
class TableWarp:
    table: TableModel
    corner_points_px: np.ndarray
    homography: np.ndarray
    inverse_homography: np.ndarray

    @classmethod
    def from_corners(
        cls,
        table: TableModel,
        corners: Iterable[Iterable[float]],
    ) -> "TableWarp":
        source = np.asarray(list(corners), dtype=np.float32)
        if source.shape != (4, 2):
            raise ValueError("Table calibration requires four [x, y] points")
        width = table.warp_width_px
        height = table.warp_height_px
        margin = table.processing_margin_px
        destination = np.float32(
            [
                [margin, margin],
                [width - margin - 1, margin],
                [width - margin - 1, height - margin - 1],
                [margin, height - margin - 1],
            ]
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        return cls(
            table=table,
            corner_points_px=source,
            homography=homography,
            inverse_homography=np.linalg.inv(homography),
        )

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            image,
            self.homography,
            (self.table.warp_width_px, self.table.warp_height_px),
            flags=cv2.INTER_AREA,
        )

    def warped_to_source(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(values, self.inverse_homography).reshape(-1, 2)

    def source_to_warped(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(values, self.homography).reshape(-1, 2)

    def warped_px_to_table_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        margin = self.table.processing_margin_px
        x_mm = (x_px - margin) / self.table.px_per_mm
        y_from_top_mm = (y_px - margin) / self.table.px_per_mm
        if self.table.origin.startswith("bottom_left"):
            return x_mm, self.table.width_mm - y_from_top_mm
        return x_mm, y_from_top_mm

    def table_mm_to_warped_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        margin = self.table.processing_margin_px
        x_px = x_mm * self.table.px_per_mm + margin
        if self.table.origin.startswith("bottom_left"):
            y_from_top_mm = self.table.width_mm - y_mm
        else:
            y_from_top_mm = y_mm
        y_px = y_from_top_mm * self.table.px_per_mm + margin
        return x_px, y_px


def click_table_corners_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Click table playing corners")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="configs/table_manual.yaml")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=1000)
    args = parser.parse_args(argv)

    return click_table_corners(
        image_path=args.image,
        output_path=args.output,
        max_width=args.max_width,
        max_height=args.max_height,
    )


def click_table_corners(
    *,
    image_path: str,
    output_path: str = "configs/table_manual.yaml",
    max_width: int = 1600,
    max_height: int = 1000,
) -> int:
    source = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = source.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    preview = cv2.resize(
        source,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    points: list[tuple[int, int]] = []
    window = "Table calibration: TL, TR, BR, BL"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = preview.copy()
        for index, point in enumerate(points):
            cv2.circle(display, point, 8, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                display,
                str(index + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                str(index + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) > 1:
            cv2.polylines(
                display,
                [np.array(points, dtype=np.int32)],
                len(points) == 4,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        next_name = POINT_NAMES[len(points)] if len(points) < 4 else "press Enter"
        cv2.putText(
            display,
            f"Next: {next_name} | Backspace undo | R reset | Esc cancel",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"Next: {next_name} | Backspace undo | R reset | Esc cancel",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return 1
        if key in (8, 127) and points:
            points.pop()
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key in (10, 13) and len(points) == 4:
            break

    cv2.destroyAllWindows()
    full_resolution_points = [
        [round(x / scale, 3), round(y / scale, 3)] for x, y in points
    ]
    save_yaml(
        output_path,
        {
            "source_image": image_path,
            "image_width": width,
            "image_height": height,
            "point_order": POINT_NAMES,
            "corner_points_px": full_resolution_points,
        },
    )
    print(f"Saved calibration to {output_path}")
    return 0


__all__ = [
    "POINT_NAMES",
    "TableWarp",
    "click_table_corners",
    "click_table_corners_command",
]

