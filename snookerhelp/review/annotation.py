from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from snookerhelp.recognition.color import BallColorClassifier
from snookerhelp.core.config import PROJECT_ROOT, resolve_path
from snookerhelp.recognition.estimator import StateEstimator
from snookerhelp.qa.validation import ball_points_from_state, load_json


LABEL_KEYS = {
    ord("1"): "white",
    ord("2"): "red",
    ord("3"): "yellow",
    ord("4"): "green",
    ord("5"): "brown",
    ord("6"): "blue",
    ord("7"): "pink",
    ord("8"): "black",
    ord("9"): "unknown",
}

ARROW_LEFT = 2424832
ARROW_UP = 2490368
ARROW_RIGHT = 2555904
ARROW_DOWN = 2621440


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detector-seeded manual annotation editor for snooker-ball centers"
    )
    parser.add_argument("--image", required=True, help="Source JPEG path")
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--detector-output", default=None)
    parser.add_argument(
        "--no-detector-seed",
        action="store_true",
        help="Start from an empty annotation set instead of current detections.",
    )
    parser.add_argument(
        "--coordinate-system",
        choices=("source_px", "warped_px", "table_mm"),
        default="warped_px",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=1000)
    parser.add_argument("--magnifier-size", type=int, default=90)
    parser.add_argument("--magnifier-scale", type=int, default=4)
    args = parser.parse_args(argv)

    source_path = resolve_path(args.image)
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    estimator = StateEstimator.from_config(args.config)
    if args.coordinate_system == "source_px":
        annotation_image = source_image
    else:
        annotation_image = estimator.table_warp.warp_image(source_image)

    image_height, image_width = annotation_image.shape[:2]
    base_scale = min(
        args.max_width / image_width,
        args.max_height / image_height,
        1.0,
    )
    window_size = (
        max(1, int(round(image_width * base_scale))),
        max(1, int(round(image_height * base_scale))),
    )

    detector_state: dict[str, Any] | None = None
    if not args.no_detector_seed:
        if args.detector_output:
            detector_state = load_json(args.detector_output)
        else:
            detector_state = estimator.process(source_path).state

    points = (
        _seed_points_from_detector(
            detector_state,
            args.coordinate_system,
            estimator,
            args.notes,
        )
        if detector_state is not None
        else []
    )
    selected_label = "red"
    window = "Ball center annotation"
    magnifier_window = "Magnified crop"
    zoom = 1.0
    view_center = [image_width / 2.0, image_height / 2.0]
    cursor_image_xy: tuple[float, float] | None = None

    def view() -> dict[str, float]:
        return _view_geometry(
            image_width=image_width,
            image_height=image_height,
            window_width=window_size[0],
            window_height=window_size[1],
            base_scale=base_scale,
            zoom=zoom,
            center_x=view_center[0],
            center_y=view_center[1],
        )

    def display_to_image(x: int, y: int) -> tuple[float, float]:
        geometry = view()
        return (
            geometry["left"] + x / geometry["scale_x"],
            geometry["top"] + y / geometry["scale_y"],
        )

    def image_to_display(x: float, y: float) -> tuple[int, int]:
        geometry = view()
        return (
            int(round((x - geometry["left"]) * geometry["scale_x"])),
            int(round((y - geometry["top"]) * geometry["scale_y"])),
        )

    def nearest_point_index(image_x: float, image_y: float) -> int | None:
        if not points:
            return None
        display_points = [
            _point_to_image_xy(point, args.coordinate_system, estimator)
            for point in points
        ]
        return int(
            np.argmin(
                [
                    np.hypot(px - image_x, py - image_y)
                    for px, py in display_points
                ]
            )
        )

    def clamp_view_center() -> None:
        view_center[0] = min(max(view_center[0], 0.0), float(image_width))
        view_center[1] = min(max(view_center[1], 0.0), float(image_height))

    def on_mouse(
        event: int, x: int, y: int, flags: int, parameter: object
    ) -> None:
        del parameter
        nonlocal points, zoom, cursor_image_xy
        image_x, image_y = display_to_image(x, y)
        cursor_image_xy = (
            min(max(image_x, 0.0), image_width - 1.0),
            min(max(image_y, 0.0), image_height - 1.0),
        )
        if event == cv2.EVENT_MOUSEMOVE:
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            zoom = _adjust_zoom(zoom, 1.25 if flags > 0 else 0.8)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                nearest = nearest_point_index(image_x, image_y)
                if nearest is not None:
                    points[nearest]["label"] = selected_label
                    points[nearest]["notes"] = args.notes or "label_corrected"
                return
            saved_x, saved_y = _image_xy_to_saved_xy(
                image_x,
                image_y,
                args.coordinate_system,
                estimator,
            )
            point: dict[str, Any] = {
                "id": len(points) + 1,
                "label": selected_label,
                "x": round(saved_x, 4),
                "y": round(saved_y, 4),
            }
            if args.notes:
                point["notes"] = args.notes
            points.append(point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            nearest = nearest_point_index(image_x, image_y)
            if nearest is not None:
                points.pop(nearest)
                _renumber(points)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(magnifier_window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = _render_view(
            annotation_image=annotation_image,
            window_size=window_size,
            geometry=view(),
        )
        for point in points:
            image_x, image_y = _point_to_image_xy(
                point, args.coordinate_system, estimator
            )
            center = image_to_display(image_x, image_y)
            if not _point_is_visible(center, window_size):
                continue
            color = BallColorClassifier.display_bgr(point["label"])
            cv2.drawMarker(
                display,
                center,
                color,
                cv2.MARKER_CROSS,
                18,
                3,
                cv2.LINE_AA,
            )
            _draw_text(
                display,
                f"{point['id']}:{point['label']}",
                (center[0] + 10, center[1] - 10),
                0.55,
            )

        status = (
            f"Selected: {selected_label} | "
            "1 white 2 red 3 yellow 4 green 5 brown "
            "6 blue 7 pink 8 black 9 unknown"
        )
        help_text = (
            "Left add | Shift+left relabel nearest | Right delete | "
            "+/- zoom | arrows/WASD pan | 0 reset"
        )
        save_text = (
            "Shift+A accept current detections | Enter/Shift+S save | "
            "Backspace undo | R reset | Esc cancel"
        )
        _draw_status(display, status, 32)
        _draw_status(display, help_text, 64)
        _draw_status(display, save_text, 96)
        _draw_status(
            display,
            f"{len(points)} annotations | {args.coordinate_system} | zoom {zoom:.2f}x",
            128,
        )
        cv2.imshow(window, display)
        if cursor_image_xy is not None:
            cv2.imshow(
                magnifier_window,
                _magnified_crop(
                    annotation_image,
                    cursor_image_xy,
                    args.magnifier_size,
                    args.magnifier_scale,
                ),
            )

        key = cv2.waitKeyEx(20)
        ascii_key = key & 0xFF
        if key == -1:
            continue
        if ascii_key == 27:
            cv2.destroyAllWindows()
            return 1
        if ascii_key in LABEL_KEYS:
            selected_label = LABEL_KEYS[ascii_key]
        elif ascii_key in (8, 127) and points:
            points.pop()
            _renumber(points)
        elif ascii_key in (ord("r"), ord("R")):
            points.clear()
        elif ascii_key in (ord("+"), ord("=")):
            zoom = _adjust_zoom(zoom, 1.25)
        elif ascii_key in (ord("-"), ord("_")):
            zoom = _adjust_zoom(zoom, 0.8)
        elif ascii_key == ord("0"):
            zoom = 1.0
            view_center = [image_width / 2.0, image_height / 2.0]
        elif ascii_key == ord("A") and points:
            break
        elif ascii_key == ord("S") and points:
            break
        elif key in (ARROW_LEFT, ARROW_RIGHT, ARROW_UP, ARROW_DOWN) or ascii_key in (
            ord("a"),
            ord("d"),
            ord("w"),
            ord("s"),
        ):
            geometry = view()
            step_x = 0.12 * geometry["width"]
            step_y = 0.12 * geometry["height"]
            if key == ARROW_LEFT or ascii_key == ord("a"):
                view_center[0] -= step_x
            elif key == ARROW_RIGHT or ascii_key == ord("d"):
                view_center[0] += step_x
            elif key == ARROW_UP or ascii_key == ord("w"):
                view_center[1] -= step_y
            elif key == ARROW_DOWN or ascii_key == ord("s"):
                view_center[1] += step_y
            clamp_view_center()
        elif ascii_key in (10, 13) and points:
            break

    cv2.destroyAllWindows()
    output_path = (
        resolve_path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "annotations" / f"{source_path.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image_name = str(source_path.relative_to(PROJECT_ROOT))
    except ValueError:
        image_name = str(source_path)
    payload = {
        "version": 2,
        "image_name": image_name,
        "coordinate_system": args.coordinate_system,
        "image_size_px": {
            "width": image_width,
            "height": image_height,
        },
        "seeded_from_detector": detector_state is not None,
        "notes": args.notes,
        "balls": points,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(points)} annotations to {output_path}")
    return 0


def _seed_points_from_detector(
    detector_state: dict[str, Any] | None,
    coordinate_system: str,
    estimator: StateEstimator,
    notes: str | None,
) -> list[dict[str, Any]]:
    if detector_state is None:
        return []
    points: list[dict[str, Any]] = []
    for index, detection in enumerate(ball_points_from_state(detector_state), start=1):
        if coordinate_system == "source_px":
            source_point = estimator.table_warp.warped_to_source(
                np.float32([[detection["x_px"], detection["y_px"]]])
            )[0]
            saved_x, saved_y = float(source_point[0]), float(source_point[1])
        elif coordinate_system == "table_mm":
            saved_x, saved_y = detection["x_mm"], detection["y_mm"]
        else:
            saved_x, saved_y = detection["x_px"], detection["y_px"]
        point: dict[str, Any] = {
            "id": index,
            "label": detection["label"],
            "x": round(saved_x, 4),
            "y": round(saved_y, 4),
            "detection_id": detection["id"],
            "detection_confidence": round(detection["confidence"], 4),
        }
        point["notes"] = notes or "seeded_from_detector"
        points.append(point)
    return points


def _point_to_image_xy(
    point: dict[str, Any],
    coordinate_system: str,
    estimator: StateEstimator,
) -> tuple[float, float]:
    if coordinate_system == "table_mm":
        return estimator.table_warp.table_mm_to_warped_px(
            float(point["x"]), float(point["y"])
        )
    return float(point["x"]), float(point["y"])


def _image_xy_to_saved_xy(
    image_x: float,
    image_y: float,
    coordinate_system: str,
    estimator: StateEstimator,
) -> tuple[float, float]:
    if coordinate_system == "table_mm":
        return estimator.table_warp.warped_px_to_table_mm(image_x, image_y)
    return image_x, image_y


def _view_geometry(
    image_width: int,
    image_height: int,
    window_width: int,
    window_height: int,
    base_scale: float,
    zoom: float,
    center_x: float,
    center_y: float,
) -> dict[str, float]:
    target_scale = max(base_scale * zoom, 1e-6)
    width = min(float(image_width), window_width / target_scale)
    height = min(float(image_height), window_height / target_scale)
    left = min(max(center_x - width / 2.0, 0.0), image_width - width)
    top = min(max(center_y - height / 2.0, 0.0), image_height - height)
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "scale_x": window_width / width,
        "scale_y": window_height / height,
    }


def _render_view(
    annotation_image: np.ndarray,
    window_size: tuple[int, int],
    geometry: dict[str, float],
) -> np.ndarray:
    x0 = int(np.floor(geometry["left"]))
    y0 = int(np.floor(geometry["top"]))
    x1 = int(np.ceil(geometry["left"] + geometry["width"]))
    y1 = int(np.ceil(geometry["top"] + geometry["height"]))
    x1 = min(max(x1, x0 + 1), annotation_image.shape[1])
    y1 = min(max(y1, y0 + 1), annotation_image.shape[0])
    crop = annotation_image[y0:y1, x0:x1]
    return cv2.resize(crop, window_size, interpolation=cv2.INTER_AREA)


def _magnified_crop(
    image: np.ndarray,
    cursor_xy: tuple[float, float],
    crop_size: int,
    scale: int,
) -> np.ndarray:
    half = crop_size // 2
    center_x, center_y = int(round(cursor_xy[0])), int(round(cursor_xy[1]))
    x0 = max(center_x - half, 0)
    y0 = max(center_y - half, 0)
    x1 = min(center_x + half + 1, image.shape[1])
    y1 = min(center_y + half + 1, image.shape[0])
    crop = image[y0:y1, x0:x1]
    magnified = cv2.resize(
        crop,
        (crop.shape[1] * scale, crop.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    cx = (center_x - x0) * scale
    cy = (center_y - y0) * scale
    cv2.line(magnified, (cx - 24, cy), (cx + 24, cy), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(magnified, (cx, cy - 24), (cx, cy + 24), (0, 255, 255), 1, cv2.LINE_AA)
    return magnified


def _point_is_visible(point: tuple[int, int], window_size: tuple[int, int]) -> bool:
    return -30 <= point[0] <= window_size[0] + 30 and -30 <= point[1] <= window_size[1] + 30


def _adjust_zoom(current: float, factor: float) -> float:
    return min(max(current * factor, 1.0), 16.0)


def _renumber(points: list[dict[str, Any]]) -> None:
    for index, point in enumerate(points, start=1):
        point["id"] = index


def _draw_status(image: np.ndarray, text: str, y: int) -> None:
    _draw_text(image, text, (15, y), 0.65)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    raise SystemExit(main())





