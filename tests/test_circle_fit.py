import cv2
import numpy as np

from snookerhelp.recognition.circle_fit import fit_circle_least_squares, refine_circle


def test_least_squares_circle_fit_has_subpixel_accuracy() -> None:
    center = np.array([87.35, 63.72])
    radius = 24.4
    angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    points = center + np.column_stack((np.cos(angles), np.sin(angles))) * radius

    fit = fit_circle_least_squares(points)

    assert fit.success
    assert np.hypot(fit.x - center[0], fit.y - center[1]) < 0.01
    assert abs(fit.radius - radius) < 0.01


def test_image_circle_refinement_has_less_than_point_two_pixel_center_error() -> None:
    height, width = 160, 210
    center = (103.37, 78.64)
    radius = 25.2
    yy, xx = np.mgrid[:height, :width]
    distance = np.hypot(xx - center[0], yy - center[1])
    difference = (
        180.0 / (1.0 + np.exp((distance - radius) * 4.0))
    ).astype(np.float32)
    image = cv2.cvtColor(np.uint8(difference), cv2.COLOR_GRAY2BGR)

    fit = refine_circle(
        warped_image=image,
        difference=difference,
        approximate_center=(103.0, 79.0),
        approximate_radius=25.0,
    )

    assert fit.success
    assert np.hypot(fit.x - center[0], fit.y - center[1]) < 0.2
