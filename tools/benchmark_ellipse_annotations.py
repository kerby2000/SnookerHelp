from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snookerhelp.qa.ellipse_benchmark import (
    evaluate_report_file,
    write_ellipse_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare detector ellipses with tracked perfect-ellipse annotations.",
    )
    parser.add_argument("--report", required=True, help="Path to report.json")
    parser.add_argument("--annotations", required=True, help="Ground-truth JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--tolerance-px",
        type=float,
        default=3.0,
        help="Default annotation center tolerance when not stored per ball",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_report_file(
        args.report,
        args.annotations,
        default_tolerance_px=args.tolerance_px,
    )
    json_path, csv_path = write_ellipse_benchmark(result, args.output)
    summary = result["summary"]
    print(f"Ellipse benchmark: {result['image_name']}")
    print(
        "Annotated/computed: "
        f"{summary['annotated_ball_count']}/{summary['computed_ball_count']}"
    )
    print(
        "Center error px mean/median/p95/max: "
        f"{summary['mean_source_center_error']}/"
        f"{summary['median_source_center_error']}/"
        f"{summary['p95_source_center_error']}/"
        f"{summary['max_source_center_error']}"
    )
    print(
        "Contour RMS px mean/median/p95/max: "
        f"{summary['mean_contour_rms_error']}/"
        f"{summary['median_contour_rms_error']}/"
        f"{summary['p95_contour_rms_error']}/"
        f"{summary['max_contour_rms_error']}"
    )
    print(f"JSON: {Path(json_path)}")
    print(f"CSV: {Path(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
