from __future__ import annotations

import argparse

from snookerhelp.calibration.charuco import (
    calibrate_intrinsics_command,
    estimate_table_pose_command,
)
from snookerhelp.calibration.homography_bootstrap import click_table_corners_command


_KIND_TO_COMMAND = {
    "charuco": calibrate_intrinsics_command,
    "table-corners": click_table_corners_command,
    "table-pose": estimate_table_pose_command,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="v1 calibration wrapper. Remaining arguments are passed through.",
    )
    parser.add_argument(
        "--kind",
        choices=sorted(_KIND_TO_COMMAND),
        required=True,
    )
    args, passthrough = parser.parse_known_args(argv)
    return _KIND_TO_COMMAND[args.kind](passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
