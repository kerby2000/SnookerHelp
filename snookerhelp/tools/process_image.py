from __future__ import annotations

import argparse
from pathlib import Path

from snookerhelp.core.config import load_yaml, resolve_path
from snookerhelp.recognition.estimator import StateEstimator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process one image or the latest image through the v1 entrypoint.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Process the newest matching JPEG from the configured/source folder.",
    )
    parser.add_argument("--image", default=None, help="Image path for single-image mode.")
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--folder", default=None, help="Folder for --latest mode.")
    parser.add_argument("--pattern", default=None, help="Glob pattern for --latest mode.")
    parser.add_argument("--output-dir", default=None, help="Override output root directory.")
    args = parser.parse_args(argv)

    image_path = _resolve_input_image(args)
    estimator = StateEstimator.from_config(args.config)
    frame, output_directory = estimator.process_and_save(image_path, args.output_dir)
    print(
        f"Processed {image_path}; detected "
        f"{frame.state['detection']['ball_count']} balls; "
        f"outputs: {output_directory}"
    )
    return 0


def _resolve_input_image(args: argparse.Namespace) -> Path:
    if not args.latest:
        if not args.image:
            raise SystemExit("--image is required unless --latest is set")
        return resolve_path(args.image)

    config = load_yaml(args.config)
    source = config["image_source"]
    folder = resolve_path(args.folder or source["folder"])
    pattern = args.pattern or source["filename_pattern"]
    return find_latest(folder, pattern)


def find_latest(folder: str | Path, pattern: str) -> Path:
    folder_path = resolve_path(folder)
    candidates = [path for path in folder_path.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder_path}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


if __name__ == "__main__":
    raise SystemExit(main())

