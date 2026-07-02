from __future__ import annotations

import argparse
import glob
import json
from html import escape
from pathlib import Path
from typing import Any

from snookerhelp.qa.reporting import generate_image_report
from snookerhelp.core.config import resolve_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SnookerHelp v1-compatible visual reports.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    image_parser = subparsers.add_parser(
        "image",
        help="Generate one per-image report folder.",
    )
    _add_common_args(image_parser)
    image_parser.add_argument("--image", required=True, help="Input JPEG path")
    image_parser.add_argument(
        "--ball-id",
        type=int,
        default=None,
        help="Selected ball id for the geometry panel.",
    )

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Generate report folders for all images matching a glob.",
    )
    _add_common_args(dataset_parser)
    dataset_parser.add_argument(
        "--glob",
        required=True,
        help='Input glob, for example "Media/**/*.JPG"',
    )
    dataset_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    if args.mode == "image":
        return generate_single_image_command(args)
    if args.mode == "dataset":
        return generate_dataset_command(args)
    raise AssertionError(args.mode)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="outputs/reports")
    parser.add_argument("--config", default="configs/sony_dev.yaml")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--selected-ball", default="auto")


def generate_single_image_command(args: argparse.Namespace) -> int:
    _, output_directory = generate_image_report(
        image_path=args.image,
        output_root=args.output,
        config_path=args.config,
        scenario_path=args.scenario,
        selected_ball=args.selected_ball,
        ball_id=getattr(args, "ball_id", None),
    )
    print(f"Report JSON: {output_directory / 'report.json'}")
    print(f"Static QA report: {output_directory / 'report.html'}")
    return 0


def generate_dataset_command(args: argparse.Namespace) -> int:
    matches = sorted(Path(path) for path in glob.glob(args.glob, recursive=True))
    if args.limit is not None:
        matches = matches[: args.limit]
    if not matches:
        raise FileNotFoundError(f"No images matched glob: {args.glob}")

    outputs: list[dict[str, Any]] = []
    for index, image_path in enumerate(matches, start=1):
        print(f"[{index}/{len(matches)}] {image_path}")
        report, output_directory = generate_image_report(
            image_path=image_path,
            output_root=args.output,
            config_path=args.config,
            scenario_path=args.scenario,
            selected_ball=args.selected_ball,
        )
        outputs.append(
            {
                "image": str(image_path),
                "report_json": str(output_directory / "report.json"),
                "static_report": str(output_directory / "report.html"),
                "ball_count": report["summary"]["ball_count"],
                "source_refinement_success_count": report["summary"][
                    "source_refinement_success_count"
                ],
                "validation_rows": report["physical_validation"]["row_count"],
            }
        )

    output_root = resolve_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "dataset_reports.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": "snookerhelp.dataset_reports.v1", "reports": outputs}, handle, indent=2)
        handle.write("\n")
    html_index_path = output_root / "index.html"
    html_index_path.write_text(_dataset_index_html(outputs, output_root), encoding="utf-8")
    print(f"Dataset report index: {index_path}")
    print(f"Dataset HTML index: {html_index_path}")
    print("Review UI: python tools/review_reports.py --reports " + str(output_root))
    return 0


def _dataset_index_html(outputs: list[dict[str, Any]], output_root: Path) -> str:
    rows = []
    for item in outputs:
        report_path = Path(str(item["static_report"]))
        try:
            href = report_path.relative_to(output_root).as_posix()
        except ValueError:
            href = str(report_path)
        rows.append(
            f"""
            <tr>
              <td>{escape(Path(str(item["image"])).name)}</td>
              <td><a href="{escape(href)}">static QA report</a></td>
              <td>{escape(str(item["ball_count"]))}</td>
              <td>{escape(str(item["source_refinement_success_count"]))}</td>
              <td>{escape(str(item["validation_rows"]))}</td>
            </tr>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SnookerHelp dataset reports</title>
  <style>
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      background: #101418;
      color: #e8eef5;
    }}
    main {{
      max-width: 1100px;
      margin: auto;
      padding: 28px;
    }}
    code {{
      background: #1b2530;
      border: 1px solid #334456;
      border-radius: 6px;
      padding: 2px 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
    }}
    th, td {{
      border-bottom: 1px solid #2e3a48;
      padding: 9px 10px;
      text-align: left;
    }}
    a {{ color: #9bdcff; }}
    .muted {{ color: #9aa8b7; }}
  </style>
</head>
<body>
  <main>
    <h1>SnookerHelp dataset reports</h1>
    <p class="muted">
      These folders contain immutable machine reports. Use the v1 review UI for
      OK/NOK decisions, manual corrections, missing balls, and feedback export.
    </p>
    <p><code>python tools/review_reports.py --reports {escape(str(output_root))}</code></p>
    <table>
      <thead>
        <tr><th>Image</th><th>Static QA page</th><th>Balls</th><th>Source fits</th><th>Validation rows</th></tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

