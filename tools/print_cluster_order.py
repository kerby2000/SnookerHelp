from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snookerhelp.core.ball_numbering import canonical_ball_id_map


def main() -> int:
    args = _parse_args()
    report_path = _resolve_report_path(args)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state = report.get("state") or report
    balls = list(state.get("balls") or [])
    if not balls:
        raise SystemExit(f"No balls found in {report_path}")

    numbering = canonical_ball_id_map(balls)
    raw_to_canonical = {
        int(raw_id): int(metadata.get("canonical_ball_id", raw_id))
        for raw_id, metadata in numbering.items()
    }
    label_by_raw = {
        int(ball.get("id", ball.get("ball_id", 0))): str(
            ball.get("color_label") or ball.get("class") or ball.get("label") or "unknown"
        )
        for ball in balls
    }

    clusters = (
        ((state.get("scene_constraints") or {}).get("adjacent_ball_clusters") or {})
        .get("clusters")
        or []
    )
    if not clusters:
        print(f"No adjacent-ball clusters found in {report_path}")
        return 0

    print(f"Report: {report_path}")
    for cluster in clusters:
        _print_cluster(
            cluster,
            raw_to_canonical=raw_to_canonical,
            label_by_raw=label_by_raw,
            path_name=args.path,
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print adjacent-cluster traversal orders from a generated SnookerHelp report.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to report.json or a report directory containing report.json.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("outputs/reports_v1_global_cloth"),
        help="Reports root used with --image. Default: outputs/reports_v1_global_cloth.",
    )
    parser.add_argument(
        "--image",
        help="Image/report stem, e.g. DSC00540. Used when --report is omitted.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Optional path name to print first, e.g. outside_in_perimeter_walk.",
    )
    return parser.parse_args()


def _resolve_report_path(args: argparse.Namespace) -> Path:
    if args.report:
        path = args.report
        if path.is_dir():
            path = path / "report.json"
        return path
    if not args.image:
        raise SystemExit("Provide --report or --image")
    stem = Path(str(args.image)).stem
    return args.reports_root / stem / "report.json"


def _print_cluster(
    cluster: dict[str, Any],
    *,
    raw_to_canonical: dict[int, int],
    label_by_raw: dict[int, str],
    path_name: str | None,
) -> None:
    cluster_id = cluster.get("cluster_id")
    members = list(cluster.get("members") or [])
    traversal = cluster.get("traversal") or {}
    paths = traversal.get("paths") or {}
    print()
    print(
        f"Cluster {cluster_id}: status={cluster.get('status')} "
        f"members={len(members)} primary={traversal.get('primary_path')}"
    )
    print(
        "pair RMS mm: "
        f"{cluster.get('initial_pair_rms_mm')} -> {cluster.get('joint_pair_rms_mm')} "
        f"(improvement {cluster.get('improvement_mm')})"
    )
    fit_policy = cluster.get("fit_policy") or {}
    if fit_policy:
        print(f"fit policy: {json.dumps(fit_policy, sort_keys=True)}")
    ordered_path_names = list(paths)
    if path_name and path_name in paths:
        ordered_path_names.remove(path_name)
        ordered_path_names.insert(0, path_name)
    for name in ordered_path_names:
        raw_path = [int(ball_id) for ball_id in paths.get(name) or []]
        canonical_path = [raw_to_canonical.get(raw_id, raw_id) for raw_id in raw_path]
        print(f"{name} canonical: {_format_path(canonical_path)}")
        print(f"{name} raw:       {_format_path(raw_path)}")

    print()
    print("Rows sorted by primary rank:")
    print("primary canonical raw label role shell cw ccw walk rev angle_deg move_mm")
    rows = []
    for member in members:
        raw_id = int(member.get("id"))
        traversal_row = member.get("cluster_traversal") or {}
        shell = member.get("cluster_shell") or {}
        rows.append(
            (
                _rank_value(traversal_row.get("primary_rank")),
                raw_to_canonical.get(raw_id, raw_id),
                raw_id,
                label_by_raw.get(raw_id, str(member.get("label") or "unknown")),
                shell.get("role"),
                shell.get("shell_index"),
                traversal_row.get("outside_in_clockwise_rank"),
                traversal_row.get("outside_in_counterclockwise_rank"),
                traversal_row.get("outside_in_perimeter_walk_rank"),
                traversal_row.get("outside_in_perimeter_walk_reverse_rank"),
                traversal_row.get("angle_deg_from_top"),
                member.get("movement_mm"),
            )
        )
    for row in sorted(rows):
        print(
            f"{_fmt_rank(row[0])} #{row[1]:<2} raw#{row[2]:<2} {row[3]:<7} "
            f"{str(row[4]):<9} {row[5]!s:<5} {_fmt_rank(row[6])} "
            f"{_fmt_rank(row[7])} {_fmt_rank(row[8])} {_fmt_rank(row[9])} {row[10]} {row[11]}"
        )


def _rank_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 9999


def _fmt_rank(value: Any) -> str:
    try:
        return f"{int(value):>2}"
    except (TypeError, ValueError):
        return " -"


def _format_path(path: list[int]) -> str:
    return " -> ".join(f"#{ball_id}" for ball_id in path)


if __name__ == "__main__":
    raise SystemExit(main())
