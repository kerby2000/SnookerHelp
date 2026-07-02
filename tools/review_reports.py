from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snookerhelp.review.server import run_review_server


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the SnookerHelp v1 schema-driven review UI"
    )
    parser.add_argument("--reports", default="outputs/reports")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the browser automatically",
    )
    args = parser.parse_args()
    run_review_server(
        args.reports,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
