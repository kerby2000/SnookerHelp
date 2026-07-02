from __future__ import annotations

import argparse

from snookerhelp.review.server import run_review_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SnookerHelp v1 review UI")
    parser.add_argument("--reports", default="outputs/reports")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    run_review_server(
        args.reports,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
