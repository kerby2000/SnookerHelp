from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import webbrowser

from snookerhelp.recognition import table_state_from_legacy_report
from snookerhelp.review.schema import (
    CANONICAL_BALL_NUMBERING_SCHEME,
    V1_REVIEW_SCHEMA,
    default_review_feedback,
    review_feedback_from_dict,
)
from snookerhelp.core.schema import ReviewBallFeedback, ReviewFeedback


STATIC_ROOT = Path(__file__).resolve().parent / "static"


def run_review_server(
    reports_root: str | Path = "outputs/reports",
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
    open_browser: bool = False,
) -> None:
    root = Path(reports_root).resolve()
    handler = _make_handler(root)
    server = ThreadingHTTPServer((host, int(port)), handler)
    url = f"http://{host}:{port}/"
    print(f"SnookerHelp v1 review UI: {url}")
    print(f"Reports root: {root}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _make_handler(reports_root: Path) -> type[BaseHTTPRequestHandler]:
    class ReviewV1Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            try:
                self._handle_get()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
            except Exception as exc:  # pragma: no cover - defensive server guard
                try:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    return

        def do_PUT(self) -> None:  # noqa: N802
            try:
                self._handle_put()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
            except Exception as exc:  # pragma: no cover - defensive server guard
                try:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    return

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _handle_get(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self._send_file(STATIC_ROOT / "index.html")
                return
            if path.startswith("/static/"):
                self._send_file(_safe_child(STATIC_ROOT, _strip_prefix(path, "/static/")))
                return
            if path == "/api/reports":
                self._send_json({"reports": _list_reports(reports_root)})
                return
            if path.startswith("/api/table-state/"):
                stem = _strip_prefix(path, "/api/table-state/")
                self._send_json(_load_table_state_payload(reports_root, stem))
                return
            if path.startswith("/api/review/"):
                stem = _strip_prefix(path, "/api/review/")
                self._send_json(_load_review_payload(reports_root, stem))
                return
            if path.startswith("/assets/"):
                stem, relative = _asset_parts(_strip_prefix(path, "/assets/"))
                report_dir = _safe_report_dir(reports_root, stem)
                self._send_file(_safe_child(report_dir, relative))
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def _handle_put(self) -> None:
            path = unquote(urlparse(self.path).path)
            if not path.startswith("/api/review/"):
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            stem = _strip_prefix(path, "/api/review/")
            report_dir = _safe_report_dir(reports_root, stem)
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            review = review_feedback_from_dict(payload, image_name=stem)
            output_path = report_dir / "review_v1.json"
            output_path.write_text(
                json.dumps(review.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            self._send_json({"ok": True, "review": review.to_dict()})

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json({"error": "file not found"}, status=HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ReviewV1Handler


def _list_reports(reports_root: Path) -> list[dict[str, Any]]:
    reports = []
    for report_path in sorted(reports_root.glob("*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        review = _load_review_payload(reports_root, report_path.parent.name)["review_feedback"]
        reviewed = sum(
            1
            for item in review.get("balls", [])
            if item.get("decision") and item.get("decision") != "unreviewed"
        )
        reports.append(
            {
                "stem": report_path.parent.name,
                "image": report.get("image"),
                "ball_count": len((report.get("state") or {}).get("balls", [])),
                "reviewed_count": reviewed,
            }
        )
    return reports


def _load_table_state_payload(reports_root: Path, stem: str) -> dict[str, Any]:
    report_dir = _safe_report_dir(reports_root, stem)
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    table_state = table_state_from_legacy_report(report, report_stem=stem)
    return {
        "table_state": table_state.to_dict(),
        "review_feedback": _load_review_payload(reports_root, stem)["review_feedback"],
        "assets_base": f"/assets/{stem}/",
    }


def _load_review_payload(reports_root: Path, stem: str) -> dict[str, Any]:
    report_dir = _safe_report_dir(reports_root, stem)
    report_path = report_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table_state = table_state_from_legacy_report(report, report_stem=stem)
    ball_ids = [int(ball.ball_id) for ball in table_state.balls]
    raw_to_canonical = {
        int(raw_id): int(canonical_id)
        for raw_id, canonical_id in (
            table_state.diagnostics.get("raw_to_canonical_ball_ids") or {}
        ).items()
    }
    review_v1_path = report_dir / "review_v1.json"
    review_legacy_path = report_dir / "review.json"
    if review_v1_path.is_file():
        payload = json.loads(review_v1_path.read_text(encoding="utf-8"))
        review = review_feedback_from_dict(payload, image_name=stem)
        if payload.get("numbering_scheme") != CANONICAL_BALL_NUMBERING_SCHEME:
            review = _remap_review_feedback_ids(review, raw_to_canonical)
    elif review_legacy_path.is_file():
        payload = json.loads(review_legacy_path.read_text(encoding="utf-8"))
        review = review_feedback_from_dict(payload, image_name=stem)
        review = _remap_review_feedback_ids(review, raw_to_canonical)
    else:
        review = default_review_feedback(image_name=stem, ball_ids=ball_ids)
    return {"review_feedback": review.to_dict()}


def _remap_review_feedback_ids(
    review: ReviewFeedback,
    raw_to_canonical: dict[int, int],
) -> ReviewFeedback:
    remapped_balls = []
    for item in review.balls:
        canonical_id = raw_to_canonical.get(int(item.ball_id), int(item.ball_id))
        remapped_balls.append(
            ReviewBallFeedback(
                ball_id=canonical_id,
                decision=item.decision,
                issue_tags=list(item.issue_tags),
                confidence=item.confidence,
                comment=item.comment,
                manual_correction=item.manual_correction,
            )
        )
    return ReviewFeedback(
        schema_version=V1_REVIEW_SCHEMA,
        image_name=review.image_name,
        numbering_scheme=CANONICAL_BALL_NUMBERING_SCHEME,
        balls=remapped_balls,
        missing_balls=list(review.missing_balls),
        audit_trail=list(review.audit_trail),
    )


def _safe_report_dir(reports_root: Path, stem: str) -> Path:
    report_dir = _safe_child(reports_root, stem)
    report_path = report_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Unknown report: {stem}")
    return report_dir


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    child = (root / relative).resolve()
    if root != child and root not in child.parents:
        raise ValueError("path escapes root")
    return child


def _asset_parts(value: str) -> tuple[str, str]:
    parts = value.split("/", 1)
    if len(parts) != 2:
        raise ValueError("asset path must include report stem and relative file")
    return parts[0], parts[1]


def _strip_prefix(value: str, prefix: str) -> str:
    if not value.startswith(prefix):
        return value
    return value[len(prefix) :]
