from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import webbrowser

import cv2

from snookerhelp.recognition import table_state_from_legacy_report
from snookerhelp.recognition.evidence_experiment import run_evidence_experiment
from snookerhelp.recognition.evidence_maps import (
    compute_full_table_evidence_maps,
    estimate_global_cloth_reference,
)
from snookerhelp.core.config import PROJECT_ROOT, load_yaml, resolve_path
from snookerhelp.core.ground_truth import (
    empty_ground_truth,
    ground_truth_from_dict,
    load_ground_truth,
    save_ground_truth,
)
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


def _make_handler(
    reports_root: Path,
    *,
    annotations_root: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    annotation_store = (
        annotations_root.resolve()
        if annotations_root is not None
        else PROJECT_ROOT / "benchmarks" / "annotations"
    )
    experiment_cache: dict[str, Any] = {}
    experiment_cache_lock = threading.Lock()

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

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._handle_post()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
            except Exception as exc:  # pragma: no cover - defensive server guard
                try:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
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
            if path.startswith("/api/annotations/"):
                stem = _strip_prefix(path, "/api/annotations/")
                self._send_json(
                    _load_annotation_payload(
                        reports_root,
                        stem,
                        annotations_root=annotation_store,
                    )
                )
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
            if path.startswith("/api/annotations/"):
                stem = _strip_prefix(path, "/api/annotations/")
                table_state = _load_table_state(reports_root, stem).to_dict()
                payload = self._read_json_body()
                annotation = ground_truth_from_dict(
                    payload,
                    image_name=stem,
                    image_path=table_state.get("image_path"),
                )
                output_path = _annotation_path(stem, root=annotation_store)
                save_ground_truth(annotation, output_path)
                self._send_json(
                    {
                        "ok": True,
                        "ground_truth": annotation.to_dict(),
                        "storage": _display_path(output_path),
                    }
                )
                return
            if not path.startswith("/api/review/"):
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            stem = _strip_prefix(path, "/api/review/")
            report_dir = _safe_report_dir(reports_root, stem)
            payload = self._read_json_body()
            review = review_feedback_from_dict(payload, image_name=stem)
            output_path = report_dir / "review_v1.json"
            output_path.write_text(
                json.dumps(review.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            self._send_json({"ok": True, "review": review.to_dict()})

        def _handle_post(self) -> None:
            path = unquote(urlparse(self.path).path)
            prefix = "/api/experiments/evidence/"
            if not path.startswith(prefix):
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            relative = _strip_prefix(path, prefix)
            parts = relative.split("/")
            if len(parts) != 2:
                raise ValueError(
                    "experiment path must be /api/experiments/evidence/<image>/<ball_id>"
                )
            stem, ball_id_text = parts
            ball_id = int(ball_id_text)
            payload = self._read_json_body()
            with experiment_cache_lock:
                context = _experiment_context(
                    reports_root,
                    stem,
                    experiment_cache,
                )
            annotation_payload = _load_annotation_payload(
                reports_root,
                stem,
                annotations_root=annotation_store,
            )["ground_truth"]
            ground_truth_ball = next(
                (
                    item
                    for item in annotation_payload.get("balls", [])
                    if int(item.get("ball_id", -1)) == ball_id
                ),
                None,
            )
            result = run_evidence_experiment(
                source_image=context["source_image"],
                table_state=context["table_state"],
                ball_id=ball_id,
                evidence_settings=context["evidence_settings"],
                parameters=payload.get("parameters") or payload,
                ground_truth_ball=ground_truth_ball,
                global_cloth_model=context["global_cloth_model"],
                full_table_evidence_maps=context["full_table_evidence_maps"],
            )
            self._send_json(result)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise ValueError("request body is too large")
            if length <= 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object")
            return payload

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
    table_state = _load_table_state(reports_root, stem)
    return {
        "table_state": table_state.to_dict(),
        "review_feedback": _load_review_payload(reports_root, stem)["review_feedback"],
        "assets_base": f"/assets/{stem}/",
    }


def _load_table_state(reports_root: Path, stem: str) -> Any:
    report_dir = _safe_report_dir(reports_root, stem)
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    return table_state_from_legacy_report(report, report_stem=stem)


def _load_annotation_payload(
    reports_root: Path,
    stem: str,
    *,
    annotations_root: Path | None = None,
) -> dict[str, Any]:
    table_state = _load_table_state(reports_root, stem).to_dict()
    path = _annotation_path(stem, root=annotations_root)
    if path.is_file():
        value = load_ground_truth(
            path,
            image_name=stem,
            image_path=table_state.get("image_path"),
        )
    else:
        value = empty_ground_truth(
            image_name=stem,
            image_path=table_state.get("image_path"),
        )
    return {
        "ground_truth": value.to_dict(),
        "storage": _display_path(path),
    }


def _annotation_path(stem: str, *, root: Path | None = None) -> Path:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError("invalid image stem")
    annotation_root = root or PROJECT_ROOT / "benchmarks" / "annotations"
    return annotation_root / f"{stem}.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _experiment_context(
    reports_root: Path,
    stem: str,
    cache: dict[str, Any],
) -> dict[str, Any]:
    report_path = _safe_report_dir(reports_root, stem) / "report.json"
    cache_key = (stem, report_path.stat().st_mtime_ns)
    if cache.get("key") == cache_key:
        return cache["value"]

    table_state = _load_table_state(reports_root, stem).to_dict()
    source_path_value = table_state.get("image_path")
    if not source_path_value:
        raise ValueError("table state does not contain source image_path")
    source_path = resolve_path(source_path_value)
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise FileNotFoundError(f"Could not read source image: {source_path}")
    detector_config = load_yaml("configs/detector_classical.yaml")
    evidence_settings = dict(detector_config.get("evidence_maps") or {})
    cloth_balls = [
        {
            "source_refined_center_px": ball.get("source_px"),
            "source_radius_px": ball.get("radius_px"),
        }
        for ball in table_state.get("balls", [])
    ]
    global_cloth = estimate_global_cloth_reference(
        source_image=source_image,
        table_corners_px=table_state.get("table_corners_px"),
        balls=cloth_balls,
        settings=evidence_settings,
    )
    full_maps = compute_full_table_evidence_maps(
        source_image=source_image,
        table_corners_px=table_state.get("table_corners_px"),
        settings={**evidence_settings, "global_cloth_model": global_cloth},
    )
    value = {
        "table_state": table_state,
        "source_image": source_image,
        "evidence_settings": evidence_settings,
        "global_cloth_model": global_cloth,
        "full_table_evidence_maps": full_maps,
    }
    cache.clear()
    cache.update({"key": cache_key, "value": value})
    return value


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
