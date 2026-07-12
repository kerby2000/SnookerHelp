import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from snookerhelp.core.ball_numbering import CANONICAL_BALL_NUMBERING_SCHEME
from snookerhelp.review.evidence_export import review_payload_from_report
from snookerhelp.review.feedback import (
    load_feedback_from_reports_root,
    load_legacy_feedback_jsonl,
    save_feedback_jsonl,
)
from snookerhelp.review.server import (
    _load_review_payload,
    _load_table_state_payload,
    _make_handler,
)


def _write_minimal_report(root: Path) -> Path:
    report_dir = root / "DSC00001"
    report_dir.mkdir()
    report = {
        "image": "Media/example/DSC00001.JPG",
        "camera_model": {"mode": "approximate"},
        "review_evidence": {
            "source_image_path": "01_source_detection.png",
            "source_size_px": {"width": 100, "height": 80},
            "table_corner_points_px": [[0, 0], [100, 0], [100, 80], [0, 80]],
            "balls": [
                {
                    "id": 1,
                    "label": "red",
                    "source_crop_path": "crops/ball_001.jpg",
                    "source_center_px": [50.0, 40.0],
                    "ellipse_fit": {
                        "source": "radial_boundary",
                        "center_px": [50.0, 40.0],
                        "major_axis_px": 20.0,
                        "minor_axis_px": 18.0,
                        "angle_deg": 5.0,
                    },
                    "review_confidence": 0.75,
                    "physics_c_only_model_decision": {
                        "table_position_trust": "medium",
                        "reasons": ["candidate_c_and_sphere_match"],
                    },
                }
            ],
        },
        "state": {
            "balls": [
                {
                    "id": 1,
                    "class": "red",
                    "source_refined_center_px": [50.0, 40.0],
                    "source_refined_table_xy_mm": [1000.0, 500.0],
                    "source_radius_px": 9.0,
                }
            ]
        },
    }
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def _write_empty_report(root: Path) -> Path:
    report_dir = root / "DSC_EMPTY"
    report_dir.mkdir()
    report = {
        "image": "Media/example/DSC_EMPTY.JPG",
        "review_evidence": {
            "source_image_path": "01_source_detection.png",
            "source_size_px": {"width": 100, "height": 80},
            "table_corner_points_px": [[0, 0], [100, 0], [100, 80], [0, 80]],
            "balls": [],
        },
        "state": {"balls": []},
    }
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_v1_review_payload_from_report_has_table_state_and_feedback(tmp_path: Path) -> None:
    report_path = _write_minimal_report(tmp_path)

    payload = review_payload_from_report(report_path)

    assert payload["table_state"]["schema_version"] == "snookerhelp.table_state.v1"
    assert payload["review_feedback"]["schema_version"] == "snookerhelp.review_feedback.v1"
    assert payload["review_feedback"]["numbering_scheme"] == CANONICAL_BALL_NUMBERING_SCHEME
    assert payload["table_state"]["balls"][0]["ball_id"] == 8
    assert payload["review_feedback"]["balls"][0]["ball_id"] == 8
    assert payload["table_state"]["balls"][0]["confidence"]["reasons"] == [
        "image_evidence_and_physical_model_match"
    ]


def test_v1_server_payload_loaders_use_v1_schema(tmp_path: Path) -> None:
    _write_minimal_report(tmp_path)

    table_payload = _load_table_state_payload(tmp_path, "DSC00001")
    review_payload = _load_review_payload(tmp_path, "DSC00001")

    assert table_payload["table_state"]["schema_version"] == "snookerhelp.table_state.v1"
    assert table_payload["assets_base"] == "/assets/DSC00001/"
    assert review_payload["review_feedback"]["numbering_scheme"] == CANONICAL_BALL_NUMBERING_SCHEME
    assert review_payload["review_feedback"]["balls"][0]["ball_id"] == 8


def test_v1_server_put_saves_manual_correction_and_missing_ball(tmp_path: Path) -> None:
    _write_minimal_report(tmp_path)
    handler = _make_handler(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        payload = {
            "schema_version": "snookerhelp.review_feedback.v1",
            "image_name": "DSC00001",
            "numbering_scheme": CANONICAL_BALL_NUMBERING_SCHEME,
            "balls": [
                {
                    "ball_id": 8,
                    "decision": "needs_review",
                    "issue_tags": ["near cushion"],
                    "comment": "manual correction",
                    "manual_correction": {
                        "correction_type": "source_pixel+ellipse+cushion_line",
                        "source_px": [11.0, 22.0],
                        "ellipse_px": {
                            "center_px": [11.0, 22.0],
                            "major_axis_px": 30.0,
                            "minor_axis_px": 20.0,
                            "angle_deg": 5.0,
                        },
                        "cushion_line_px": [[1.0, 2.0], [3.0, 4.0]],
                    },
                }
            ],
            "missing_balls": [
                {"label_guess": "red", "source_px": [33.0, 44.0], "comment": "missed"}
            ],
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/review/DSC00001",
            data=json.dumps(payload).encode("utf-8"),
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            saved = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    saved_path = tmp_path / "DSC00001" / "review_v1.json"
    saved_file = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved_file["balls"][0]["manual_correction"]["source_px"] == [11.0, 22.0]
    assert saved_file["balls"][0]["manual_correction"]["ellipse_px"]["major_axis_px"] == 30.0
    assert saved_file["balls"][0]["manual_correction"]["cushion_line_px"][1] == [3.0, 4.0]
    assert saved_file["missing_balls"][0]["source_px"] == [33.0, 44.0]


def test_v1_server_put_saves_missing_ball_only_review(tmp_path: Path) -> None:
    _write_empty_report(tmp_path)
    handler = _make_handler(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        payload = {
            "schema_version": "snookerhelp.review_feedback.v1",
            "image_name": "DSC_EMPTY",
            "balls": [],
            "missing_balls": [
                {"label_guess": "red", "source_px": [12.0, 34.0], "comment": "only item"}
            ],
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/review/DSC_EMPTY",
            data=json.dumps(payload).encode("utf-8"),
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            saved = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    saved_path = tmp_path / "DSC_EMPTY" / "review_v1.json"
    saved_file = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved_file["balls"] == []
    assert saved_file["missing_balls"][0]["source_px"] == [12.0, 34.0]


def test_v1_server_round_trips_perfect_ellipse_annotation(tmp_path: Path) -> None:
    _write_minimal_report(tmp_path)
    annotation_root = tmp_path / "annotations"
    handler = _make_handler(tmp_path, annotations_root=annotation_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    payload = {
        "schema_version": "snookerhelp.ground_truth.v1",
        "image_name": "DSC00001",
        "coordinate_system": "source_px",
        "balls": [
            {
                "ball_id": 8,
                "label": "red",
                "point": [50.5, 40.25],
                "ellipse_px": {
                    "center_px": [50.5, 40.25],
                    "major_axis_px": 20.0,
                    "minor_axis_px": 18.0,
                    "angle_deg": 5.0,
                    "source": "manual_review_ui",
                },
            }
        ],
    }
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/annotations/DSC00001",
            data=json.dumps(payload).encode("utf-8"),
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            saved = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/annotations/DSC00001",
            timeout=5,
        ) as response:
            loaded = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert saved["ok"] is True
    assert loaded["ground_truth"]["balls"][0]["ball_id"] == 8
    assert loaded["ground_truth"]["balls"][0]["ellipse_px"]["center_px"] == [
        50.5,
        40.25,
    ]
    assert (annotation_root / "DSC00001.json").is_file()


def test_v1_static_html_uses_product_language() -> None:
    html = Path("snookerhelp/review/static/index.html").read_text(encoding="utf-8")
    app = Path("snookerhelp/review/static/app.js").read_text(encoding="utf-8")

    assert "Pixels" in html
    assert "Image evidence" in html
    assert "Physical model" in html
    assert "Final estimate" in html
    assert "Confidence" in html
    assert "Ball statistics" in html
    assert "Evidence layers" in html
    assert "mapControls" in html
    assert "layerControls" in html
    assert "floatingLegend" in html
    assert "Help / legend" in html
    assert "uiVersion" in html
    assert "sourceZoomLabel" in html
    assert "sourceFitSelected" in html
    assert 'const UI_VERSION = "v1.7.0"' in app
    assert "globalClusterText" in app
    assert "setupSourceViewport" in app
    assert "fitSourceToSelected" in app
    assert "zoomSourceAtClient" in app
    assert "updateSourceLabelScale" in app
    assert "sourcePrintClusterOrder" in html
    assert "printClusterOrder" in app
    assert "recoveredBoundaryDots" not in app
    assert "Evidence background" in app
    assert "Display tuning" in app
    assert "view only; fit is unchanged" in app
    assert "Evidence-view score" in html
    assert "Neighbor geometry" in html
    assert "neighbor ellipses" in app
    assert "Cluster labels" in html
    assert "Cluster traversal" in app
    assert "clusterTraversalText" in app
    assert "Final source center" in html
    assert "Confidence score" in html
    assert "Rejected reasons" in app
    assert "addbackScenarioSummary" in app
    assert "Add-back fit scenarios" not in app
    assert "Consensus add-back fit" not in app
    assert "Cluster arc-combo fit" in app
    assert "drawConsensusRejectRefit" not in app
    assert "consensus-selected rejects" not in html.lower()
    assert "Cluster-combination promotion" in html
    assert "arc_combination_refit" in app
    assert "overlay-matrix" in app
    assert "view_score" in app
    assert "evidenceViewScore" in app
    assert "recommendedEvidenceKey" in app
    assert "activateEvidenceRow" in app
    assert "overlaySelectionFor" in app
    assert "selectedBoundaryVariant" in app
    assert "ball_vs_cloth_probability" in app
    assert "physicalModelRows" in app
    assert "Approximate camera model limits trust" in app
    assert "Scene constraints" in app
    assert "Copy selected summary" not in html
    assert "Manual ellipse JSON" not in html
    assert "Manual cushion line" not in html
    assert "Pick manual center from crop" not in html
    assert "Click table for missing ball" not in html
    assert "experimentControls" in html
    assert "ellipseEditorControls" in html
    assert "Perfect ellipse" in html
    assert "runEvidenceExperiment" in app
    assert "savePerfectEllipse" in app
    forbidden = [
        "Candidate A",
        "Candidate B",
        "Candidate C",
        "Candidate D",
        "Hough",
        "fallback radial",
        "manual homography",
        "source refined center",
    ]
    for term in forbidden:
        assert term not in html


def test_v1_feedback_migrator_reads_legacy_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "feedback.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_type": "ball_review",
                        "image": "Media/02_random_balls/DSC00524.JPG",
                        "ball_id": 1,
                        "human_decision": "ok",
                        "issue_tags": ["weak_fit"],
                        "human_confidence": 0.8,
                        "comment": "small correction",
                        "manual_center_px": [10.0, 20.0],
                        "manual_model": "ellipse",
                    }
                ),
                json.dumps(
                    {
                        "row_type": "missing_ball_hint",
                        "image": "Media/02_random_balls/DSC00524.JPG",
                        "source_px": [30.0, 40.0],
                        "label_guess": "red",
                        "comment": "missed near cushion",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    migrated = load_legacy_feedback_jsonl(source)

    assert len(migrated) == 1
    payload = migrated[0].to_dict()
    assert payload["schema_version"] == "snookerhelp.review_feedback.v1"
    assert payload["balls"][0]["manual_correction"]["source_px"] == [10.0, 20.0]
    assert payload["missing_balls"][0]["label_guess"] == "red"


def test_v1_feedback_export_reads_review_v1_files(tmp_path: Path) -> None:
    report_path = _write_minimal_report(tmp_path)
    review = {
        "schema_version": "snookerhelp.review_feedback.v1",
        "image_name": "DSC00001",
        "numbering_scheme": CANONICAL_BALL_NUMBERING_SCHEME,
        "balls": [{"ball_id": 8, "decision": "ok", "issue_tags": ["checked"]}],
        "missing_balls": [{"label_guess": "red", "source_px": [10.0, 20.0]}],
    }
    (report_path.parent / "review_v1.json").write_text(json.dumps(review), encoding="utf-8")

    feedback_items = load_feedback_from_reports_root(tmp_path)
    output = tmp_path / "feedback_v1.jsonl"
    save_feedback_jsonl(feedback_items, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["schema_version"] == "snookerhelp.review_feedback.v1"
    assert rows[0]["balls"][0]["decision"] == "ok"
    assert rows[0]["missing_balls"][0]["source_px"] == [10.0, 20.0]
