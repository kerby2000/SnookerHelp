from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_UI_TERMS = [
    "Candidate A",
    "Candidate B",
    "Candidate C",
    "Candidate D",
    "Hough",
    "fallback radial",
    "manual homography",
    "source refined center",
]


@dataclass(frozen=True)
class GateResult:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def run_architecture_gates(project_root: str | Path = ".") -> list[GateResult]:
    root = Path(project_root)
    return [
        _active_code_has_no_legacy_or_vision_imports(root),
        _vision_source_tree_deleted(root),
        _static_review_ui_uses_product_language(root),
        _active_static_report_is_v1_owned(root),
        _old_static_feedback_renderer_deleted(root),
        _old_review_ui_deleted(root),
        _legacy_source_tree_deleted(root),
    ]


def architecture_gate_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run lightweight SnookerHelp v1 architecture/refactor gates.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    results = run_architecture_gates(args.root)
    payload = {
        "schema_version": "snookerhelp.architecture_gates.v1",
        "ok": all(result.ok for result in results),
        "results": [result.to_dict() for result in results],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            status = "PASS" if result.ok else "FAIL"
            print(f"{status} {result.name}: {result.detail}")
    return 0 if payload["ok"] else 1


def _active_code_has_no_legacy_or_vision_imports(root: Path) -> GateResult:
    offenders: list[str] = []
    for package_root in [root / "snookerhelp", root / "tools"]:
        for path in package_root.rglob("*.py"):
            if _imports_package(path, {"vision", "legacy"}):
                offenders.append(_rel(path, root))
    if offenders:
        return GateResult(
            "active_import_boundary",
            False,
            "active code references legacy compatibility packages: "
            + ", ".join(offenders[:8]),
        )
    return GateResult(
        "active_import_boundary",
        True,
        "active snookerhelp/tools code has no vision.* or legacy.* imports",
    )


def _vision_source_tree_deleted(root: Path) -> GateResult:
    vision_root = root / "vision"
    if vision_root.exists():
        return GateResult(
            "vision_source_tree_deleted",
            False,
            "vision source tree still exists: " + _rel(vision_root, root),
        )
    return GateResult(
        "vision_source_tree_deleted",
        True,
        "vision compatibility source tree has been removed after v1 gates passed",
    )


def _static_review_ui_uses_product_language(root: Path) -> GateResult:
    html_path = root / "snookerhelp" / "review" / "static" / "index.html"
    text = html_path.read_text(encoding="utf-8")
    missing = [
        term
        for term in (
            "Pixels",
            "Image evidence",
            "Physical model",
            "Final estimate",
            "Confidence",
            "Ball statistics",
        )
        if term not in text
    ]
    forbidden = [term for term in FORBIDDEN_UI_TERMS if term in text]
    if missing or forbidden:
        return GateResult(
            "review_ui_language",
            False,
            f"missing={missing}; forbidden={forbidden}",
        )
    return GateResult(
        "review_ui_language",
        True,
        "v1 review UI exposes product language, not prototype candidate labels",
    )


def _active_static_report_is_v1_owned(root: Path) -> GateResult:
    path = root / "snookerhelp" / "qa" / "report_html.py"
    text = path.read_text(encoding="utf-8")
    if "legacy.static_reports_v0" in text or "snookerhelp_visual_feedback_v1" in text:
        return GateResult(
            "static_report_owner",
            False,
            "active static report still depends on old feedback renderer",
        )
    return GateResult(
        "static_report_owner",
        True,
        "active static report renderer is v1-owned read-only QA output",
    )


def _old_static_feedback_renderer_deleted(root: Path) -> GateResult:
    path = root / "legacy" / "static_reports_v0" / "report_html.py"
    if path.exists():
        return GateResult(
            "old_static_feedback_deleted",
            False,
            f"old static feedback renderer still exists: {_rel(path, root)}",
        )
    return GateResult(
        "old_static_feedback_deleted",
        True,
        "old browser-local static feedback renderer has been deleted",
    )


def _old_review_ui_deleted(root: Path) -> GateResult:
    obsolete = [
        root / "legacy" / "ui_v0" / "review_app.py",
        root / "legacy" / "tools_v0" / "review_reports.py",
        root / "legacy" / "tools_v0" / "review_reports_v1.py",
        root / "vision" / "review_app.py",
    ]
    existing = [_rel(path, root) for path in obsolete if path.exists()]
    if existing:
        return GateResult(
            "old_review_ui_deleted",
            False,
            "old review UI/server files still exist: " + ", ".join(existing),
        )
    return GateResult(
        "old_review_ui_deleted",
        True,
        "old review UI/server files have been deleted; active review uses v1",
    )


def _legacy_source_tree_deleted(root: Path) -> GateResult:
    legacy_root = root / "legacy"
    if legacy_root.exists():
        return GateResult(
            "legacy_source_tree_deleted",
            False,
            "legacy source tree still exists: " + _rel(legacy_root, root),
        )
    return GateResult(
        "legacy_source_tree_deleted",
        True,
        "legacy source tree has been removed after replacement gates passed",
    )


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _imports_package(path: Path, blocked_roots: set[str]) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in blocked_roots:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".", 1)[0] in blocked_roots:
                return True
    return False


__all__ = ["GateResult", "architecture_gate_command", "run_architecture_gates"]
