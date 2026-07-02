import ast
from pathlib import Path


def test_active_v1_code_does_not_import_vision_package() -> None:
    for root in [Path("snookerhelp"), Path("tools")]:
        for path in root.rglob("*.py"):
            assert not _imports_package(path, "vision"), path


def test_active_v1_code_does_not_import_legacy_package() -> None:
    for root in [Path("snookerhelp"), Path("tools")]:
        for path in root.rglob("*.py"):
            assert not _imports_package(path, "legacy"), path


def test_vision_compatibility_package_is_deleted() -> None:
    assert not Path("vision").exists()


def _imports_package(path: Path, blocked_root: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] == blocked_root:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".", 1)[0] == blocked_root:
                return True
    return False
