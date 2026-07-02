from pathlib import Path


def test_optional_manim_renderer_source_is_deleted_from_active_and_legacy_tools() -> None:
    assert not Path("tools/render_geometry_scenes.py").exists()
    assert not Path("legacy/tools_v0/render_geometry_scenes.py").exists()


def test_manim_docs_do_not_reference_deleted_renderer_command() -> None:
    text = Path("docs/manim_geometry_visualizations.md").read_text(encoding="utf-8")

    assert "legacy/tools_v0/render_geometry_scenes.py" not in text
    assert "python tools/render_geometry_scenes.py" not in text
