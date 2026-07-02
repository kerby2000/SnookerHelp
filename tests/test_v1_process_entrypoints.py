from pathlib import Path


def test_active_process_single_command_uses_v1_package_entrypoint() -> None:
    text = Path("tools/process_single_image.py").read_text(encoding="utf-8")

    assert "snookerhelp.tools.process_image" in text
    assert "vision.state_estimator" not in text


def test_active_process_latest_command_uses_v1_package_entrypoint() -> None:
    text = Path("tools/process_latest_image.py").read_text(encoding="utf-8")

    assert "snookerhelp.tools.process_image" in text
    assert "vision.state_estimator" not in text


def test_v1_process_entrypoint_does_not_run_root_scripts() -> None:
    text = Path("snookerhelp/tools/process_image.py").read_text(encoding="utf-8")

    assert "runpy" not in text
    assert "process_single_image.py" not in text
    assert "process_latest_image.py" not in text


def test_legacy_process_commands_are_deleted() -> None:
    assert not Path("legacy/tools_v0/process_single_image.py").exists()
    assert not Path("legacy/tools_v0/process_latest_image.py").exists()
