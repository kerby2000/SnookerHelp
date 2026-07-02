from pathlib import Path


def test_active_sample_evaluator_uses_v1_package_entrypoint() -> None:
    text = Path("tools/evaluate_samples.py").read_text(encoding="utf-8")

    assert "snookerhelp.qa.samples" in text
    assert "vision.state_estimator" not in text


def test_unified_validate_samples_uses_package_command() -> None:
    text = Path("snookerhelp/tools/validate.py").read_text(encoding="utf-8")

    assert "evaluate_samples_command" in text
    assert '"samples": evaluate_samples_command' in text
    assert "runpy" not in text


def test_legacy_sample_evaluator_is_available() -> None:
    assert not Path("legacy/tools_v0/evaluate_samples.py").exists()


def test_active_physical_validation_commands_use_v1_package_entrypoints() -> None:
    command_to_module = {
        "tools/evaluate_accuracy.py": "snookerhelp.qa.accuracy_cli",
        "tools/evaluate_cushion_touch.py": "snookerhelp.qa.cushion_touch_cli",
        "tools/evaluate_repeatability.py": "snookerhelp.qa.repeatability_cli",
        "tools/evaluate_spot_positions.py": "snookerhelp.qa.spot_positions_cli",
        "tools/evaluate_touching_balls.py": "snookerhelp.qa.touching_balls_cli",
    }

    for command_path, module_name in command_to_module.items():
        text = Path(command_path).read_text(encoding="utf-8")
        assert module_name in text
        assert "vision." not in text


def test_v1_validation_dispatcher_has_all_validation_kinds() -> None:
    text = Path("snookerhelp/tools/validate.py").read_text(encoding="utf-8")

    for kind in [
        "accuracy",
        "architecture",
        "cushion",
        "repeatability",
        "samples",
        "spot",
        "touching",
    ]:
        assert f'"{kind}"' in text


def test_legacy_physical_validation_commands_are_deleted() -> None:
    for path in [
        Path("legacy/tools_v0/evaluate_accuracy.py"),
        Path("legacy/tools_v0/evaluate_cushion_touch.py"),
        Path("legacy/tools_v0/evaluate_repeatability.py"),
        Path("legacy/tools_v0/evaluate_spot_positions.py"),
        Path("legacy/tools_v0/evaluate_touching_balls.py"),
    ]:
        assert not path.exists()
