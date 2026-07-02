from pathlib import Path


def test_active_model_scoring_benchmark_uses_v1_package_entrypoint() -> None:
    text = Path("tools/benchmark_model_scoring.py").read_text(encoding="utf-8")

    assert "snookerhelp.qa.benchmark" in text
    assert "vision.config" not in text


def test_v1_benchmark_module_owns_model_scoring_command() -> None:
    text = Path("snookerhelp/qa/benchmark.py").read_text(encoding="utf-8")

    assert "benchmark_model_scoring_command" in text
    assert "collect_model_scoring_rows" in text
    assert "write_model_scoring_csv" in text


def test_legacy_model_scoring_benchmark_is_deleted() -> None:
    assert not Path("legacy/tools_v0/benchmark_model_scoring.py").exists()
