from snookerhelp.qa.refactor_gates import run_architecture_gates
from snookerhelp.tools.validate import _KIND_TO_COMMAND


def test_architecture_gate_passes_current_refactor_boundary() -> None:
    results = run_architecture_gates(".")

    assert results
    assert all(result.ok for result in results), [
        result.to_dict() for result in results if not result.ok
    ]


def test_architecture_gate_is_available_from_unified_validate_command() -> None:
    assert "architecture" in _KIND_TO_COMMAND
