from __future__ import annotations

import argparse

from snookerhelp.qa.accuracy_cli import main as evaluate_accuracy_command
from snookerhelp.qa.cushion_touch_cli import main as evaluate_cushion_command
from snookerhelp.qa.refactor_gates import architecture_gate_command
from snookerhelp.qa.repeatability_cli import main as evaluate_repeatability_command
from snookerhelp.qa.samples import evaluate_samples_command
from snookerhelp.qa.spot_positions_cli import main as evaluate_spot_command
from snookerhelp.qa.touching_balls_cli import main as evaluate_touching_command


_KIND_TO_COMMAND = {
    "accuracy": evaluate_accuracy_command,
    "architecture": architecture_gate_command,
    "cushion": evaluate_cushion_command,
    "repeatability": evaluate_repeatability_command,
    "samples": evaluate_samples_command,
    "spot": evaluate_spot_command,
    "touching": evaluate_touching_command,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified v1 validation entrypoint. Remaining arguments are passed through.",
    )
    parser.add_argument("--kind", choices=sorted(_KIND_TO_COMMAND), required=True)
    args, passthrough = parser.parse_known_args(argv)
    return _KIND_TO_COMMAND[args.kind](passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
