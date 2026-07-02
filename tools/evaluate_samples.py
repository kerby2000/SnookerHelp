from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snookerhelp.qa.samples import evaluate_samples_command


if __name__ == "__main__":
    raise SystemExit(evaluate_samples_command(sys.argv[1:]))
