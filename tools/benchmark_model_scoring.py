from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snookerhelp.qa.benchmark import benchmark_model_scoring_command


if __name__ == "__main__":
    raise SystemExit(benchmark_model_scoring_command(sys.argv[1:]))
