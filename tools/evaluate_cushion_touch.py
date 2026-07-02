from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snookerhelp.qa.cushion_touch_cli import _touch_row, main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
