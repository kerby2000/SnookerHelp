from __future__ import annotations

from pathlib import Path
from typing import Iterator

from snookerhelp.core.schema import TableState
from snookerhelp.review.evidence_export import table_state_from_report


def iter_report_json_files(reports_root: str | Path = "outputs/reports") -> Iterator[Path]:
    yield from sorted(Path(reports_root).glob("*/report.json"))


def load_table_state(report_json: str | Path) -> TableState:
    return table_state_from_report(report_json)


def load_dataset_table_states(reports_root: str | Path = "outputs/reports") -> list[TableState]:
    return [load_table_state(path) for path in iter_report_json_files(reports_root)]


__all__ = [
    "iter_report_json_files",
    "load_dataset_table_states",
    "load_table_state",
]
