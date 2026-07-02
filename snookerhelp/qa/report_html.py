from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


PANEL_FILES = [
    ("01_source_detection.png", "1. Source detection"),
    ("02_warped_detection.png", "2. Warped detection"),
    ("03_source_zoom_grid.png", "3. Source zoom grid"),
    ("04_geometry_selected_ball.png", "4. Geometry selected ball"),
    ("05_error_comparison.png", "5. Coordinate comparison"),
    ("06_physical_validation.png", "6. Physical validation"),
    ("07_pipeline_summary.png", "7. Pipeline summary"),
]


def write_report_html(
    output_path: str | Path,
    report: dict[str, Any],
) -> None:
    Path(output_path).write_text(render_report_html(report), encoding="utf-8")


def render_report_html(report: dict[str, Any]) -> str:
    image_name = str(report.get("image") or "")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    validation = (
        report.get("physical_validation")
        if isinstance(report.get("physical_validation"), dict)
        else {}
    )
    selected = (
        report.get("selected_ball")
        if isinstance(report.get("selected_ball"), dict)
        else {}
    )
    cards = _summary_cards(summary, validation, selected)
    panels = _panel_sections(report)
    coordinate_rows = _coordinate_rows(report.get("coordinate_rows"))
    validation_rows = _validation_rows(validation.get("rows"))
    review_rows = _review_rows(report.get("ball_review_rows"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SnookerHelp visual debug report - {_e(image_name)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0d141b;
      --panel: #121d27;
      --panel-2: #172532;
      --line: #2c4153;
      --text: #e8f0f8;
      --muted: #9db0c2;
      --good: #43d17d;
      --warn: #ffcc66;
      --bad: #ff7070;
      --link: #7ecbff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 24px 32px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, #101c26, #0b1118);
    }}
    main {{
      width: min(1560px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 24px 0 60px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 22px; }}
    a {{ color: var(--link); }}
    .muted {{ color: var(--muted); }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      max-width: 1100px;
      line-height: 1.45;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin: 20px 0 28px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 14px 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 5px;
    }}
    .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .panel {{
      margin: 22px 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 16px;
    }}
    .panel img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
      background: #fff;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      margin: 16px 0 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: #c5d6e6;
      background: var(--panel-2);
      position: sticky;
      top: 0;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .status-pass {{ color: var(--good); font-weight: 700; }}
    .status-warn {{ color: var(--warn); font-weight: 700; }}
    .status-fail {{ color: var(--bad); font-weight: 700; }}
    code {{
      color: #9bdcff;
      background: rgba(255,255,255,0.06);
      border-radius: 4px;
      padding: 1px 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>SnookerHelp visual debug report</h1>
    <div class="muted">{_e(image_name)}</div>
    <p class="note">
      This is a static QA artifact. Use the v1 review UI for OK/NOK decisions,
      missing balls, and manual corrections:
      <code>python tools/review_reports.py --reports outputs/reports</code>.
    </p>
  </header>
  <main>
    <section class="cards">{cards}</section>
    <section class="panel">
      <h2>Report data</h2>
      <p class="muted">Raw structured data: <a href="report.json">report.json</a></p>
    </section>
    {panels}
    <section>
      <h2>Coordinate comparison</h2>
      {coordinate_rows}
    </section>
    <section>
      <h2>Physical validation</h2>
      {validation_rows}
    </section>
    <section>
      <h2>Ball review evidence summary</h2>
      {review_rows}
    </section>
  </main>
</body>
</html>
"""


def _summary_cards(
    summary: dict[str, Any],
    validation: dict[str, Any],
    selected: dict[str, Any],
) -> str:
    detection = summary.get("detection") if isinstance(summary.get("detection"), dict) else {}
    camera = summary.get("camera_model") if isinstance(summary.get("camera_model"), dict) else {}
    validation_summary = (
        validation.get("summary") if isinstance(validation.get("summary"), dict) else {}
    )
    cards = [
        ("Detected balls", summary.get("ball_count", detection.get("ball_count", "n/a"))),
        ("Source fit success", _fraction(summary.get("source_refinement_success_fraction"))),
        ("Camera model", _display_token(camera.get("mode", "n/a"))),
        ("Selected ball", _selected_label(selected)),
        ("Validation rows", validation.get("row_count", len(validation.get("rows") or []))),
        ("Validation mean abs error", _mm(validation_summary.get("mean"))),
        ("Validation max abs error", _mm(validation_summary.get("max"))),
        ("Raw candidates", detection.get("raw_candidate_count", "n/a")),
    ]
    return "\n".join(
        f"""
        <div class="card">
          <div class="label">{_e(label)}</div>
          <div class="value">{_e(value)}</div>
        </div>
        """
        for label, value in cards
    )


def _panel_sections(report: dict[str, Any]) -> str:
    panel_names = set(report.get("panels") or [])
    sections = []
    for file_name, title in PANEL_FILES:
        if panel_names and file_name not in panel_names:
            continue
        sections.append(
            f"""
            <section class="panel">
              <h2>{_e(title)}</h2>
              <a href="{_e(file_name)}"><img src="{_e(file_name)}" alt="{_e(title)}"></a>
            </section>
            """
        )
    return "\n".join(sections)


def _coordinate_rows(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return '<p class="muted">No coordinate rows available.</p>'
    columns = [
        ("id", "ID"),
        ("label", "Label"),
        ("warped_x_mm", "Warp X"),
        ("warped_y_mm", "Warp Y"),
        ("source_x_mm", "Source X"),
        ("source_y_mm", "Source Y"),
        ("warped_to_source_delta_mm", "Delta"),
        ("source_fit_residual_px", "Residual px"),
    ]
    return _table(columns, rows, limit=80)


def _validation_rows(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return '<p class="muted">No physical validation rows available.</p>'
    columns = [
        ("index", "#"),
        ("kind", "Kind"),
        ("item", "Item"),
        ("measured_mm", "Measured"),
        ("expected_mm", "Expected"),
        ("signed_error_mm", "Signed error"),
        ("abs_error_mm", "Abs error"),
        ("status", "Status"),
        ("region", "Region"),
    ]
    return _table(columns, rows, limit=120)


def _review_rows(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return '<p class="muted">No ball review rows available.</p>'
    columns = [
        ("id", "ID"),
        ("label", "Label"),
        ("source_refinement_success", "Source fit"),
        ("source_fit_residual_px", "Residual px"),
        ("source_fit_point_count", "Points"),
        ("source_radius_px", "Radius px"),
        ("rough_to_refined_shift_px", "Shift px"),
        ("fit_explanation", "Evidence note"),
    ]
    return _table(columns, rows, limit=120)


def _table(
    columns: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> str:
    visible = rows[:limit]
    header = "".join(f"<th>{_e(title)}</th>" for _, title in columns)
    body = "\n".join(
        "<tr>"
        + "".join(
            f"<td{_status_class(key, row.get(key))}>{_e(_format_cell(row.get(key)))}</td>"
            for key, _ in columns
        )
        + "</tr>"
        for row in visible
        if isinstance(row, dict)
    )
    note = (
        f'<p class="muted">Showing first {limit} of {len(rows)} rows.</p>'
        if len(rows) > limit
        else ""
    )
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>{note}'


def _status_class(key: str, value: Any) -> str:
    if key != "status":
        return ""
    token = str(value or "").lower()
    if token in {"pass", "ok", "accepted"}:
        return ' class="status-pass"'
    if token in {"warn", "review", "needs_review"}:
        return ' class="status-warn"'
    if token in {"fail", "nok", "rejected"}:
        return ' class="status-fail"'
    return ""


def _format_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ", ".join(_format_cell(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_format_cell(val)}" for key, val in value.items())
    return _display_token(str(value))


def _display_token(value: Any) -> str:
    text = str(value)
    replacements = {
        "manual_homography": "table-corner bootstrap",
        "source_refined_center_px": "final source-pixel estimate",
        "circle_radial": "circle diagnostic",
        "fallback_radial": "fallback estimate",
        "candidate_c": "image evidence",
        "candidate_d": "physical model",
        "candidate_b": "segmentation diagnostic",
        "candidate_a": "circle diagnostic",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ")


def _selected_label(selected: dict[str, Any]) -> str:
    ball_id = selected.get("id")
    label = selected.get("label")
    if ball_id is None:
        return "n/a"
    return f"#{ball_id} {label or ''}".strip()


def _fraction(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _mm(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f} mm"
    except (TypeError, ValueError):
        return str(value)


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


__all__ = ["PANEL_FILES", "render_report_html", "write_report_html"]
