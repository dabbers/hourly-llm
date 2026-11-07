"""HTML assembly for the report: one self-contained page, system font, light and dark themes."""

from __future__ import annotations

import math
from collections.abc import Callable
from html import escape

import pandas as pd

CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --base: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --base: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --base: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 36px 0 8px; }
h3 { font-size: 14px; margin: 20px 0 6px; color: var(--ink2); }
p, li { color: var(--ink2); } p.lead { color: var(--ink); }
.muted { color: var(--muted); }
nav a { margin-right: 14px; color: var(--s1); text-decoration: none; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin: 10px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }
.chart { width: 100%; height: auto; display: block; }
.chart .title { font-size: 13px; font-weight: 600; fill: var(--ink); }
.chart .sub { font-size: 11px; fill: var(--ink2); }
.chart .tick, .chart .lg, .chart .axis { font-size: 11px; fill: var(--muted); }
.chart .lg { fill: var(--ink2); }
.chart .val { font-size: 10px; fill: var(--ink2); }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .base { stroke: var(--base); stroke-width: 1; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .dot { stroke: var(--surface); stroke-width: 2; }
.tblwrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { padding: 5px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
td.l, th.l { text-align: left; }
td { font-variant-numeric: tabular-nums; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; }
dt { color: var(--ink); font-weight: 600; } dd { margin: 0; color: var(--ink2); }
code { font-size: 12px; background: var(--grid); padding: 1px 4px; border-radius: 3px; }
"""


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_int(v) -> str:
    return "–" if _is_nan(v) else f"{round(v):,}"


def fmt_f(digits: int) -> Callable:
    return lambda v: "–" if _is_nan(v) else f"{v:,.{digits}f}"


def fmt_pct(v) -> str:
    return "–" if _is_nan(v) else f"{v * 100:.1f}%"


def fmt_usd(v) -> str:
    if _is_nan(v):
        return "–"
    return f"${v:,.4f}" if abs(v) < 1 else f"${v:,.2f}"


def fmt_text(v) -> str:
    return "–" if _is_nan(v) else escape(str(v))


def table(df: pd.DataFrame, columns: list[tuple[str, str, Callable]], *, left: int = 1) -> str:
    """columns: (dataframe column, header label, formatter). The first `left` columns are text."""
    if df is None or df.empty:
        return '<p class="muted">no rows</p>'
    head = "".join(f'<th class="{"l" if i < left else ""}">{escape(label)}</th>' for i, (_, label, _) in enumerate(columns))
    body = []
    for _, row in df.iterrows():
        cells = []
        for i, (col, _, fmt) in enumerate(columns):
            v = row.get(col)
            cells.append(f'<td class="{"l" if i < left else ""}">{fmt(v)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def page(title: str, intro: str, sections: list[tuple[str, str, str]]) -> str:
    """sections: (anchor id, heading, html body)."""
    nav = "".join(f'<a href="#{sid}">{escape(h)}</a>' for sid, h, _ in sections)
    body = "".join(f'<section id="{sid}"><h2>{escape(h)}</h2>{html}</section>' for sid, h, html in sections)
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{escape(title)}</title>"
        f"<style>{CSS}</style></head><body><main><h1>{escape(title)}</h1>{intro}<nav>{nav}</nav>{body}</main></body></html>"
    )
