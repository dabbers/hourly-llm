"""Inline SVG charts for the report: no JavaScript, no external files, light and dark themes via
CSS custom properties defined by report/html.py. Marks follow the data-viz rules used in this
repo: bars at most 24 px wide with a 4 px rounded data end on a square baseline, 2 px gaps,
2 px lines, hairline solid gridlines, text in ink tokens never in series colors, a legend for two
or more series, values labelled only when the chart is small, native <title> tooltips."""

from __future__ import annotations

import math
from collections.abc import Callable
from html import escape

MAX_SERIES = 6
W, H = 760, 300
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 72, 20, 48, 48


def fmt_value(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}K"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a == 0:
        return "0"
    return f"{v:.3g}"


def fmt_usd(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if abs(v) >= 1:
        return f"${v:,.2f}"
    return f"${v:.3g}"


def nice_ticks(vmax: float, n: int = 5) -> list[float]:
    if not vmax or vmax <= 0 or math.isnan(vmax):
        return [0.0, 1.0]
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if vmax / step <= n:
            break
    top = math.ceil(vmax / step) * step
    k = round(top / step)
    return [round(i * step, 10) for i in range(k + 1)]


def _legend(names: list[str], x_right: int, y: int) -> str:
    if len(names) < 2:
        return ""
    items = []
    x = x_right
    for i, name in reversed(list(enumerate(names))):
        label = escape(name)
        width = 14 + 6.2 * len(name) + 16
        x -= width
        items.append(
            f'<rect x="{x:.0f}" y="{y - 9}" width="10" height="10" rx="2" style="fill:var(--s{i + 1})"/>'
            f'<text x="{x + 14:.0f}" y="{y}" class="lg">{label}</text>'
        )
    return "".join(items)


def _frame(title: str, subtitle: str, y_label: str, ticks: list[float], y_of: Callable[[float], float], fmt: Callable[[float], str]) -> list[str]:
    parts = [f'<text x="{M_LEFT}" y="20" class="title">{escape(title)}</text>']
    if subtitle:
        parts.append(f'<text x="{M_LEFT}" y="36" class="sub">{escape(subtitle)}</text>')
    for t in ticks:
        y = y_of(t)
        cls = "base" if t == ticks[0] else "grid"
        parts.append(f'<line x1="{M_LEFT}" x2="{W - M_RIGHT}" y1="{y:.1f}" y2="{y:.1f}" class="{cls}"/>')
        parts.append(f'<text x="{M_LEFT - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{escape(fmt(t))}</text>')
    if y_label:
        parts.append(f'<text transform="translate(12 {(M_TOP + H - M_BOTTOM) / 2:.0f}) rotate(-90)" class="axis" text-anchor="middle">{escape(y_label)}</text>')
    return parts


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    r = min(r, w / 2, h) if h > 0 else 0
    y0 = y + h
    return f"M{x:.1f},{y0:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} L{x + w:.1f},{y0:.1f} Z"


def bar_chart(
    title: str,
    categories: list[str],
    series: dict[str, list[float]],
    *,
    subtitle: str = "",
    y_label: str = "",
    fmt: Callable[[float], str] = fmt_value,
    label_values: bool | None = None,
) -> str:
    """Grouped columns: one band per category, one column per series."""
    names = list(series)[:MAX_SERIES]
    n, k = len(categories), len(names)
    if n == 0 or k == 0:
        return '<p class="muted">no data</p>'
    vals = [[(v if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0) for v in series[s]] for s in names]
    vmax = max((max(col) for col in vals if col), default=0.0)
    ticks = nice_ticks(vmax)
    top = ticks[-1] or 1.0
    plot_h = H - M_TOP - M_BOTTOM
    plot_w = W - M_LEFT - M_RIGHT

    def y_of(v: float) -> float:
        return M_TOP + plot_h - (v / top) * plot_h

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{escape(title)}">']
    parts += _frame(title, subtitle, y_label, ticks, y_of, fmt)
    band = plot_w / n
    gap = 2.0
    bar_w = min(24.0, max(3.0, (band * 0.8 - gap * (k - 1)) / k))
    group_w = bar_w * k + gap * (k - 1)
    if label_values is None:
        label_values = n * k <= 8  # more bars than that and the labels collide; tooltip + table carry them
    for ci, cat in enumerate(categories):
        x0 = M_LEFT + band * ci + (band - group_w) / 2
        for si, name in enumerate(names):
            v = vals[si][ci] if ci < len(vals[si]) else 0.0
            x = x0 + si * (bar_w + gap)
            y = y_of(v)
            h = M_TOP + plot_h - y
            tip = f"{cat} · {name}: {fmt(v)}" if k > 1 else f"{cat}: {fmt(v)}"
            parts.append(f'<path d="{_bar_path(x, y, bar_w, h)}" style="fill:var(--s{si + 1})"><title>{escape(tip)}</title></path>')
            if label_values and v > 0:
                parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" class="val" text-anchor="middle">{escape(fmt(v))}</text>')
        label = cat if len(cat) <= 16 else cat[:15] + "…"
        parts.append(f'<text x="{M_LEFT + band * ci + band / 2:.1f}" y="{H - M_BOTTOM + 18}" class="tick" text-anchor="middle"><title>{escape(cat)}</title>{escape(label)}</text>')
    parts.append(_legend(names, W - M_RIGHT, 20))
    parts.append("</svg>")
    return "".join(parts)


def line_chart(
    title: str,
    series: dict[str, tuple[list[float], list[float]]],
    *,
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    fmt: Callable[[float], str] = fmt_value,
    x_fmt: Callable[[float], str] = fmt_value,
    y_max: float | None = None,
) -> str:
    """Multi-series line chart over a numeric x (seconds). Series share one y axis and one unit."""
    names = [s for s in list(series)[:MAX_SERIES] if series[s][0]]
    if not names:
        return '<p class="muted">no data</p>'
    xmax = max(max(series[s][0]) for s in names) or 1.0
    vmax = y_max if y_max is not None else max(max(series[s][1]) for s in names)
    ticks = nice_ticks(vmax)
    top = ticks[-1] or 1.0
    xticks = nice_ticks(xmax, 6)
    xtop = xticks[-1] or 1.0
    plot_h = H - M_TOP - M_BOTTOM
    plot_w = W - M_LEFT - M_RIGHT

    def y_of(v: float) -> float:
        return M_TOP + plot_h - (min(v, top) / top) * plot_h

    def x_of(v: float) -> float:
        return M_LEFT + (v / xtop) * plot_w

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{escape(title)}">']
    parts += _frame(title, subtitle, y_label, ticks, y_of, fmt)
    for t in xticks:
        parts.append(f'<text x="{x_of(t):.1f}" y="{H - M_BOTTOM + 18}" class="tick" text-anchor="middle">{escape(x_fmt(t))}</text>')
    if x_label:
        parts.append(f'<text x="{M_LEFT + plot_w / 2:.0f}" y="{H - 10}" class="axis" text-anchor="middle">{escape(x_label)}</text>')
    for si, name in enumerate(names):
        xs, ys = series[name]
        pts = " ".join(f"{x_of(x):.1f},{y_of(y):.1f}" for x, y in zip(xs, ys, strict=False))
        tip = f"{name}: max {fmt(max(ys))}, mean {fmt(sum(ys) / len(ys))}"
        parts.append(f'<polyline points="{pts}" class="line" style="stroke:var(--s{si + 1})"><title>{escape(tip)}</title></polyline>')
        parts.append(f'<circle cx="{x_of(xs[-1]):.1f}" cy="{y_of(ys[-1]):.1f}" r="4" class="dot" style="fill:var(--s{si + 1})"/>')
    parts.append(_legend(names, W - M_RIGHT, 20))
    parts.append("</svg>")
    return "".join(parts)


def scatter_chart(
    title: str,
    points: dict[str, tuple[list[float], list[float]]],
    *,
    lines: dict[str, tuple[list[float], list[float]]] | None = None,
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    fmt: Callable[[float], str] = fmt_value,
    x_fmt: Callable[[float], str] = fmt_value,
    y_ref: float | None = None,
    y_ref_label: str = "",
) -> str:
    """Scatter of (x, y) points per series, each series optionally with a line in its colour (a
    fit, or the path through its means), and an optional dashed horizontal reference line. Both
    axes start at 0 and share one unit per axis; NaN points are skipped, y is clipped to the axis."""
    lines = lines or {}
    names = list(dict.fromkeys([*points, *lines]))[:MAX_SERIES]

    def clean(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
        out = []
        for x, y in zip(xs, ys, strict=False):
            if x is None or y is None or math.isnan(float(x)) or math.isnan(float(y)):
                continue
            out.append((float(x), float(y)))
        return out

    pts = {n: clean(*points[n]) for n in names if n in points}
    lns = {n: clean(*lines[n]) for n in names if n in lines}
    everything = [p for v in [*pts.values(), *lns.values()] for p in v]
    if not everything:
        return '<p class="muted">no data</p>'
    xmax = max(p[0] for p in everything)
    ymax = max([p[1] for p in everything] + ([y_ref] if y_ref is not None else []))
    ticks = nice_ticks(ymax)
    top = ticks[-1] or 1.0
    xticks = nice_ticks(xmax, 6)
    xtop = xticks[-1] or 1.0
    plot_h = H - M_TOP - M_BOTTOM
    plot_w = W - M_LEFT - M_RIGHT

    def y_of(v: float) -> float:
        return M_TOP + plot_h - (min(max(v, 0.0), top) / top) * plot_h

    def x_of(v: float) -> float:
        return M_LEFT + (min(max(v, 0.0), xtop) / xtop) * plot_w

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{escape(title)}">']
    parts += _frame(title, subtitle, y_label, ticks, y_of, fmt)
    for t in xticks:
        parts.append(f'<text x="{x_of(t):.1f}" y="{H - M_BOTTOM + 18}" class="tick" text-anchor="middle">{escape(x_fmt(t))}</text>')
    if x_label:
        parts.append(f'<text x="{M_LEFT + plot_w / 2:.0f}" y="{H - 10}" class="axis" text-anchor="middle">{escape(x_label)}</text>')
    if y_ref is not None:
        y = y_of(y_ref)
        parts.append(f'<line x1="{M_LEFT}" x2="{W - M_RIGHT}" y1="{y:.1f}" y2="{y:.1f}" class="base" stroke-dasharray="4 3"/>')
        if y_ref_label:
            parts.append(f'<text x="{W - M_RIGHT}" y="{y - 4:.1f}" class="sub" text-anchor="end">{escape(y_ref_label)}</text>')
    for si, name in enumerate(names):
        if lns.get(name):
            seq = sorted(lns[name])
            path = " ".join(f"{x_of(x):.1f},{y_of(y):.1f}" for x, y in seq)
            parts.append(f'<polyline points="{path}" class="line" style="stroke:var(--s{si + 1})"><title>{escape(name)}</title></polyline>')
        for x, y in pts.get(name, []):
            tip = f"{name}: {x_fmt(x)}, {fmt(y)}"
            parts.append(f'<circle cx="{x_of(x):.1f}" cy="{y_of(y):.1f}" r="4" class="dot" style="fill:var(--s{si + 1})"><title>{escape(tip)}</title></circle>')
    parts.append(_legend(names, W - M_RIGHT, 20))
    parts.append("</svg>")
    return "".join(parts)
