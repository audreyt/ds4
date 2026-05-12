#!/usr/bin/env python3
"""Generate an SVG throughput graph from a ds4-bench CSV file.

The benchmark intentionally reports instantaneous throughput at each context
frontier. This script keeps the plot equally direct: one line for incremental
prefill t/s, one line for greedy generation t/s, and separate y axes because
the two values live on very different scales.
"""

import argparse
import csv
import html
import math
from pathlib import Path


PREFILL_COLOR = "#2563eb"
GEN_COLOR = "#dc2626"
TEXT_COLOR = "#1f2933"
MUTED_COLOR = "#64748b"
GRID_COLOR = "#e2e8f0"
AXIS_COLOR = "#334155"


def nice_ceil(value):
    """Round a positive axis maximum up to a human-friendly tick boundary."""
    if value <= 0:
        return 1.0

    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for step in (1, 2, 2.5, 3, 4, 5, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def nice_step(span, target_ticks):
    """Return a readable tick spacing close to span / target_ticks."""
    if span <= 0:
        return 1.0

    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    for step in (1, 2, 2.5, 5, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def fmt_tick(value):
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def read_points(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"ctx_tokens", "prefill_tps", "gen_tps"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise SystemExit(f"{path}: missing CSV column(s): {missing_list}")

        for row in reader:
            rows.append(
                (
                    int(row["ctx_tokens"]),
                    float(row["prefill_tps"]),
                    float(row["gen_tps"]),
                )
            )

    if len(rows) < 2:
        raise SystemExit(f"{path}: need at least two data rows")

    rows.sort(key=lambda item: item[0])
    return rows


def derive_title(csv_path):
    words = csv_path.stem.replace("_", " ").replace("-", " ").split()
    return " ".join(word.upper() if word[0:1].lower() == "m" and word[1:].isdigit() else word.capitalize() for word in words) + " t/s"


def points_to_polyline(points, x_min, x_max, y_max, plot):
    left, top, width, height = plot

    def project(point):
        x, y = point
        px = left + (x - x_min) / (x_max - x_min) * width
        py = top + height - y / y_max * height
        return f"{px:.2f},{py:.2f}"

    return " ".join(project(point) for point in points)


def render_svg(rows, title, width, height):
    margin_left = 82
    margin_right = 82
    margin_top = 66
    margin_bottom = 72
    plot = (
        margin_left,
        margin_top,
        width - margin_left - margin_right,
        height - margin_top - margin_bottom,
    )
    left, top, plot_width, plot_height = plot
    right = left + plot_width
    bottom = top + plot_height

    ctx_values = [row[0] for row in rows]
    prefill_values = [row[1] for row in rows]
    gen_values = [row[2] for row in rows]
    x_min = 0
    x_max = max(ctx_values)
    prefill_max = nice_ceil(max(prefill_values) * 1.05)
    gen_max = nice_ceil(max(gen_values) * 1.05)

    x_step = nice_step(x_max - x_min, 6)
    x_ticks = []
    tick = math.ceil(x_min / x_step) * x_step
    while tick <= x_max:
        x_ticks.append(tick)
        tick += x_step

    prefill_step = nice_step(prefill_max, 5)
    gen_step = nice_step(gen_max, 5)
    prefill_ticks = [tick for tick in frange(0, prefill_max, prefill_step)]
    gen_ticks = [tick for tick in frange(0, gen_max, gen_step)]

    prefill_points = [(row[0], row[1]) for row in rows]
    gen_points = [(row[0], row[2]) for row in rows]
    prefill_poly = points_to_polyline(prefill_points, x_min, x_max, prefill_max, plot)
    gen_poly = points_to_polyline(gen_points, x_min, x_max, gen_max, plot)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".title { font-size: 26px; font-weight: 700; fill: #1f2933; }",
        ".axis-label { font-size: 14px; font-weight: 600; fill: #334155; }",
        ".tick { font-size: 12px; fill: #64748b; }",
        ".legend { font-size: 13px; font-weight: 600; fill: #1f2933; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>',
    ]

    # Horizontal grid and left-axis labels use the prefill scale.
    for tick in prefill_ticks:
        y = bottom - tick / prefill_max * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{fmt_tick(tick)}</text>')

    # Right-axis labels use the generation scale.
    for tick in gen_ticks:
        y = bottom - tick / gen_max * plot_height
        parts.append(f'<text class="tick" x="{right + 12}" y="{y + 4:.2f}" text-anchor="start">{fmt_tick(tick)}</text>')

    for tick in x_ticks:
        x = left + (tick - x_min) / (x_max - x_min) * plot_width
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 24}" text-anchor="middle">{fmt_tick(tick)}</text>')

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
            f'<line x1="{right}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
            f'<text class="axis-label" x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle">ctx size</text>',
            f'<text class="axis-label" x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})">prefill t/s</text>',
            f'<text class="axis-label" x="{width - 22}" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(90 {width - 22} {top + plot_height / 2:.1f})">generation t/s</text>',
            f'<polyline fill="none" stroke="{PREFILL_COLOR}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{prefill_poly}"/>',
            f'<polyline fill="none" stroke="{GEN_COLOR}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{gen_poly}"/>',
        ]
    )

    legend_x = right - 170
    legend_y = top + 18
    parts.extend(
        [
            f'<rect x="{legend_x - 14}" y="{legend_y - 18}" width="176" height="62" rx="6" fill="#ffffff" stroke="#cbd5e1"/>',
            f'<rect x="{legend_x}" y="{legend_y - 6}" width="12" height="12" fill="{PREFILL_COLOR}"/>',
            f'<text class="legend" x="{legend_x + 22}" y="{legend_y + 5}">prefill</text>',
            f'<rect x="{legend_x}" y="{legend_y + 20}" width="12" height="12" fill="{GEN_COLOR}"/>',
            f'<text class="legend" x="{legend_x + 22}" y="{legend_y + 31}">generation</text>',
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def frange(start, stop, step):
    value = start
    # A small epsilon keeps exact decimal steps from losing their final tick.
    while value <= stop + step * 0.001:
        yield round(value, 10)
        value += step


def main():
    parser = argparse.ArgumentParser(description="Plot ds4-bench throughput CSV data as SVG.")
    parser.add_argument("csv", type=Path, help="input CSV produced by ds4-bench")
    parser.add_argument("-o", "--output", type=Path, help="output SVG path")
    parser.add_argument("--title", help="graph title; defaults to a title derived from the CSV name")
    parser.add_argument("--width", type=int, default=960, help="SVG width in pixels")
    parser.add_argument("--height", type=int, default=540, help="SVG height in pixels")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.csv.with_name(f"{args.csv.stem}_ts.svg")

    rows = read_points(args.csv)
    title = args.title or derive_title(args.csv)
    output.write_text(render_svg(rows, title, args.width, args.height), encoding="utf-8")


if __name__ == "__main__":
    main()
