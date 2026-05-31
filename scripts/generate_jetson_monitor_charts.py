from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


NUMERIC_COLUMNS = (
    "ram_used_mb",
    "ram_total_mb",
    "swap_used_mb",
    "swap_total_mb",
    "cpu_avg_pct",
    "cpu_max_freq_mhz",
    "emc_pct",
    "emc_freq_mhz",
    "gr3d_pct",
    "gr3d_freq_mhz",
    "max_temp_c",
    "throttling_suspected",
)
BACKGROUND = "#10141c"
FOREGROUND = "#e8edf2"
MUTED = "#9aa6b2"
GRID = "#343b49"
RAM = "#4cc9f0"
SWAP = "#f72585"
CPU = "#80ed99"
GPU = "#f9c74f"
TEMP = "#ff595e"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate charts and summary stats from Jetson tegrastats CSV.")
    parser.add_argument("--input", default="logs/jetson_telemetry.csv", help="Input CSV from src.tools.jetson_monitor.")
    parser.add_argument("--output-dir", default="docs/imgs", help="Directory for generated PNG charts.")
    parser.add_argument(
        "--summary-output",
        default="logs/jetson_telemetry_summary.json",
        help="Output JSON summary under logs/.",
    )
    return parser


def load_rows(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        raise FileNotFoundError(f"Jetson telemetry CSV not found: {path}")

    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Jetson telemetry CSV has no header: {path}")

        missing = [column for column in ("timestamp_utc", "raw") if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Jetson telemetry CSV missing required columns: {', '.join(missing)}")

        for raw_row in reader:
            row: dict[str, float | str] = {
                "timestamp_utc": raw_row.get("timestamp_utc", ""),
                "raw": raw_row.get("raw", ""),
            }
            for column in NUMERIC_COLUMNS:
                value = raw_row.get(column, "")
                row[column] = float(value) if value not in ("", None) else float("nan")
            rows.append(row)

    if not rows:
        raise ValueError(f"Jetson telemetry CSV has no rows: {path}")
    return rows


def finite_values(rows: list[dict[str, float | str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, float) and value == value:
            values.append(value)
    return values


def build_summary(rows: list[dict[str, float | str]]) -> dict[str, object]:
    stats: dict[str, object] = {"samples": len(rows)}
    for column in NUMERIC_COLUMNS:
        values = finite_values(rows, column)
        if values:
            stats[column] = {
                "min": min(values),
                "max": max(values),
                "mean": round(mean(values), 3),
            }
    throttle_values = finite_values(rows, "throttling_suspected")
    stats["throttling_samples"] = int(sum(1 for value in throttle_values if value >= 1.0))
    stats["first_timestamp_utc"] = rows[0].get("timestamp_utc", "")
    stats["last_timestamp_utc"] = rows[-1].get("timestamp_utc", "")
    return stats


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    *,
    size: int,
    fill: str = FOREGROUND,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill, anchor=anchor)


def draw_line_chart(
    rows: list[dict[str, float | str]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    series: list[tuple[str, str, str]],
    y_label: str,
) -> None:
    image = Image.new("RGB", (1800, 1040), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (80, 60), title, size=46, bold=True)
    draw_text(draw, (80, 120), subtitle, size=26, fill=MUTED)

    left, top, right, bottom = 130, 220, 1680, 850
    all_values: list[float] = []
    for column, _, _ in series:
        all_values.extend(finite_values(rows, column))
    if not all_values:
        raise ValueError(f"No finite values for chart: {title}")

    y_min = min(0.0, min(all_values))
    y_max = max(all_values) * 1.12
    if y_max <= y_min:
        y_max = y_min + 1.0

    for tick_index in range(6):
        value = y_min + (y_max - y_min) * tick_index / 5
        y = bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw_text(draw, (left - 16, y), f"{value:.0f}", size=22, fill=MUTED, anchor="rm")

    draw.line((left, top, left, bottom), fill=FOREGROUND, width=2)
    draw.line((left, bottom, right, bottom), fill=FOREGROUND, width=2)

    sample_count = len(rows)
    x_step = (right - left) / max(sample_count - 1, 1)
    for column, label, color in series:
        points: list[tuple[float, float]] = []
        for index, row in enumerate(rows):
            value = row.get(column)
            if not isinstance(value, float) or value != value:
                continue
            x = left + x_step * index
            y = bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=4, joint="curve")
        for point in points[:: max(1, len(points) // 30)]:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)

    legend_x = 130
    for _, label, color in series:
        draw.rounded_rectangle((legend_x, 910, legend_x + 34, 944), radius=4, fill=color)
        draw_text(draw, (legend_x + 48, 906), label, size=25, bold=True)
        legend_x += 320

    draw_text(draw, ((left + right) / 2, 980), "Sample index", size=24, fill=MUTED, anchor="mm")
    draw_text(draw, (42, (top + bottom) / 2), y_label, size=24, fill=MUTED, anchor="mm")
    image.save(output_path, quality=95)


def write_summary(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_report(summary: dict[str, object], output_dir: Path, summary_path: Path) -> None:
    print("\n## Jetson Telemetry Analytics\n")
    print(f"- Samples analyzed: {summary['samples']}")
    for column in ("ram_used_mb", "swap_used_mb", "gr3d_pct", "cpu_avg_pct", "max_temp_c"):
        value = summary.get(column)
        if isinstance(value, dict):
            print(f"- {column}: mean={value['mean']} max={value['max']}")
    print(f"- Throttling suspected samples: {summary.get('throttling_samples', 0)}")
    print("\nGenerated artifacts:")
    print(f"- `{summary_path}`")
    print(f"- `{output_dir / 'jetson_memory_timeseries.png'}`")
    print(f"- `{output_dir / 'jetson_compute_thermal_timeseries.png'}`")


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    summary = build_summary(rows)
    write_summary(summary, summary_path)
    draw_line_chart(
        rows,
        output_dir / "jetson_memory_timeseries.png",
        title="Jetson UMA Memory Pressure",
        subtitle=f"{len(rows)} tegrastats samples | input: {input_path}",
        series=[
            ("ram_used_mb", "RAM used MB", RAM),
            ("swap_used_mb", "SWAP used MB", SWAP),
        ],
        y_label="MB",
    )
    draw_line_chart(
        rows,
        output_dir / "jetson_compute_thermal_timeseries.png",
        title="Jetson Compute and Thermal Profile",
        subtitle=f"{len(rows)} tegrastats samples | throttle samples: {summary.get('throttling_samples', 0)}",
        series=[
            ("gr3d_pct", "GPU GR3D %", GPU),
            ("cpu_avg_pct", "CPU avg %", CPU),
            ("max_temp_c", "Max temp C", TEMP),
        ],
        y_label="% / C",
    )
    print_report(summary, output_dir, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
