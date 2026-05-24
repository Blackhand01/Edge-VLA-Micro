from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


LATENCY_COLUMNS = ("audio_ms", "vision_ms", "ttft_ms", "decode_time_ms", "safety_ms")
REQUIRED_COLUMNS = (*LATENCY_COLUMNS, "tps")
PIE_LABELS = (
    "ASR (Whisper)",
    "Vision Capture",
    "VLM TTFT (Prompt Eval)",
    "VLM Decode",
    "Safety Guardrail",
)
COLORS = ("#4cc9f0", "#80ed99", "#f72585", "#f9c74f", "#adb5bd")
BACKGROUND = "#10141c"
FOREGROUND = "#e8edf2"
MUTED = "#9aa6b2"
GRID = "#343b49"
ACCENT = "#f72585"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate industrial whitepaper charts from Edge-VLA performance telemetry."
    )
    parser.add_argument("--input", default="logs/performance.csv", help="Input performance CSV path.")
    parser.add_argument("--output-dir", default="docs", help="Directory where chart PNG files are written.")
    return parser


def load_performance_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Performance CSV not found: {path}")

    rows: list[dict[str, float]] = []
    skipped_rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Performance CSV has no header: {path}")
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Performance CSV missing required columns: {', '.join(missing_columns)}")
        for row in reader:
            try:
                rows.append({column: float(row[column]) for column in REQUIRED_COLUMNS})
            except (KeyError, TypeError, ValueError):
                skipped_rows += 1

    if not rows:
        raise ValueError(f"Performance CSV has no usable rows: {path}")
    if skipped_rows:
        print(f"Skipped {skipped_rows} malformed telemetry rows.")
    return rows


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
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


def calculate_latency_means(rows: list[dict[str, float]]) -> dict[str, float]:
    return {column: mean(row[column] for row in rows) for column in LATENCY_COLUMNS}


def generate_latency_pie_chart(means: dict[str, float], output_path: Path, runs_analyzed: int) -> None:
    values = [float(means[column]) for column in LATENCY_COLUMNS]
    total_latency = sum(values)
    if total_latency <= 0.0:
        raise ValueError("No positive latency values found for latency pie chart.")

    bottleneck_pct = float((means["audio_ms"] + means["ttft_ms"]) / total_latency * 100.0)
    image = Image.new("RGB", (1800, 1260), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (90, 70), "Average Control-Loop Latency Breakdown", size=48, bold=True)
    draw_text(
        draw,
        (90, 130),
        f"{runs_analyzed} runs | ASR + VLM TTFT bottleneck: {bottleneck_pct:.1f}% of profiled latency",
        size=28,
        fill="#f9c74f",
        bold=True,
    )

    pie_box = (170, 240, 1030, 1100)
    center = (600, 670)
    start_angle = -90.0
    for label, value, color in zip(PIE_LABELS, values, COLORS):
        extent = value / total_latency * 360.0
        end_angle = start_angle + extent
        draw.pieslice(pie_box, start=start_angle, end=end_angle, fill=color, outline=BACKGROUND, width=3)
        mid_angle = math.radians(start_angle + extent / 2.0)
        pct = value / total_latency * 100.0
        if pct >= 2.0:
            x = center[0] + math.cos(mid_angle) * 265
            y = center[1] + math.sin(mid_angle) * 265
            draw_text(draw, (x, y), f"{pct:.1f}%", size=26, fill=BACKGROUND, bold=True, anchor="mm")
        start_angle = end_angle

    legend_x = 1110
    for index, (label, value, color) in enumerate(zip(PIE_LABELS, values, COLORS)):
        y = 270 + index * 130
        pct = value / total_latency * 100.0
        draw.rounded_rectangle((legend_x, y, legend_x + 38, y + 38), radius=5, fill=color)
        draw_text(draw, (legend_x + 60, y - 2), label, size=30, bold=True)
        draw_text(draw, (legend_x + 60, y + 42), f"{value:,.0f} ms average | {pct:.1f}%", size=25, fill=MUTED)

    draw_text(draw, (1110, 990), "Interpretation", size=32, bold=True)
    draw_text(draw, (1110, 1040), "The dominant latency is front-loaded in speech recognition", size=24, fill=MUTED)
    draw_text(draw, (1110, 1075), "and VLM TTFT, supporting the TensorRT/INT8 roadmap.", size=24, fill=MUTED)
    image.save(output_path, quality=95)


def generate_tps_bar_chart(rows: list[dict[str, float]], output_path: Path) -> None:
    tps = [row["tps"] for row in rows]
    avg_tps = sum(tps) / len(tps)
    image = Image.new("RGB", (1900, 1150), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (90, 70), "VLM Decode Throughput per Inference Run", size=48, bold=True)
    draw_text(
        draw,
        (90, 130),
        f"Mean decode throughput: {avg_tps:.1f} TPS | Edge UMA baseline before TensorRT acceleration",
        size=28,
        fill="#f9c74f",
        bold=True,
    )

    left, top, right, bottom = 130, 230, 1780, 940
    max_y = max(max(tps) * 1.18, avg_tps * 1.25, 1.0)
    for tick in range(0, int(max_y // 10 + 2) * 10, 10):
        y = bottom - (tick / max_y) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw_text(draw, (left - 18, y), str(tick), size=22, fill=MUTED, anchor="rm")

    draw.line((left, top, left, bottom), fill=FOREGROUND, width=2)
    draw.line((left, bottom, right, bottom), fill=FOREGROUND, width=2)
    bar_gap = 10
    bar_width = max(18, ((right - left) - bar_gap * (len(tps) - 1)) / len(tps))
    for index, value in enumerate(tps):
        x0 = left + index * (bar_width + bar_gap)
        x1 = x0 + bar_width
        y = bottom - (value / max_y) * (bottom - top)
        draw.rounded_rectangle((x0, y, x1, bottom), radius=5, fill="#4cc9f0", outline="#dce7ef", width=1)
        draw_text(draw, ((x0 + x1) / 2.0, y - 12), f"{value:.1f}", size=18, anchor="mb")
        draw_text(draw, ((x0 + x1) / 2.0, bottom + 28), str(index + 1), size=18, fill=MUTED, anchor="mm")

    avg_y = bottom - (avg_tps / max_y) * (bottom - top)
    dash_x = left
    while dash_x < right:
        draw.line((dash_x, avg_y, min(dash_x + 22, right), avg_y), fill=ACCENT, width=4)
        dash_x += 44
    draw_text(draw, (right - 10, avg_y - 18), f"Mean: {avg_tps:.1f} TPS", size=28, fill=ACCENT, bold=True, anchor="rb")
    draw_text(draw, ((left + right) / 2.0, 1030), "Run ID", size=26, bold=True, anchor="mm")
    draw_text(draw, (40, (top + bottom) / 2.0), "TPS", size=26, bold=True, anchor="mm")
    image.save(output_path, quality=95)


def print_markdown_report(rows: list[dict[str, float]], means: dict[str, float]) -> None:
    avg_tps = mean(row["tps"] for row in rows)
    avg_total = sum(means.values())
    asr_ttft_pct = float((means["audio_ms"] + means["ttft_ms"]) / avg_total * 100.0) if avg_total > 0 else 0.0
    print("\n## Edge-VLA Latency Analytics\n")
    print(f"- Runs analyzed: {len(rows)}")
    print(f"- Mean ASR (Whisper): {means['audio_ms']:.1f} ms")
    print(f"- Mean Vision Capture: {means['vision_ms']:.1f} ms")
    print(f"- Mean VLM TTFT (Prompt Eval): {means['ttft_ms']:.1f} ms")
    print(f"- Mean VLM Decode: {means['decode_time_ms']:.1f} ms")
    print(f"- Mean Safety Guardrail: {means['safety_ms']:.1f} ms")
    print(f"- Mean measured profiled latency: {avg_total:.1f} ms")
    print(f"- Mean decode throughput: {avg_tps:.1f} TPS")
    print(f"- ASR + TTFT bottleneck share: {asr_ttft_pct:.1f}%")
    print("\nGenerated artifacts:")
    print("- `docs/latency_pie_chart.png`")
    print("- `docs/tps_bar_chart.png`")


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_performance_csv(Path(args.input))
    means = calculate_latency_means(rows)
    generate_latency_pie_chart(means, output_dir / "latency_pie_chart.png", len(rows))
    generate_tps_bar_chart(rows, output_dir / "tps_bar_chart.png")
    print_markdown_report(rows, means)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
