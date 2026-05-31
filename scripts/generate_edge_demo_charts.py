from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#10141c"
FOREGROUND = "#e8edf2"
MUTED = "#9aa6b2"
GRID = "#343b49"
ACCENT = "#f72585"
COLORS = ("#4cc9f0", "#80ed99", "#f72585", "#f9c74f", "#adb5bd")
PIE_LABELS = (
    "ASR (Mac Whisper)",
    "Vision Capture",
    "Jetson SmolVLM Inference",
    "PX4 / MAVSDK Action",
    "HTTP + Safety Overhead",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Mac + Jetson distributed demo charts.")
    parser.add_argument("--input", default="logs/telemetry.csv", help="Input Mac sensor telemetry CSV.")
    parser.add_argument("--output-dir", default="docs/imgs", help="Directory where chart PNG files are written.")
    return parser


def read_float(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    if value in ("", None):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_edge_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row
            for row in reader
            if row.get("profile") == "edge_distributed"
            and row.get("component") == "mac_sensor_client"
            and row.get("ok") == "1"
        ]

    if not rows:
        raise ValueError(f"No accepted edge_distributed mac_sensor_client rows found in {path}")
    return rows


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


def calculate_latency_means(rows: list[dict[str, str]]) -> dict[str, float]:
    http_overheads = []
    for row in rows:
        http_ms = read_float(row, "http_ms")
        server_total_ms = read_float(row, "total_ms")
        safety_ms = read_float(row, "safety_ms")
        http_overheads.append(max(0.0, http_ms - server_total_ms) + safety_ms)

    return {
        "audio_ms": mean(read_float(row, "audio_ms") for row in rows),
        "vision_ms": mean(read_float(row, "vision_ms") for row in rows),
        "vlm_ms": mean(read_float(row, "ttft_ms") for row in rows),
        "action_ms": mean(read_float(row, "action_ms") for row in rows),
        "http_safety_ms": mean(http_overheads),
    }


def generate_edge_latency_pie_chart(means: dict[str, float], output_path: Path, runs_analyzed: int) -> None:
    columns = ("audio_ms", "vision_ms", "vlm_ms", "action_ms", "http_safety_ms")
    values = [means[column] for column in columns]
    total_latency = sum(values)
    if total_latency <= 0.0:
        raise ValueError("No positive latency values found for edge latency pie chart.")

    bottleneck_pct = (means["audio_ms"] + means["vlm_ms"]) / total_latency * 100.0
    image = Image.new("RGB", (1800, 1260), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (90, 70), "Mac + Jetson Control-Loop Latency Breakdown", size=48, bold=True)
    draw_text(
        draw,
        (90, 130),
        f"{runs_analyzed} accepted commands | ASR + Jetson VLM bottleneck: {bottleneck_pct:.1f}% of profiled latency",
        size=28,
        fill="#f9c74f",
        bold=True,
    )

    pie_box = (170, 240, 1030, 1100)
    center = (600, 670)
    start_angle = -90.0
    for value, color in zip(values, COLORS):
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
    draw_text(draw, (1110, 1040), "The edge hardware is stable; latency is dominated by", size=24, fill=MUTED)
    draw_text(draw, (1110, 1075), "Mac ASR plus SmolVLM visual inference on the Jetson.", size=24, fill=MUTED)
    image.save(output_path, quality=95)


def generate_edge_tps_bar_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    tps = [read_float(row, "tps") for row in rows if read_float(row, "tps") > 0.0]
    if not tps:
        raise ValueError("No positive TPS values found for edge TPS chart.")

    avg_tps = sum(tps) / len(tps)
    image = Image.new("RGB", (1900, 1150), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (90, 70), "Jetson SmolVLM Throughput per Visual Inference", size=48, bold=True)
    draw_text(
        draw,
        (90, 130),
        f"Mean decode throughput: {avg_tps:.1f} TPS | SmolVLM-256M CUDA baseline on Orin Nano",
        size=28,
        fill="#f9c74f",
        bold=True,
    )

    left, top, right, bottom = 130, 230, 1780, 940
    max_y = max(max(tps) * 1.4, avg_tps * 1.5, 12.0)
    tick_step = 2 if max_y <= 20 else 10
    for tick in range(0, int(max_y // tick_step + 2) * tick_step, tick_step):
        y = bottom - (tick / max_y) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw_text(draw, (left - 18, y), str(tick), size=22, fill=MUTED, anchor="rm")

    draw.line((left, top, left, bottom), fill=FOREGROUND, width=2)
    draw.line((left, bottom, right, bottom), fill=FOREGROUND, width=2)
    bar_gap = 70
    bar_width = min(260, max(110, ((right - left) - bar_gap * (len(tps) - 1)) / max(len(tps), 1)))
    total_width = len(tps) * bar_width + (len(tps) - 1) * bar_gap
    start_x = left + ((right - left) - total_width) / 2.0

    for index, value in enumerate(tps):
        x0 = start_x + index * (bar_width + bar_gap)
        x1 = x0 + bar_width
        y = bottom - (value / max_y) * (bottom - top)
        draw.rounded_rectangle((x0, y, x1, bottom), radius=5, fill="#4cc9f0", outline="#dce7ef", width=1)
        draw_text(draw, ((x0 + x1) / 2.0, y - 14), f"{value:.1f}", size=24, anchor="mb")
        draw_text(draw, ((x0 + x1) / 2.0, bottom + 34), str(index + 1), size=22, fill=MUTED, anchor="mm")

    avg_y = bottom - (avg_tps / max_y) * (bottom - top)
    dash_x = left
    while dash_x < right:
        draw.line((dash_x, avg_y, min(dash_x + 22, right), avg_y), fill=ACCENT, width=4)
        dash_x += 44
    draw_text(draw, (right - 10, avg_y - 18), f"Mean: {avg_tps:.1f} TPS", size=28, fill=ACCENT, bold=True, anchor="rb")
    draw_text(draw, ((left + right) / 2.0, 1030), "Visual Inference Run ID", size=26, bold=True, anchor="mm")
    draw_text(draw, (40, (top + bottom) / 2.0), "TPS", size=26, bold=True, anchor="mm")
    image.save(output_path, quality=95)


def print_report(rows: list[dict[str, str]], means: dict[str, float], output_dir: Path) -> None:
    tps = [read_float(row, "tps") for row in rows if read_float(row, "tps") > 0.0]
    total = sum(means.values())
    bottleneck_pct = (means["audio_ms"] + means["vlm_ms"]) / total * 100.0 if total else 0.0
    print("\n## Mac + Jetson Edge Demo Charts\n")
    print(f"- Accepted commands analyzed: {len(rows)}")
    print(f"- Visual VLM runs analyzed: {len(tps)}")
    print(f"- Mean ASR: {means['audio_ms']:.1f} ms")
    print(f"- Mean vision capture: {means['vision_ms']:.1f} ms")
    print(f"- Mean Jetson SmolVLM inference: {means['vlm_ms']:.1f} ms")
    print(f"- Mean PX4/MAVSDK action: {means['action_ms']:.1f} ms")
    print(f"- Mean HTTP + safety overhead: {means['http_safety_ms']:.1f} ms")
    print(f"- ASR + Jetson VLM bottleneck share: {bottleneck_pct:.1f}%")
    if tps:
        print(f"- Mean visual inference TPS: {mean(tps):.1f}")
    print("\nGenerated artifacts:")
    print(f"- `{output_dir / 'edge_latency_pie_chart.png'}`")
    print(f"- `{output_dir / 'edge_tps_bar_chart.png'}`")


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_edge_rows(Path(args.input))
    means = calculate_latency_means(rows)
    generate_edge_latency_pie_chart(means, output_dir / "edge_latency_pie_chart.png", len(rows))
    generate_edge_tps_bar_chart(rows, output_dir / "edge_tps_bar_chart.png")
    print_report(rows, means, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
