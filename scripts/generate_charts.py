from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


LATENCY_COLUMNS = ("audio_ms", "vision_ms", "ttft_ms", "decode_time_ms", "safety_ms")
REQUIRED_COLUMNS = (*LATENCY_COLUMNS, "tps")
PIE_LABELS = (
    "ASR (Whisper)",
    "Vision Capture",
    "VLM TTFT (Prompt Eval)",
    "VLM Decode",
    "Safety Guardrail",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate industrial whitepaper charts from Edge-VLA performance telemetry."
    )
    parser.add_argument("--input", default="logs/performance.csv", help="Input performance CSV path.")
    parser.add_argument("--output-dir", default="docs", help="Directory where chart PNG files are written.")
    return parser


def load_performance_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Performance CSV not found: {path}")

    df = pd.read_csv(path, on_bad_lines="skip")
    if df.empty:
        raise ValueError(f"Performance CSV has no usable rows: {path}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Performance CSV missing required columns: {', '.join(missing_columns)}")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    clean_df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if clean_df.empty:
        raise ValueError(f"Performance CSV has no rows with valid numeric metrics: {path}")

    skipped_rows = len(df) - len(clean_df)
    if skipped_rows:
        print(f"Skipped {skipped_rows} malformed telemetry rows.")

    return clean_df


def calculate_latency_means(df: pd.DataFrame) -> pd.Series:
    return df.loc[:, LATENCY_COLUMNS].mean()


def generate_latency_pie_chart(means: pd.Series, output_path: Path) -> None:
    values = [
        float(means["audio_ms"]),
        float(means["vision_ms"]),
        float(means["ttft_ms"]),
        float(means["decode_time_ms"]),
        float(means["safety_ms"]),
    ]
    positive_items = [(label, value) for label, value in zip(PIE_LABELS, values) if value > 0.0]
    if not positive_items:
        raise ValueError("No positive latency values found for latency pie chart.")

    labels = [item[0] for item in positive_items]
    sizes = [item[1] for item in positive_items]
    total_latency = sum(sizes)
    asr_ttft_total = float(means["audio_ms"] + means["ttft_ms"])
    bottleneck_pct = (asr_ttft_total / total_latency * 100.0) if total_latency > 0 else 0.0
    explode = [0.075 if label in {"ASR (Whisper)", "VLM TTFT (Prompt Eval)"} else 0.015 for label in labels]
    colors = ["#4cc9f0", "#80ed99", "#f72585", "#f9c74f", "#adb5bd"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    fig.patch.set_facecolor("#10141c")
    ax.set_facecolor("#10141c")

    wedges, _, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors[: len(labels)],
        explode=explode,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        pctdistance=0.74,
        labeldistance=1.13,
        startangle=105,
        counterclock=False,
        wedgeprops={"linewidth": 1.2, "edgecolor": "#10141c"},
        textprops={"color": "#e8edf2", "fontsize": 9},
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    ax.set_title("Average Control-Loop Latency Breakdown", fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5,
        -0.08,
        f"Primary bottleneck: ASR + VLM TTFT = {bottleneck_pct:.1f}% of average cycle latency",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#f72585" if bottleneck_pct >= 80.0 else "#f9c74f",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(
        wedges,
        [f"{label}: {value:.0f} ms" for label, value in zip(labels, sizes)],
        title="Mean latency",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.04, 0.86, 1.0))
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def generate_tps_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    tps = df["tps"].astype(float).reset_index(drop=True)
    run_ids = list(range(1, len(tps) + 1))
    avg_tps = float(tps.mean()) if len(tps) else 0.0

    plt.style.use("dark_background")
    fig_width = max(10.0, min(22.0, 0.42 * len(tps) + 5.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), dpi=180)
    fig.patch.set_facecolor("#10141c")
    ax.set_facecolor("#10141c")

    bars = ax.bar(
        run_ids,
        tps,
        width=0.72,
        color="#4cc9f0",
        edgecolor="#dce7ef",
        linewidth=0.6,
        label="Measured edge VLM TPS",
    )
    ax.axhline(
        avg_tps,
        color="#f72585",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean TPS: {avg_tps:.1f}",
    )
    ax.set_title("VLM Decode Throughput per Inference Run", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Run ID", fontsize=10)
    ax.set_ylabel("Tokens Per Second (TPS)", fontsize=10)
    ax.set_ylim(0.0, max(float(tps.max()) * 1.18, avg_tps * 1.25, 1.0))
    ax.grid(axis="y", color="#3a4150", alpha=0.55, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    if len(run_ids) <= 30:
        ax.set_xticks(run_ids)
    else:
        step = max(1, len(run_ids) // 20)
        ax.set_xticks(run_ids[::step])

    ax.bar_label(bars, labels=[f"{value:.1f}" for value in tps], padding=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def print_markdown_report(df: pd.DataFrame, means: pd.Series) -> None:
    avg_tps = float(df["tps"].mean())
    avg_total = float(means.sum())
    asr_ttft_pct = float((means["audio_ms"] + means["ttft_ms"]) / avg_total * 100.0) if avg_total > 0 else 0.0

    print("\n## Edge-VLA Latency Analytics\n")
    print(f"- Runs analyzed: {len(df)}")
    print(f"- Mean ASR (Whisper): {means['audio_ms']:.1f} ms")
    print(f"- Mean Vision Capture: {means['vision_ms']:.1f} ms")
    print(f"- Mean VLM TTFT (Prompt Eval): {means['ttft_ms']:.1f} ms")
    print(f"- Mean VLM Decode: {means['decode_time_ms']:.1f} ms")
    print(f"- Mean Safety Guardrail: {means['safety_ms']:.1f} ms")
    print(f"- Mean measured cycle latency: {avg_total:.1f} ms")
    print(f"- Mean decode throughput: {avg_tps:.1f} TPS")
    print(f"- ASR + TTFT bottleneck share: {asr_ttft_pct:.1f}%")
    print("\nGenerated artifacts:")
    print("- `docs/latency_pie_chart.png`")
    print("- `docs/tps_bar_chart.png`")


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_performance_csv(input_path)
    means = calculate_latency_means(df)
    generate_latency_pie_chart(means, output_dir / "latency_pie_chart.png")
    generate_tps_bar_chart(df, output_dir / "tps_bar_chart.png")
    print_markdown_report(df, means)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
