#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT = Path("logs/jetson_telemetry.csv")
DEFAULT_INTERVAL_MS = 1000
DEFAULT_WARN_TEMP_C = 80.0

RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB\b")
SWAP_RE = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB\b")
GR3D_RE = re.compile(r"\bGR3D_FREQ\s+(\d+)%(?:@\[?(\d+)\]?)?")
EMC_RE = re.compile(r"\bEMC_FREQ\s+(\d+)%(?:@(\d+))?")
CPU_RE = re.compile(r"\bCPU\s+\[([^\]]+)\]")
TEMP_RE = re.compile(r"\b([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C\b")


@dataclass(frozen=True)
class TegraStatsSample:
    timestamp_utc: str
    raw: str
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    swap_used_mb: int | None = None
    swap_total_mb: int | None = None
    cpu_avg_pct: float | None = None
    cpu_max_freq_mhz: int | None = None
    emc_pct: int | None = None
    emc_freq_mhz: int | None = None
    gr3d_pct: int | None = None
    gr3d_freq_mhz: int | None = None
    max_temp_c: float | None = None
    temps_json: str = "{}"
    throttling_suspected: bool = False

    def as_csv_row(self) -> dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "swap_used_mb": self.swap_used_mb,
            "swap_total_mb": self.swap_total_mb,
            "cpu_avg_pct": self.cpu_avg_pct,
            "cpu_max_freq_mhz": self.cpu_max_freq_mhz,
            "emc_pct": self.emc_pct,
            "emc_freq_mhz": self.emc_freq_mhz,
            "gr3d_pct": self.gr3d_pct,
            "gr3d_freq_mhz": self.gr3d_freq_mhz,
            "max_temp_c": self.max_temp_c,
            "throttling_suspected": int(self.throttling_suspected),
            "temps_json": self.temps_json,
            "raw": self.raw,
        }


CSV_FIELDS = list(TegraStatsSample(timestamp_utc="", raw="").as_csv_row().keys())


def parse_tegrastats_line(raw: str, *, warn_temp_c: float = DEFAULT_WARN_TEMP_C) -> TegraStatsSample:
    line = raw.strip()
    timestamp = datetime.now(timezone.utc).isoformat()

    ram = RAM_RE.search(line)
    swap = SWAP_RE.search(line)
    gr3d = GR3D_RE.search(line)
    emc = EMC_RE.search(line)
    cpu = CPU_RE.search(line)

    temps = {name: float(value) for name, value in TEMP_RE.findall(line)}
    max_temp = max(temps.values()) if temps else None
    temps_json = json.dumps(temps, sort_keys=True, separators=(",", ":"))

    cpu_avg_pct, cpu_max_freq_mhz = _parse_cpu_block(cpu.group(1) if cpu else "")

    lower = line.lower()
    throttle_text = "throt" in lower or "thermal" in lower and "slow" in lower
    throttle_temp = max_temp is not None and max_temp >= warn_temp_c

    return TegraStatsSample(
        timestamp_utc=timestamp,
        raw=line,
        ram_used_mb=_int_group(ram, 1),
        ram_total_mb=_int_group(ram, 2),
        swap_used_mb=_int_group(swap, 1),
        swap_total_mb=_int_group(swap, 2),
        cpu_avg_pct=cpu_avg_pct,
        cpu_max_freq_mhz=cpu_max_freq_mhz,
        emc_pct=_int_group(emc, 1),
        emc_freq_mhz=_int_group(emc, 2),
        gr3d_pct=_int_group(gr3d, 1),
        gr3d_freq_mhz=_int_group(gr3d, 2),
        max_temp_c=max_temp,
        temps_json=temps_json,
        throttling_suspected=bool(throttle_text or throttle_temp),
    )


def _int_group(match: re.Match[str] | None, index: int) -> int | None:
    if match is None:
        return None
    value = match.group(index)
    return int(value) if value else None


def _parse_cpu_block(block: str) -> tuple[float | None, int | None]:
    if not block:
        return None, None

    utilizations: list[int] = []
    freqs: list[int] = []
    for entry in block.split(","):
        if "off" in entry:
            continue
        util_match = re.search(r"(\d+)%", entry)
        freq_match = re.search(r"@(\d+)", entry)
        if util_match:
            utilizations.append(int(util_match.group(1)))
        if freq_match:
            freqs.append(int(freq_match.group(1)))

    avg_pct = round(sum(utilizations) / len(utilizations), 2) if utilizations else None
    max_freq = max(freqs) if freqs else None
    return avg_pct, max_freq


def stream_tegrastats(interval_ms: int) -> Iterable[str]:
    tegrastats_path = shutil.which("tegrastats")
    if not tegrastats_path:
        raise RuntimeError("tegrastats not found. Run this monitor on the Jetson, not on the Mac.")

    process = subprocess.Popen(
        [tegrastats_path, "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            yield line
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def monitor(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    deadline = time.monotonic() + args.duration_s if args.duration_s else None

    print(
        f"[INFO] Jetson monitor: interval={args.interval_ms}ms "
        f"output={args.output} warn_temp_c={args.warn_temp_c}",
        flush=True,
    )

    try:
        with args.output.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()

            for raw_line in stream_tegrastats(args.interval_ms):
                sample = parse_tegrastats_line(raw_line, warn_temp_c=args.warn_temp_c)
                writer.writerow(sample.as_csv_row())
                handle.flush()

                print(_format_summary(sample, print_raw=args.print_raw), flush=True)

                if deadline is not None and time.monotonic() >= deadline:
                    break
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
        return 130
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    return 0


def _format_summary(sample: TegraStatsSample, *, print_raw: bool) -> str:
    ram = _format_mb_pair(sample.ram_used_mb, sample.ram_total_mb)
    swap = _format_mb_pair(sample.swap_used_mb, sample.swap_total_mb)
    cpu = _format_pct(sample.cpu_avg_pct)
    gpu = _format_pct(sample.gr3d_pct)
    temp = f"{sample.max_temp_c:.1f}C" if sample.max_temp_c is not None else "n/a"
    throttle = " THROTTLE?" if sample.throttling_suspected else ""

    summary = (
        f"[JETSON] ram={ram} swap={swap} cpu_avg={cpu} "
        f"gpu={gpu} max_temp={temp}{throttle}"
    )
    return f"{summary}\n[RAW] {sample.raw}" if print_raw else summary


def _format_mb_pair(used: int | None, total: int | None) -> str:
    if used is None or total is None:
        return "n/a"
    return f"{used}/{total}MB"


def _format_pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Jetson tegrastats to CSV for thermal, memory, CPU, and GPU bottleneck analysis."
    )
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run until Ctrl-C.")
    parser.add_argument("--warn-temp-c", type=float, default=DEFAULT_WARN_TEMP_C)
    parser.add_argument("--print-raw", action="store_true", help="Also print the original tegrastats line.")
    return parser.parse_args()


def main() -> int:
    return monitor(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
