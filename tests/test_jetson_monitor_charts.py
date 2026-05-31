from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_jetson_monitor_charts import build_summary, load_rows


class JetsonMonitorChartsTests(unittest.TestCase):
    def test_load_rows_and_build_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "jetson_telemetry.csv"
            csv_path.write_text(
                "timestamp_utc,ram_used_mb,ram_total_mb,swap_used_mb,swap_total_mb,"
                "cpu_avg_pct,cpu_max_freq_mhz,emc_pct,emc_freq_mhz,gr3d_pct,"
                "gr3d_freq_mhz,max_temp_c,throttling_suspected,temps_json,raw\n"
                "2026-05-30T00:00:00Z,2000,7851,0,3925,10,729,4,2133,45,918,61.5,0,{},raw1\n"
                "2026-05-30T00:00:01Z,2500,7851,128,3925,30,1510,15,2133,80,1098,82.0,1,{},raw2\n",
                encoding="utf-8",
            )

            rows = load_rows(csv_path)
            summary = build_summary(rows)

            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["throttling_samples"], 1)
            self.assertEqual(summary["ram_used_mb"]["max"], 2500.0)
            self.assertEqual(summary["gr3d_pct"]["mean"], 62.5)
            self.assertEqual(summary["max_temp_c"]["max"], 82.0)


if __name__ == "__main__":
    unittest.main()
