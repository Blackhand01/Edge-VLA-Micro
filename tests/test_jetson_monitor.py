from __future__ import annotations

import unittest

from src.tools.jetson_monitor import parse_tegrastats_line


class JetsonMonitorTests(unittest.TestCase):
    def test_parse_common_tegrastats_line(self) -> None:
        sample = parse_tegrastats_line(
            "RAM 2201/7851MB (lfb 120x4MB) SWAP 0/3925MB (cached 0MB) "
            "CPU [12%@729,off,7%@729,3%@729,0%@729,1%@729] "
            "EMC_FREQ 4%@2133 GR3D_FREQ 57%@[918] AO@41C GPU@62.5C CPU@59C thermal@60.2C",
            warn_temp_c=80.0,
        )

        self.assertEqual(sample.ram_used_mb, 2201)
        self.assertEqual(sample.ram_total_mb, 7851)
        self.assertEqual(sample.swap_used_mb, 0)
        self.assertEqual(sample.swap_total_mb, 3925)
        self.assertEqual(sample.cpu_avg_pct, 4.6)
        self.assertEqual(sample.cpu_max_freq_mhz, 729)
        self.assertEqual(sample.emc_pct, 4)
        self.assertEqual(sample.emc_freq_mhz, 2133)
        self.assertEqual(sample.gr3d_pct, 57)
        self.assertEqual(sample.gr3d_freq_mhz, 918)
        self.assertEqual(sample.max_temp_c, 62.5)
        self.assertFalse(sample.throttling_suspected)
        self.assertIn('"GPU":62.5', sample.temps_json)

    def test_marks_high_temperature_as_suspected_throttling(self) -> None:
        sample = parse_tegrastats_line(
            "RAM 7000/7851MB SWAP 1000/3925MB CPU [99%@1510] GR3D_FREQ 99%@[1098] GPU@83.2C",
            warn_temp_c=80.0,
        )

        self.assertTrue(sample.throttling_suspected)


if __name__ == "__main__":
    unittest.main()
