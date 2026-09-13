import math
import tempfile
import unittest
from pathlib import Path

from dd_loop_upper_bound_alpha1 import (
    DD_N_HE3,
    DD_P_T,
    calculate_loop_upper_bound,
    run_default_study,
    scan_strict_bound,
)


class Alpha1Tests(unittest.TestCase):
    def test_bosch_hale_reference_points(self):
        total_10_kev_mb = (
            DD_N_HE3.cross_section_m2(10.0) + DD_P_T.cross_section_m2(10.0)
        ) / 1.0e-31
        total_100_kev_mb = (
            DD_N_HE3.cross_section_m2(100.0) + DD_P_T.cross_section_m2(100.0)
        ) / 1.0e-31
        self.assertAlmostEqual(total_10_kev_mb, 0.5591251985, places=8)
        self.assertAlmostEqual(total_100_kev_mb, 70.04894814, places=6)

    def test_zero_density_has_zero_gain(self):
        result = calculate_loop_upper_bound(100.0, 0.0, 20e-6, 1e-4)
        self.assertEqual(result.k_loop_max, 0.0)
        self.assertEqual(result.fusion_power_w, 0.0)

    def test_energy_ledger_closes(self):
        result = calculate_loop_upper_bound(100.0, 1.62e16, 20e-6, 1e-4)
        self.assertLess(result.energy_ledger.closure_fraction, 1e-14)

    def test_required_n_tau_reconstructs_unity(self):
        base = calculate_loop_upper_bound(100.0, 1.62e16, 20e-6, 1e-4)
        check = calculate_loop_upper_bound(
            100.0, 1.0, base.required_n_tau_m3_s, 1e-4,
            topology="head_on_equal_beams",
        )
        self.assertTrue(math.isclose(check.k_loop_max, 1.0, rel_tol=1e-12))

    def test_strict_scan_maximum_is_upper_boundary(self):
        best, _ = scan_strict_bound(1.62e16, 20e-6, 1e-4, grid_points=101)
        self.assertEqual(best.topology, "head_on_equal_beams")
        self.assertTrue(math.isclose(best.tail_energy_kev, 100.0, rel_tol=1e-12))

    def test_default_study_writes_reproducible_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_default_study(Path(tmp))
            self.assertTrue((Path(tmp) / "alpha1_summary.json").is_file())
            self.assertTrue((Path(tmp) / "alpha1_table.csv").is_file())
            self.assertEqual(summary["decision"], "NO_GO_SELF_SUSTAINING_DD_AT_CURRENT_PARAMETERS")


if __name__ == "__main__":
    unittest.main()
