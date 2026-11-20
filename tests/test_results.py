"""Standard-library checks on committed aggregate results (no raw NHANES records)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestArchivedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))

    def test_provenance_and_counts(self):
        d = self.data
        self.assertTrue(d["data_verified_as_xpt"])
        self.assertEqual(d["development"]["n"], 5604)
        self.assertEqual(d["validation"]["n"], 3736)
        self.assertEqual(d["development"]["events12"], 229)
        self.assertEqual(d["validation"]["events12"], 206)

    def test_discrimination_in_unit_interval(self):
        m = self.data["model_stats"]
        for model in ("FIB-4", "Logistic", "Gradient boosting"):
            self.assertTrue(0 <= m[model]["auroc"] <= 1)
            self.assertTrue(0 <= m[model]["auprc"] <= 1)
        self.assertGreater(m["Gradient boosting"]["auroc"], m["FIB-4"]["auroc"])

    def test_capacity_and_case_capture(self):
        for name, records in self.data["policies"].items():
            for percent in (10, 20, 30):
                policy = records[str(percent)]
                self.assertAlmostEqual(policy["referral_fraction"], percent/100, places=8)
                self.assertTrue(0 <= policy["case_capture"] <= 1)
            self.assertLessEqual(records["10"]["case_capture"], records["20"]["case_capture"])
            self.assertLessEqual(records["20"]["case_capture"], records["30"]["case_capture"])
        diff = (self.data["policies"]["Gradient boosting"]["20"]["case_capture"] -
                self.data["policies"]["FIB-4"]["20"]["case_capture"])
        self.assertAlmostEqual(diff, 0.3063751046045848, places=7)

    def test_all_six_figures_present(self):
        for basename in (
            "fig1_cohort_flow", "fig2_roc", "fig3_calibration",
            "fig4_capacity", "fig5_equity_referral", "fig6_importance"
        ):
            image = ROOT / "figures" / f"{basename}.png"
            self.assertTrue(image.is_file())
            self.assertGreater(image.stat().st_size, 10_000)

if __name__ == "__main__":
    unittest.main()
