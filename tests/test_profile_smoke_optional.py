import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("RUN_NATURE_SMOKE") == "1", "Set RUN_NATURE_SMOKE=1 to run optional integration smoke test.")
class TestNatureProfileSmoke(unittest.TestCase):
    def test_profile_smoke_creates_required_artifacts(self) -> None:
        cmd = [sys.executable, str(ROOT / "test.py"), "--profile", "nature_comms_v1", "--smoke"]
        subprocess.run(cmd, cwd=str(ROOT), check=True)

        expected = [
            ROOT / "outputs" / "paper_even_final" / "39_claim_loss_ablation_nature" / "loss_ablation_metrics_long.csv",
            ROOT / "outputs" / "paper_even_final" / "40_claim_visibility_causal_global" / "visibility_causal_metrics.csv",
            ROOT / "outputs" / "paper_even_final" / "40_claim_visibility_causal_high_value" / "visibility_causal_metrics.csv",
            ROOT / "outputs" / "paper_even_final" / "99_stats_tables" / "main_table.csv",
            ROOT / "outputs" / "paper_figures_nature_v1" / "fig2_loss_ablation_parity_vs_mmd_nll.pdf",
        ]
        missing = [str(p) for p in expected if not p.exists()]
        self.assertFalse(missing, msg="Missing artifacts:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
