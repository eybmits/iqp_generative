import unittest

import numpy as np

from experiments.legacy.exp44_beta_holdout_score_state_diagnostic import (
    build_score_mass_rows,
    classify_holdout_learning,
    holdout_score_diagnostics,
)


class TestExp44ScoreStateDiagnostic(unittest.TestCase):
    def test_build_score_mass_rows_conservation(self) -> None:
        scores = np.array([1, 1, 2, 2, 3, 3], dtype=np.float64)
        p_star = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1], dtype=np.float64)
        holdout_mask = np.array([False, True, False, False, True, False], dtype=bool)

        p_train = p_star.copy()
        p_train[holdout_mask] = 0.0
        p_train /= float(np.sum(p_train))
        q_model = np.array([0.08, 0.10, 0.22, 0.18, 0.24, 0.18], dtype=np.float64)
        q_model /= float(np.sum(q_model))

        rows = build_score_mass_rows(
            p_star=p_star,
            p_train=p_train,
            q_model=q_model,
            holdout_mask=holdout_mask,
            scores=scores,
        )

        p_star_sum = sum(float(r["p_star_mass"]) for r in rows)
        p_train_sum = sum(float(r["p_train_mass"]) for r in rows)
        removed_sum = sum(float(r["removed_holdout_mass"]) for r in rows)
        holdout_count = sum(int(r["holdout_state_count"]) for r in rows)

        self.assertAlmostEqual(p_star_sum, 1.0, places=12)
        self.assertAlmostEqual(p_train_sum, 1.0, places=12)
        self.assertAlmostEqual(removed_sum, float(np.sum(p_star[holdout_mask])), places=12)
        self.assertEqual(holdout_count, int(np.sum(holdout_mask)))

    def test_classification_state_diverse_match(self) -> None:
        scores = np.array([4, 4, 4, 4, 5, 5, 5, 5], dtype=np.float64)
        holdout_mask = np.ones(8, dtype=bool)
        p_star = np.ones(8, dtype=np.float64) / 8.0

        q_model = np.array([0.12, 0.13, 0.12, 0.13, 0.13, 0.12, 0.13, 0.12], dtype=np.float64)
        q_model /= float(np.sum(q_model))

        rows, summary = holdout_score_diagnostics(
            p_star=p_star,
            q_model=q_model,
            holdout_mask=holdout_mask,
            scores=scores,
            qdiag=120,
        )
        verdict, _reason = classify_holdout_learning(rows, summary)
        self.assertEqual(verdict, "State-diverse Match")

    def test_classification_score_hit_but_state_collapse(self) -> None:
        scores = np.array([4, 4, 4, 4, 5, 5, 5, 5], dtype=np.float64)
        holdout_mask = np.ones(8, dtype=bool)
        p_star = np.ones(8, dtype=np.float64) / 8.0

        # Score shares match (0.5/0.5), but each level is dominated by a single state.
        q_model = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        q_model /= float(np.sum(q_model))

        rows, summary = holdout_score_diagnostics(
            p_star=p_star,
            q_model=q_model,
            holdout_mask=holdout_mask,
            scores=scores,
            qdiag=120,
        )
        verdict, _reason = classify_holdout_learning(rows, summary)
        self.assertEqual(verdict, "Score-hit but State-collapse")

    def test_classification_missed_holdout_structure(self) -> None:
        scores = np.array([4, 4, 4, 4, 5, 5, 5, 5], dtype=np.float64)
        holdout_mask = np.ones(8, dtype=bool)
        p_star = np.ones(8, dtype=np.float64) / 8.0

        # Very large score-share mismatch: almost all mass on score level 4.
        q_model = np.array([0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q_model /= float(np.sum(q_model))

        rows, summary = holdout_score_diagnostics(
            p_star=p_star,
            q_model=q_model,
            holdout_mask=holdout_mask,
            scores=scores,
            qdiag=120,
        )
        verdict, _reason = classify_holdout_learning(rows, summary)
        self.assertEqual(verdict, "Missed holdout structure")


if __name__ == "__main__":
    unittest.main()
