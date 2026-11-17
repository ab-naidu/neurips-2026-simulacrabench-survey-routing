"""Regression checks for held-out calibration and submission ordering."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_routed"))
import main as candidate


class RoutedTests(unittest.TestCase):
    def test_ties_and_interior_weights(self):
        same = np.array([0.5, 0.5])
        self.assertEqual(candidate._choose_weight(same, same, 2), 1)
        alternating = np.array([0.9, 0.1])
        self.assertEqual(candidate._choose_weight(alternating, alternating[::-1], 2), 0.5)

    def test_entire_visible_respondents_are_masked_for_weight_selection(self):
        keys = ("a", "b")
        schema = {"gated_value": "SKIP", "items": {
            key: {"class": "PREDICT", "values": ["0", "1"]} for key in keys}}
        visible = pd.DataFrame({"respondent_id": np.arange(100), "a": "0", "b": "1"})
        frame = pd.concat([visible, pd.DataFrame([{"respondent_id": 999}])],
                          ignore_index=True)

        def heldout_predictions(calibration_frame, *args):
            mask = calibration_frame[list(keys)].isna()
            self.assertTrue(mask["a"].equals(mask["b"]))
            self.assertEqual(mask["a"].sum(), 20)
            self.assertNotIn(999, calibration_frame["respondent_id"].values)
            return [np.array([0.5, 0.5])] * 40

        with patch.object(candidate.gate_expert, "predict", side_effect=heldout_predictions), \
                patch.object(candidate.statistical, "predict", side_effect=heldout_predictions):
            weights = candidate._fit_weights(frame, schema, list(keys), {})
        np.testing.assert_equal(weights, [1, 1])
        pd.testing.assert_frame_equal(frame.iloc[:100].reset_index(drop=True), visible)

    def test_schema_order_with_rare_levels_and_few_visible_cases(self):
        schema = {"gated_value": "SKIP", "items": {
            "region": {"class": "GIVEN", "values": ["r"]},
            "first": {"class": "PREDICT", "values": ["a", "b"]},
            "next": {"class": "PREDICT", "values": ["x", "y"],
                     "gate": {"parent": "first", "observed_if": ["a"]}}}}
        frame = pd.DataFrame({"next": ["SKIP", None], "first": ["b", None],
                              "region": ["r", "r"]})
        with patch.object(candidate.gate_expert, "llm_prior", return_value={}):
            output = candidate.predict(frame, schema)
        self.assertEqual([len(p) for p in output], [2, 3])
        for p in output:
            self.assertTrue(np.isfinite(p).all() and (np.asarray(p) >= 0).all())
            self.assertAlmostEqual(sum(p), 1)


if __name__ == "__main__":
    unittest.main()
