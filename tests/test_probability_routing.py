"""Probability routing tests use neutral local texts and no network calls."""
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from jev_research.probability_routing import (
    InsufficientRoutingSupport, select_probability_interval, select_routing_cohort,
)


def make_frame(split="validation", groups=None, offset=0):
    # probability, number of rows, number of intentionally wrong predictions
    groups = groups or [(0.1, 100, 0), (0.35, 200, 100), (0.45, 200, 180),
                        (0.55, 200, 100), (0.9, 300, 0)]
    rows = []
    for probability, count, errors in groups:
        for within_group in range(count):
            row_id = offset + len(rows)
            text = f"Neutral probability-routing fixture number {row_id}."
            predicted = int(probability >= 0.5)
            truth = 1 - predicted if within_group < errors else predicted
            rows.append({
                "row_id": row_id, "text": text, "cleaned_text": text.lower(),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_split": split, "p_xgb": probability,
                "y_true": truth, "y_xgb": predicted,
                "error_type": "correct" if truth == predicted else "false_negative" if truth else "false_positive",
                "dataset_sha256": "a" * 64, "baseline_fingerprint": "b" * 64,
            })
    return pd.DataFrame(rows)


class ProbabilityRoutingTests(unittest.TestCase):
    def setUp(self):
        self.validation = make_frame()
        self.test = make_frame("test", offset=10000)
        self.result = select_probability_interval(self.validation, target_size=400, test_size=1000)

    def test_peak_bin_is_distinct_from_feasible_supported_interval(self):
        self.assertEqual(self.result.metadata["minimum_validation_support"], 440)
        self.assertEqual(self.result.metadata["highest_error_rate_bin"]["lower"], 0.45)
        self.assertEqual(self.result.metadata["highest_error_rate_bin"]["error_rate"], 0.9)
        self.assertEqual((self.result.interval["lower"], self.result.interval["upper"]), (0.35, 0.6))
        self.assertEqual(self.result.metadata["selected_validation"]["n"], 600)
        self.assertEqual(self.result.metadata["selected_validation"]["errors"], 380)
        self.assertEqual(len(self.result.diagnostics), 20)
        self.assertEqual(len(self.result.candidates), 210)
        self.assertTrue(self.result.metadata["selected_validation"]["n"] >= 440)
        json.dumps(self.result.metadata, allow_nan=False)

    def test_error_count_and_error_rate_are_separate_diagnostics(self):
        frame = make_frame(groups=[(0.45, 200, 180), (0.95, 1000, 200)])
        result = select_probability_interval(frame, target_size=100, test_size=2000)
        self.assertEqual(result.metadata["highest_error_rate_bin"]["lower"], 0.45)
        self.assertEqual(result.metadata["highest_error_count_bin"]["lower"], 0.95)

    def test_low_support_diagnostic_peak_is_visible_but_not_supported_peak(self):
        frame = make_frame(groups=[(0.45, 200, 180), (0.75, 1, 1), (0.95, 1000, 200)])
        result = select_probability_interval(frame, target_size=100, test_size=2000)
        self.assertEqual(result.metadata["raw_highest_error_rate_bin"]["lower"], 0.75)
        self.assertEqual(result.metadata["highest_error_rate_bin"]["lower"], 0.45)
        row = result.diagnostics.loc[result.diagnostics.lower.eq(0.75)].iloc[0]
        self.assertFalse(row.supported_peak_eligible)

    def test_exact_boundaries_include_zero_and_one_without_double_counting(self):
        frame = make_frame(groups=[(0.0, 1, 0), (0.05, 1, 1), (0.5, 1, 1), (1.0, 1, 0)])
        result = select_probability_interval(frame, target_size=1, test_size=10)
        bins = result.diagnostics
        self.assertEqual(int(bins.n.sum()), 4)
        self.assertEqual(int(bins.loc[bins.lower.eq(0), "n"].iloc[0]), 1)
        self.assertEqual(int(bins.loc[bins.lower.eq(0.05), "n"].iloc[0]), 1)
        self.assertEqual(int(bins.loc[bins.lower.eq(0.5), "n"].iloc[0]), 1)
        self.assertEqual(int(bins.loc[bins.upper.eq(1), "n"].iloc[0]), 1)
        self.assertTrue(bins.loc[bins.upper.eq(1), "upper_inclusive"].iloc[0])

    def test_error_rate_ties_prefer_narrower_then_lower_window(self):
        frame = make_frame(groups=[(0.1, 100, 10), (0.3, 100, 10),
                                   (0.7, 100, 10), (0.9, 100, 10)])
        result = select_probability_interval(frame, target_size=50, test_size=400, coverage_buffer=1)
        self.assertEqual((result.interval["lower"], result.interval["upper"]), (0.1, 0.15))

    def test_validation_only_api_has_no_test_labels_or_probabilities(self):
        shuffled = self.validation.sample(frac=1, random_state=9)
        replay = select_probability_interval(shuffled, target_size=400, test_size=len(self.test))
        self.assertEqual(self.result.metadata, replay.metadata)
        wrong_source = self.validation.copy()
        wrong_source["source_split"] = "test"
        with self.assertRaisesRegex(ValueError, "Only validation"):
            select_probability_interval(wrong_source, target_size=400, test_size=1000)

    def test_impossible_support_and_invalid_values_fail_instead_of_relaxing(self):
        with self.assertRaises(InsufficientRoutingSupport):
            select_probability_interval(self.validation, target_size=1001, test_size=1000)
        with self.assertRaises(InsufficientRoutingSupport):
            select_probability_interval(self.validation, target_size=1000, test_size=1000)
        for invalid in [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"]:
            damaged = self.validation.copy().astype({"p_xgb": object})
            damaged.loc[0, "p_xgb"] = invalid
            with self.assertRaises(ValueError):
                select_probability_interval(damaged, target_size=400, test_size=1000)
        for kwargs in [{"coverage_buffer": 0.9}, {"bin_width": 0.3}, {"target_size": True}]:
            values = {"target_size": 400, "test_size": 1000, **kwargs}
            with self.assertRaises(ValueError):
                select_probability_interval(self.validation, **values)

    def test_test_selection_keeps_both_correct_and_incorrect_predictions(self):
        result = select_routing_cohort(self.test, self.result, target_size=400)
        self.assertEqual(len(result.cohort), 400)
        self.assertEqual(result.metadata["eligible_test_count"], 600)
        self.assertEqual(result.metadata["unqueried_interval_count"], 200)
        self.assertTrue(result.cohort.y_true.eq(result.cohort.y_xgb).any())
        self.assertTrue(result.cohort.y_true.ne(result.cohort.y_xgb).any())
        self.assertFalse(result.metadata["test_labels_used_for_selection"])
        self.assertFalse({"text", "cleaned_text"} & set(result.manifest.columns))
        self.assertTrue(result.cohort.p_xgb.ge(0.35).all() and result.cohort.p_xgb.lt(0.6).all())

    def test_changed_or_absent_test_labels_do_not_change_selected_ids_or_fingerprint(self):
        first = select_routing_cohort(self.test, self.result, target_size=400)
        changed = self.test.copy()
        changed["y_true"] = 1 - changed.y_true
        changed["error_type"] = "This field is intentionally ignored by routing."
        second = select_routing_cohort(changed, self.result.interval, target_size=400)
        unlabeled = self.test.drop(columns=["y_true", "y_xgb", "error_type"])
        third = select_routing_cohort(unlabeled, self.result.interval, target_size=400)
        for other in (second, third):
            self.assertEqual(first.cohort.sample_id.tolist(), other.cohort.sample_id.tolist())
            self.assertEqual(first.metadata, other.metadata)

    def test_test_selection_is_stable_under_reordering_and_changes_with_seed(self):
        first = select_routing_cohort(self.test, self.result.interval, target_size=400)
        second = select_routing_cohort(self.test.sample(frac=1, random_state=7), self.result.interval, target_size=400)
        pd.testing.assert_frame_equal(first.manifest, second.manifest)
        self.assertEqual(first.metadata, second.metadata)
        other_seed = select_routing_cohort(self.test, self.result.interval, target_size=400, seed=43)
        self.assertNotEqual(first.cohort.sample_id.tolist(), other_seed.cohort.sample_id.tolist())

    def test_insufficient_test_capacity_does_not_expand_interval(self):
        with self.assertRaisesRegex(InsufficientRoutingSupport, "automatic expansion is forbidden"):
            select_routing_cohort(self.test, {"lower": 0.45, "upper": 0.5}, target_size=300)

    def test_routing_endpoints_obey_same_boundary_convention(self):
        frame = make_frame("test", groups=[(0.0, 1, 0), (0.5, 1, 0), (1.0, 1, 0)])
        low = select_routing_cohort(frame, {"lower": 0.0, "upper": 0.5}, target_size=1)
        self.assertEqual(low.cohort.p_xgb.tolist(), [0.0])
        high = select_routing_cohort(frame, {"lower": 0.5, "upper": 1.0,
                                            "upper_inclusive": np.bool_(True)}, target_size=2)
        self.assertEqual(set(high.cohort.p_xgb), {0.5, 1.0})
        with self.assertRaisesRegex(ValueError, "inclusive"):
            select_routing_cohort(frame, {"lower": 0.0, "upper": 0.5, "upper_inclusive": True}, target_size=1)

    def test_provenance_and_all_text_hashes_are_checked_before_selection(self):
        for column, value in [("source_split", "validation"), ("dataset_sha256", "c" * 64),
                              ("baseline_fingerprint", "c" * 64)]:
            bad = self.test.copy()
            bad[column] = value
            with self.assertRaises(ValueError):
                select_routing_cohort(bad, self.result.interval, target_size=400)
        bad = self.test.copy()
        # An out-of-interval row is checked too, before any provider work.
        bad.loc[0, "text"] = "Changed neutral text."
        with self.assertRaisesRegex(ValueError, "checksums do not match"):
            select_routing_cohort(bad, self.result.interval, target_size=400)
        with self.assertRaisesRegex(ValueError, "size differs"):
            select_routing_cohort(self.test, self.result.interval, target_size=399)
        with self.assertRaisesRegex(ValueError, "population differs"):
            select_routing_cohort(self.test.iloc[:-1], self.result.interval, target_size=400)


if __name__ == "__main__":
    unittest.main()
