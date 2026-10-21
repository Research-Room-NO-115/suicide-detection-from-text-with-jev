import json
import math
from pathlib import Path
import tempfile
import unittest

from jev_research.reporting import (
    build_comparison_report,
    exact_mcnemar_pvalue,
    render_markdown,
    wilson_interval,
    write_report,
)


def row(identifier, label, base, tuned, **extra):
    return {
        "sample_id": identifier,
        "y_true": label,
        "error_type": "false_negative" if label else "false_positive",
        "p_xgb": 0.1 if label else 0.9,
        "source_split": "test",
        "jev_partition": "evaluation",
        "base_score": base,
        "tuned_score": tuned,
        **extra,
    }


class ReportingTests(unittest.TestCase):
    def test_exact_paired_counts_and_stratum_denominators(self):
        report = build_comparison_report([
            row("a", 1, 0.1, 0.9),  # Tuned improvement.
            row("b", 1, 0.9, 0.1),  # Tuned regression.
            row("c", 0, 0.2, 0.1),  # Both correct.
            row("d", 0, 0.9, 0.1),  # Tuned improvement.
            row("e", 1, 0.1, 0.2),  # Both wrong.
        ])
        all_errors = report["primary_paired"]["all_errors"]
        self.assertEqual(all_errors["base"]["corrected"], 2)
        self.assertEqual(all_errors["tuned"]["corrected"], 3)
        self.assertEqual(all_errors["paired_outcomes"]["improvements"], 2)
        self.assertEqual(all_errors["paired_outcomes"]["regressions"], 1)
        self.assertEqual(all_errors["paired_outcomes"]["net_additional_corrections"], 1)
        fn = report["primary_paired"]["false_negative"]
        self.assertEqual(fn["paired_n"], 3)
        self.assertEqual(fn["base"]["corrected"], 1)
        self.assertEqual(fn["tuned"]["corrected"], 1)
        self.assertEqual(report["primary_paired"]["false_positive"]["tuned"]["rate"], 1.0)

    def test_missing_and_failed_outcomes_never_silently_shrink_selected_denominator(self):
        report = build_comparison_report([
            row("a", 1, 0.1, 0.9),
            row("b", 1, 0.9, None, tuned_status="failed"),
            row("c", 0, None, 0.1),
            row("d", 0, math.nan, None),
        ])
        paired = report["primary_paired"]["all_errors"]
        self.assertEqual((paired["selected_n"], paired["paired_n"], paired["excluded_unresolved_pairs"]), (4, 1, 3))
        base = report["all_selected_by_arm"]["base"]["all_errors"]
        tuned = report["all_selected_by_arm"]["tuned"]["all_errors"]
        self.assertEqual((base["selected_n"], base["observed_n"], base["missing_n"]), (4, 2, 2))
        self.assertEqual(base["possible_all_selected_rate"], [0.25, 0.75])
        self.assertEqual((tuned["failed_n"], tuned["missing_n"]), (1, 1))
        self.assertEqual(tuned["possible_all_selected_rate"], [0.5, 1.0])

    def test_wilson_edges_and_empty_denominator(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2775327998628892)
        low, high = wilson_interval(10, 10)
        self.assertAlmostEqual(low, 0.7224672001371107)
        self.assertAlmostEqual(high, 1.0)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)
        with self.assertRaises(ValueError):
            wilson_interval(True, 1)

    def test_mcnemar_matches_exact_binomial(self):
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_pvalue(0, 5), 0.0625)
        self.assertEqual(exact_mcnemar_pvalue(5, 0), 0.0625)
        self.assertEqual(exact_mcnemar_pvalue(2, 3), 1.0)

    def test_all_positive_error_cohort_has_no_specificity_claim(self):
        report = build_comparison_report([row("a", 1, 0.1, 0.9)])
        self.assertIsNone(report["primary_paired"]["false_positive"]["tuned"]["rate"])
        self.assertTrue(any("always-positive" in item for item in report["limitations"]))
        self.assertIn("Not estimable (n=0)", render_markdown(report))

    def test_no_complete_pairs_is_not_reported_as_success(self):
        report = build_comparison_report([row("a", 1, 0.8, None)])
        primary = report["primary_paired"]["all_errors"]
        self.assertEqual(primary["paired_n"], 0)
        self.assertIsNone(primary["tuned"]["rate"])
        self.assertIsNone(primary["paired_outcomes"]["exact_mcnemar_pvalue"])
        self.assertIn("Not estimable", render_markdown(report))

    def test_base_only_results_are_visible_without_tuned_training_claim(self):
        report = build_comparison_report([
            row("a", 1, 0.9, None),
            row("b", 1, 0.1, None),
            row("c", 0, 0.1, None),
        ], metadata={"training_status": "blocked_not_run", "paid_inference_attempts": 3})
        self.assertEqual(report["comparison_status"], "unavailable_no_complete_pairs")
        base_fn = report["all_selected_by_arm"]["base"]["false_negative"]
        self.assertEqual(base_fn["observed"]["corrected"], 1)
        self.assertEqual(base_fn["observed"]["n"], 2)
        self.assertIsNotNone(base_fn["observed"]["wilson_95"])
        markdown = render_markdown(report)
        self.assertIn("1/2 (50.0%", markdown)
        self.assertIn("Weight fine-tuning was not run", markdown)
        self.assertIn("cannot be estimated", markdown)

    def test_rejects_invalid_numeric_scores_instead_of_dropping_them(self):
        for value in (True, "0.8", -0.1, 1.1, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_comparison_report([row("a", 1, value, 0.9)])

    def test_rejects_wrong_error_provenance_duplicate_ids_and_changed_threshold(self):
        invalid = [
            [row("a", 1, 0.2, 0.8, p_xgb=0.9)],
            [row("a", 1, 0.2, 0.8, error_type="false_positive")],
            [row("a", 1, 0.2, 0.8, source_split="validation")],
            [row("a", 1, 0.2, 0.8, jev_partition="tuning")],
            [row("a", 1, 0.2, 0.8), row("a", 0, 0.8, 0.2)],
            [row("a", 1, 0.2, 0.8, tuned_status="failed")],
        ]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                build_comparison_report(records)
        with self.assertRaises(ValueError):
            build_comparison_report([row("a", 1, 0.2, 0.8)], threshold=0.6)

    def test_outputs_omit_texts_and_are_valid_json(self):
        private_text = "This raw sample must never be exported."
        report = build_comparison_report([row("a", 1, 0.2, 0.8, text=private_text)], metadata={"dataset_sha256": "abc", "model_ids": ["base", "custom-model"]})
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_report(report, temporary)
            serialized = Path(paths["json"]).read_text()
            self.assertNotIn(private_text, serialized)
            self.assertNotIn(private_text, Path(paths["markdown"]).read_text())
            self.assertEqual(json.loads(serialized), report)
        with self.assertRaises(ValueError):
            build_comparison_report([row("a", 1, 0.2, 0.8)], metadata={"api_key": "never-export"})


if __name__ == "__main__":
    unittest.main()
