import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from jev_research.hybrid_reporting import (
    build_hybrid_report,
    render_hybrid_markdown,
    write_hybrid_report,
)


def table():
    labels = [1, 0, 1, 0, 1, 0]
    probabilities = [0.2, 0.8, 0.8, 0.2, 0.2, 0.2]
    return [
        {"row_id": index, "y_true": label, "p_xgb": probability,
         "source_split": "test", "text_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
         "dataset_sha256": "dataset", "baseline_fingerprint": "baseline",
         "text": f"Private raw sentence {index}; never export."}
        for index, (label, probability) in enumerate(zip(labels, probabilities))
    ]


def selected_rows(full=None):
    return [
        {**row, "sample_id": f"sample-{row['row_id']}"}
        for row in (full or table())[:4]
    ]


def outcomes():
    return [
        {"sample_id": "sample-0", "status": "succeeded", "score": 0.9},
        {"sample_id": "sample-1", "status": "succeeded", "score": 0.1},
        {"sample_id": "sample-2", "status": "succeeded", "score": 0.2},
        {"sample_id": "sample-3", "status": "failed", "score": None},
    ]


class HybridReportingTests(unittest.TestCase):
    def test_same_cases_include_corrections_regressions_and_failure_fallback(self):
        report = build_hybrid_report(table(), selected_rows(), outcomes())
        primary = report["selected_subset"]
        self.assertEqual(primary["n"], 4)
        self.assertEqual(primary["xgboost"]["correct"], 2)
        self.assertEqual(primary["hybrid"]["correct"], 3)
        self.assertEqual(primary["paired"]["corrected_xgboost_errors"], 2)
        self.assertEqual(primary["paired"]["introduced_errors"], 1)
        self.assertEqual(primary["paired"]["net_additional_correct"], 1)
        self.assertEqual(primary["paired"]["accuracy_change"], 0.25)
        self.assertEqual(primary["xgboost"]["confusion"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(primary["hybrid"]["confusion"], {"tn": 2, "fp": 0, "fn": 1, "tp": 1})
        self.assertAlmostEqual(primary["xgboost"]["brier_score"], 0.34)
        self.assertAlmostEqual(primary["hybrid"]["brier_score"], 0.175)
        self.assertAlmostEqual(primary["hybrid"]["roc_auc"], 0.875)
        self.assertEqual(report["routing"]["outcome_status_counts"],
                         {"succeeded": 3, "failed": 1, "pending": 0, "not_attempted": 0})
        self.assertEqual(report["routing"]["xgboost_fallback_selected_n"], 1)
        self.assertIn("fallback", report["coverage_status"])

    def test_full_test_changes_only_exact_selected_manifest(self):
        report = build_hybrid_report(table(), selected_rows(), outcomes())
        full = report["full_test_partial_policy"]
        self.assertEqual(full["n"], 6)
        self.assertEqual(full["xgboost"]["correct"], 3)
        self.assertEqual(full["hybrid"]["correct"], 4)
        self.assertEqual(full["paired"]["net_additional_correct"], 1)
        self.assertAlmostEqual(full["paired"]["accuracy_change"], 1 / 6)
        self.assertEqual(report["routing"]["unrouted_full_test_n"], 2)
        self.assertEqual(full["error_recovery"]["false_negative"]["n"], 2)
        self.assertEqual(full["error_recovery"]["false_negative"]["corrected"], 1)
        self.assertTrue(any("partial routing" in item for item in report["limitations"]))

    def test_missing_and_pending_outcomes_stay_in_denominators(self):
        rows = [outcomes()[0], {"sample_id": "sample-1", "status": "pending", "score": None}]
        report = build_hybrid_report(table(), selected_rows(), rows)
        self.assertEqual(report["selected_subset"]["n"], 4)
        self.assertEqual(report["selected_subset"]["hybrid"]["correct"], 3)
        self.assertEqual(report["routing"]["jev_success_n"], 1)
        self.assertEqual(report["routing"]["xgboost_fallback_selected_n"], 3)
        self.assertEqual(report["routing"]["outcome_status_counts"]["pending"], 1)
        self.assertEqual(report["routing"]["outcome_status_counts"]["not_attempted"], 2)

    def test_no_requests_is_operational_xgboost_fallback_not_jev_success(self):
        report = build_hybrid_report(table(), selected_rows(), [])
        primary = report["selected_subset"]
        self.assertEqual(primary["xgboost"], primary["hybrid"])
        self.assertEqual(primary["paired"]["net_additional_correct"], 0)
        self.assertEqual(primary["paired"]["exact_mcnemar_pvalue"], 1.0)
        self.assertEqual(report["routing"]["jev_success_n"], 0)
        self.assertEqual(report["routing"]["outcome_status_counts"]["not_attempted"], 4)

    def test_recovery_and_regression_rates_have_correct_conditional_denominators(self):
        report = build_hybrid_report(table(), selected_rows(), outcomes())
        rates = report["selected_subset"]["error_recovery"]
        self.assertEqual((rates["false_negative"]["corrected"], rates["false_negative"]["n"]), (1, 1))
        self.assertEqual((rates["false_positive"]["corrected"], rates["false_positive"]["n"]), (1, 1))
        self.assertEqual(rates["all_xgboost_errors"]["rate"], 1.0)
        self.assertEqual(rates["regression_among_xgboost_correct"]["rate"], 0.5)
        self.assertIsNotNone(rates["false_negative"]["wilson_95"])
        self.assertIsNotNone(report["selected_subset"]["hybrid"]["accuracy_wilson_95"])

    def test_class_error_flows_reconcile_corrections_new_misses_and_net_changes(self):
        report = build_hybrid_report(table(), selected_rows(), outcomes())
        for name in ("selected_subset", "full_test_partial_policy"):
            population = report[name]
            paired = population["paired"]
            self.assertEqual(paired["corrected_false_negatives"], 1)
            self.assertEqual(paired["corrected_false_positives"], 1)
            self.assertEqual(paired["introduced_false_negatives"], 1)
            self.assertEqual(paired["introduced_false_positives"], 0)
            self.assertEqual(paired["net_false_negative_reduction"], 0)
            self.assertEqual(paired["net_false_positive_reduction"], 1)
            self.assertEqual(paired["introduced_false_negatives"] + paired["introduced_false_positives"],
                             paired["introduced_errors"])
            for error, key in (("false_negative", "fn"), ("false_positive", "fp")):
                self.assertEqual(population["xgboost"]["confusion"][key]
                                 - paired[f"corrected_{error}s"] + paired[f"introduced_{error}s"],
                                 population["hybrid"]["confusion"][key])
                self.assertEqual(population["xgboost"]["confusion"][key]
                                 - population["hybrid"]["confusion"][key],
                                 paired[f"net_{error}_reduction"])
        markdown = render_hybrid_markdown(report)
        self.assertIn("Newly introduced | Hybrid errors | Net reduction", markdown)
        self.assertIn("Missed suicide cases (FN) | 1 | 1 | 1 | 1 | +0", markdown)
        self.assertIn("False alarms (FP) | 1 | 1 | 0 | 0 | +1", markdown)
        self.assertIn("Count / eligible", markdown)
        self.assertNotIn("Corrected / eligible", markdown)

    def test_exact_mcnemar_matches_discordant_pairs(self):
        full = [dict(table()[0], row_id=index, text_sha256=hashlib.sha256(str(index).encode()).hexdigest()) for index in range(5)]
        selected = [{**row, "sample_id": f"s{index}"} for index, row in enumerate(full)]
        scores = [{"sample_id": f"s{index}", "status": "succeeded", "score": 0.9} for index in range(5)]
        report = build_hybrid_report(full, selected, scores)
        self.assertEqual(report["selected_subset"]["paired"]["exact_mcnemar_pvalue"], 0.0625)
        self.assertIsNone(report["selected_subset"]["xgboost"]["roc_auc"])
        self.assertIsNone(report["selected_subset"]["error_recovery"]["false_positive"]["rate"])

    def test_rejects_misalignment_duplicates_or_non_test_provenance(self):
        bad_cases = []
        for field, value in (("row_id", 100), ("p_xgb", 0.25), ("y_true", 0),
                             ("source_split", "validation"), ("dataset_sha256", "other"),
                             ("text_sha256", "a" * 64)):
            selected = selected_rows()
            selected[0][field] = value
            bad_cases.append((table(), selected, outcomes()))
        bad_cases.append((table(), selected_rows() + [selected_rows()[0]], outcomes()))
        bad_cases.append((table() + [table()[0]], selected_rows(), outcomes()))
        bad_cases.append((table(), selected_rows(), outcomes() + [outcomes()[0]]))
        bad_cases.append((table(), selected_rows(), [{"sample_id": "unknown", "status": "succeeded", "score": 0.5}]))
        for full, selected, scores in bad_cases:
            with self.subTest(selected=selected[0]), self.assertRaises(ValueError):
                build_hybrid_report(full, selected, scores)

    def test_rejects_bad_scores_or_stale_failed_scores(self):
        for score in (True, "0.9", math.inf, -0.1, 1.1, math.nan, None):
            scores = copy.deepcopy(outcomes())
            scores[0]["score"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                build_hybrid_report(table(), selected_rows(), scores)
        scores = copy.deepcopy(outcomes())
        scores[0]["status"] = "failed"
        with self.assertRaises(ValueError):
            build_hybrid_report(table(), selected_rows(), scores)
        for threshold in (True, "0.5", 0.4):
            with self.assertRaises(ValueError):
                build_hybrid_report(table(), selected_rows(), outcomes(), threshold=threshold)

    def test_inputs_are_not_mutated_and_public_reports_contain_no_raw_text(self):
        full = table()
        selected = selected_rows(full)
        scores = outcomes()
        before = copy.deepcopy((full, selected, scores))
        report = build_hybrid_report(full, selected, scores, metadata={"dataset_sha256": "safe", "inference_budget": {"attempts_spent": 4}})
        self.assertEqual((full, selected, scores), before)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_hybrid_report(report, temporary)
            restored = json.loads(Path(paths["json"]).read_text())
            self.assertEqual(restored, report)
            for path in paths.values():
                content = Path(path).read_text()
                self.assertNotIn("Private raw sentence", content)
                self.assertNotIn("sample-0", content)
            markdown = Path(paths["markdown"]).read_text()
            self.assertIn("newly introduced errors: 1", markdown)
            self.assertIn("Full test: partial routing", markdown)
        with self.assertRaises(ValueError):
            build_hybrid_report(full, selected, scores, metadata={"api_key": "secret"})

    def test_does_not_force_gain_when_hybrid_regresses(self):
        full = table()[2:4]
        selected = [{**row, "sample_id": f"s{row['row_id']}"} for row in full]
        scores = [{"sample_id": "s2", "status": "succeeded", "score": 0.1},
                  {"sample_id": "s3", "status": "succeeded", "score": 0.9}]
        report = build_hybrid_report(full, selected, scores)
        paired = report["selected_subset"]["paired"]
        self.assertEqual(paired["corrected_xgboost_errors"], 0)
        self.assertEqual(paired["introduced_errors"], 2)
        self.assertEqual(paired["net_additional_correct"], -2)
        self.assertEqual(paired["accuracy_change"], -1.0)
        self.assertEqual(paired["introduced_false_negatives"], 1)
        self.assertEqual(paired["introduced_false_positives"], 1)
        self.assertEqual(paired["net_false_negative_reduction"], -1)
        self.assertEqual(paired["net_false_positive_reduction"], -1)


if __name__ == "__main__":
    unittest.main()
