"""Selection tests use neutral local fixtures and never call Jev."""
import hashlib
import unittest

import pandas as pd

from jev_research.selection import (
    InsufficientErrorsError, select_error_cohort, validate_error_cohort,
)


def frame(split, offset, fn=45, fp=45, correct=8):
    rows = []
    for position, error in enumerate(["false_negative"] * fn + ["false_positive"] * fp + ["correct"] * correct):
        row_id = offset + position
        text = f"Neutral fixture paragraph number {row_id}."
        label = 1 if error == "false_negative" else 0
        probability = 0.2 if error != "false_positive" else 0.8
        rows.append({
            "row_id": row_id, "text": text, "cleaned_text": text.lower(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "y_true": label, "p_xgb": probability, "y_xgb": int(probability >= 0.5),
            "error_type": error, "source_split": split,
            "dataset_sha256": hashlib.sha256(b"dataset").hexdigest(),
            "baseline_fingerprint": hashlib.sha256(b"baseline").hexdigest(),
        })
    return pd.DataFrame(rows)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.validation = frame("validation", 0)
        self.test = frame("test", 1000)

    def test_exact_budget_and_no_correct_examples(self):
        result = select_error_cohort(self.validation, self.test)
        self.assertEqual(len(result.cohort), 100)
        self.assertEqual(result.cohort["sample_id"].nunique(), 100)
        self.assertTrue((result.cohort["y_true"] != result.cohort["y_xgb"]).all())
        self.assertEqual(result.cohort["jev_role"].value_counts().to_dict(),
                         {"train": 40, "evaluation": 40, "validation": 20})
        self.assertNotIn("text", result.manifest.columns)
        self.assertNotIn("cleaned_text", result.manifest.columns)

    def test_ties_and_input_permutations_do_not_change_selection(self):
        first = select_error_cohort(self.validation, self.test)
        second = select_error_cohort(
            self.validation.sample(frac=1, random_state=7),
            self.test.sample(frac=1, random_state=8),
        )
        pd.testing.assert_frame_equal(first.manifest, second.manifest)
        self.assertEqual(first.metadata["manifest_sha256"], second.metadata["manifest_sha256"])
        different = select_error_cohort(self.validation, self.test, seed=43)
        self.assertNotEqual(first.metadata["manifest_sha256"], different.metadata["manifest_sha256"])

    def test_training_never_uses_xgboost_test_rows(self):
        cohort = select_error_cohort(self.validation, self.test).cohort
        for role in ("train", "validation"):
            rows = cohort.loc[cohort["jev_role"].eq(role)]
            self.assertTrue(rows["source_split"].eq("validation").all())
            self.assertEqual(set(rows["error_type"]), {"false_negative", "false_positive"})
        evaluation = cohort.loc[cohort["jev_role"].eq("evaluation")]
        self.assertTrue(evaluation["source_split"].eq("test").all())
        self.assertFalse(set(cohort.loc[cohort["jev_partition"].eq("tuning"), "row_id"]) & set(evaluation["row_id"]))

    def test_false_negative_and_false_positive_definitions(self):
        cohort = select_error_cohort(self.validation, self.test).cohort
        negatives = cohort.loc[cohort["error_type"].eq("false_negative")]
        positives = cohort.loc[cohort["error_type"].eq("false_positive")]
        self.assertTrue(negatives["y_true"].eq(1).all() and negatives["y_xgb"].eq(0).all())
        self.assertTrue(positives["y_true"].eq(0).all() and positives["y_xgb"].eq(1).all())

    def test_balanced_quotas_are_independent_of_natural_error_mix(self):
        result = select_error_cohort(
            frame("validation", 0, fn=100, fp=35),
            frame("test", 1000, fn=60, fp=22),
        )
        for role, per_class in (("train", 20), ("validation", 10), ("evaluation", 20)):
            counts = result.cohort.loc[result.cohort["jev_role"].eq(role), "error_type"].value_counts().to_dict()
            self.assertEqual(counts, {"false_negative": per_class, "false_positive": per_class})
        self.assertIn("artificial 50/50", result.metadata["sampling"])

    def test_insufficient_minority_error_quota_has_no_fallback(self):
        with self.assertRaisesRegex(InsufficientErrorsError, "Balanced quotas are not relaxed"):
            select_error_cohort(frame("validation", 0, fn=100, fp=29), self.test)

    def test_only_false_negatives_is_rejected_for_binary_tuning(self):
        with self.assertRaisesRegex(InsufficientErrorsError, "both false negatives and false positives"):
            select_error_cohort(frame("validation", 0, fn=100, fp=0), self.test)

    def test_insufficient_errors_does_not_silently_reduce_budget(self):
        with self.assertRaises(InsufficientErrorsError):
            select_error_cohort(frame("validation", 0, fn=20, fp=20), self.test)

    def test_overlap_and_mismatched_provenance_rejected(self):
        duplicated = self.test.copy()
        duplicated.loc[0, "row_id"] = self.validation.loc[0, "row_id"]
        with self.assertRaisesRegex(ValueError, "row IDs overlap"):
            select_error_cohort(self.validation, duplicated)
        unrelated = self.test.copy()
        unrelated["baseline_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "different baseline_fingerprint"):
            select_error_cohort(self.validation, unrelated)

    def test_corrupted_labels_and_thresholds_rejected(self):
        corrupted = self.validation.copy()
        corrupted.loc[0, "error_type"] = "correct"
        with self.assertRaisesRegex(ValueError, "error_type"):
            select_error_cohort(corrupted, self.test)
        corrupted = self.validation.copy()
        corrupted.loc[0, "p_xgb"] = 0.7
        with self.assertRaisesRegex(ValueError, "fixed XGBoost threshold"):
            select_error_cohort(corrupted, self.test)

    def test_cap_cannot_be_raised_or_split_exceeded(self):
        with self.assertRaises(ValueError):
            select_error_cohort(self.validation, self.test, max_samples=101)
        with self.assertRaises(ValueError):
            select_error_cohort(self.validation, self.test, tuning_size=61, evaluation_size=40)

    def test_manifest_replay_and_tampered_partition(self):
        result = select_error_cohort(self.validation, self.test)
        validate_error_cohort(result.manifest, self.validation, self.test)
        tampered = result.manifest.copy()
        tampered.loc[0, "jev_role"] = "evaluation"
        with self.assertRaisesRegex(ValueError, "jev_role"):
            validate_error_cohort(tampered, self.validation, self.test)
        reordered = result.manifest.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "deterministic selection"):
            validate_error_cohort(reordered, self.validation, self.test)


if __name__ == "__main__":
    unittest.main()
