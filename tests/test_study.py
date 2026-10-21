"""Offline integration checks for immutable study selection and paid-call scope."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from jev_research.client import JevClientError
from jev_research.study import (
    MANIFEST_COLUMNS, cohort_digest, load_study, prepare_study,
    report_saved_results, run_base_evaluation,
)


def baseline_frame(split, offset):
    records = []
    for index in range(120):
        truth = index % 2
        text = f"private neutral example {split} {index}"
        records.append({
            "row_id": offset + index, "text": text, "cleaned_text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "y_true": truth, "p_xgb": 0.2 if truth else 0.8, "y_xgb": 1 - truth,
            "error_type": "false_negative" if truth else "false_positive",
            "source_split": split, "dataset_sha256": "d" * 64,
            "baseline_fingerprint": "b" * 64,
        })
    return pd.DataFrame(records)


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.artifacts = Path(self.temp.name) / "private"
        self.results = Path(self.temp.name) / "public"
        baseline = SimpleNamespace(
            frames={"validation": baseline_frame("validation", 0),
                    "test": baseline_frame("test", 1000)},
            metadata={"baseline_fingerprint": "b" * 64,
                      "identity": {"dataset_sha256": "d" * 64}},
        )
        self.baseline_patch = patch("jev_research.study.build_baseline", return_value=baseline)
        self.baseline_patch.start()
        prepare_study("ignored.csv", self.artifacts, self.results)

    def tearDown(self):
        self.baseline_patch.stop()
        self.temp.cleanup()

    def test_prepare_has_disjoint_40_20_40_roles_and_no_public_raw_text(self):
        cohort, metadata = load_study(self.artifacts)
        self.assertEqual(len(cohort), 100)
        self.assertEqual(cohort.jev_role.value_counts().to_dict(), {"train": 40, "evaluation": 40, "validation": 20})
        for role in ("train", "validation"):
            rows = [json.loads(line) for line in (self.artifacts / f"{role}.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 40 if role == "train" else 20)
            self.assertEqual({record["label"] for record in rows}, {0, 1})
            self.assertTrue(set(record["sample_id"] for record in rows).isdisjoint(
                set(cohort.loc[cohort.jev_role.eq("evaluation"), "sample_id"])))
        self.assertTrue(cohort.loc[cohort.jev_role.isin(["train", "validation"]), "source_split"].eq("validation").all())
        self.assertTrue(cohort.loc[cohort.jev_role.eq("evaluation"), "source_split"].eq("test").all())
        for path in self.results.iterdir():
            self.assertNotIn("private neutral example", path.read_text())
        status = json.loads((self.results / "training_status.json").read_text())
        self.assertEqual(status["status"], "blocked_not_run")
        self.assertEqual(status["training_jobs_created"], 0)
        self.assertEqual(metadata["max_paid_inference_attempts"], 100)

    def test_prepare_replay_is_deterministic_and_preserves_study(self):
        before = (self.artifacts / "study.json").read_text()
        manifest = (self.results / "selection_manifest.json").read_text()
        prepare_study("ignored.csv", self.artifacts, self.results)
        self.assertEqual(before, (self.artifacts / "study.json").read_text())
        self.assertEqual(manifest, (self.results / "selection_manifest.json").read_text())

    def test_metadata_tampering_is_detected_before_client_creation(self):
        cohort, _ = load_study(self.artifacts)
        cohort.loc[0, "jev_role"] = "evaluation"
        cohort.to_csv(self.artifacts / "cohort.csv.gz", index=False)
        with patch("jev_research.client.BudgetedJevClient") as client:
            with self.assertRaises(ValueError):
                run_base_evaluation(self.artifacts, self.results)
            client.assert_not_called()

    def test_text_tampering_is_detected_before_any_provider_call(self):
        cohort, _ = load_study(self.artifacts)
        cohort.loc[99, "text"] = "tampered private text"
        cohort.to_csv(self.artifacts / "cohort.csv.gz", index=False)
        with patch("jev_research.client.BudgetedJevClient") as client:
            with self.assertRaises(ValueError):
                run_base_evaluation(self.artifacts, self.results)
            client.assert_not_called()

    def successful_transport(self):
        cohort, _ = load_study(self.artifacts)
        by_text = cohort.set_index("text").y_true.to_dict()
        session = Mock()
        def response(*args, **kwargs):
            score = 0.8 if by_text[kwargs["json"]["state"]] else 0.2
            result = Mock(status_code=200)
            result.json.return_value = {"answers": {"department": {"score": score}}}
            return result
        session.post.side_effect = response
        return session

    def test_base_evaluation_only_calls_the_40_prespecified_test_errors(self):
        cohort, _ = load_study(self.artifacts)
        expected = cohort.loc[cohort.jev_role.eq("evaluation")]
        session = self.successful_transport()
        with patch("jev_research.client.requests.Session", return_value=session), \
                patch.dict(os.environ, {"METIS_API_KEY": "offline-study-key"}):
            summary = run_base_evaluation(self.artifacts, self.results)
        self.assertEqual(summary["attempts_spent"], 40)
        calls = session.post.call_args_list
        self.assertEqual(len(calls), 40)
        self.assertEqual({call.kwargs["json"]["state"] for call in calls}, set(expected.text))
        public = pd.read_json(self.results / "base_predictions.json")
        self.assertEqual(len(public), 40)
        self.assertNotIn("text", public.columns)
        self.assertTrue(public.base_status.eq("succeeded").all())
        report = json.loads((self.results / "report.json").read_text())
        self.assertEqual(report["comparison_status"], "unavailable_no_complete_pairs")
        self.assertEqual(report["primary_paired"]["all_errors"]["paired_n"], 0)
        self.assertEqual(report["all_selected_by_arm"]["base"]["all_errors"]["observed_n"], 40)
        self.assertEqual(report["all_selected_by_arm"]["tuned"]["all_errors"]["missing_n"], 40)
        # A full CLI rerun reads all 40 successes from the same durable ledger.
        with patch("jev_research.client.requests.Session", return_value=session), \
                patch.dict(os.environ, {"METIS_API_KEY": ""}):
            replay = run_base_evaluation(self.artifacts, self.results)
        self.assertEqual(replay["attempts_spent"], 40)
        self.assertEqual(session.post.call_count, 40)

    def test_first_failure_stops_paid_work_and_does_not_trust_stale_scores(self):
        cohort, _ = load_study(self.artifacts)
        evaluation = cohort.loc[cohort.jev_role.eq("evaluation")]
        stale_id = evaluation.iloc[-1].sample_id
        (self.artifacts / "base_outcomes.json").write_text(json.dumps({
            stale_id: {"status": "succeeded", "score": 0.9}}))
        session = Mock()
        session.post.side_effect = requests.Timeout("Offline synthetic failure")
        with patch("jev_research.client.requests.Session", return_value=session), \
                patch.dict(os.environ, {"METIS_API_KEY": "offline-study-key"}):
            with self.assertRaises(RuntimeError):
                run_base_evaluation(self.artifacts, self.results)
        self.assertEqual(session.post.call_count, 1)
        public = pd.read_json(self.results / "base_predictions.json")
        self.assertFalse(public.base_status.eq("succeeded").any())
        self.assertTrue(public.base_score.isna().all())
        report = json.loads((self.results / "report.json").read_text())
        self.assertEqual(report["all_selected_by_arm"]["base"]["all_errors"]["failed_n"], 1)
        self.assertEqual(report["all_selected_by_arm"]["base"]["all_errors"]["missing_n"], 39)
        # A failed request remains spent and is not retried by rerunning the study.
        with patch("jev_research.client.requests.Session", return_value=session), \
                patch.dict(os.environ, {"METIS_API_KEY": "offline-study-key"}):
            with self.assertRaises(RuntimeError):
                run_base_evaluation(self.artifacts, self.results)
        self.assertEqual(session.post.call_count, 1)

    def test_offline_report_ignores_mutated_public_labels_and_scores(self):
        session = self.successful_transport()
        with patch("jev_research.client.requests.Session", return_value=session), \
                patch.dict(os.environ, {"METIS_API_KEY": "offline-study-key"}):
            run_base_evaluation(self.artifacts, self.results)
        before = json.loads((self.results / "base_predictions.json").read_text())
        mutation = json.loads(json.dumps(before))
        mutation[0]["y_true"] = 1 - mutation[0]["y_true"]
        mutation[0]["base_score"] = 0.123
        (self.results / "base_predictions.json").write_text(json.dumps(mutation))
        with patch("jev_research.client.requests.Session", return_value=session), \
                patch.dict(os.environ, {"METIS_API_KEY": ""}):
            report_saved_results(self.artifacts, self.results)
        after = json.loads((self.results / "base_predictions.json").read_text())
        self.assertEqual(after, before)
        self.assertEqual(session.post.call_count, 40)


if __name__ == "__main__":
    unittest.main()
