"""Offline cost/safety checks. No test contacts the provider."""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import requests

from jev_research.client import (
    AttemptAlreadyRecorded, BudgetedJevClient, BudgetExhausted, CohortMismatch,
    HARD_MAX_ATTEMPTS, JevClientError, MissingCredential, text_sha256,
)


def cohort(size=4):
    return [{"sample_id": f"sample-{i}", "text_sha256": text_sha256(f"neutral example {i}"),
             "y_true": i % 2, "y_xgb": 1 - i % 2,
             "error_type": "false_negative" if i % 2 else "false_positive",
             "source_split": "test", "jev_partition": "evaluation"}
            for i in range(size)]


def successful_session(score=0.8):
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {"answers": {"department": {"score": score}}}
    session.post.return_value = response
    return session


def process_attempt(arguments):
    path, index = arguments
    client = BudgetedJevClient(path, cohort(12), max_attempts=3, session=successful_session())
    try:
        client.predict(f"sample-{index}", f"neutral example {index}")
        return "succeeded"
    except BudgetExhausted:
        return "blocked"


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "attempts.sqlite3"
        self.key_patch = patch.dict(os.environ, {"METIS_API_KEY": "offline-test-secret"})
        self.key_patch.start()

    def tearDown(self):
        self.key_patch.stop()
        self.temp.cleanup()

    def client(self, session=None, **kwargs):
        return BudgetedJevClient(self.path, cohort(), session=session or successful_session(), **kwargs)

    def test_success_caches_and_payload_has_no_labels(self):
        session = successful_session()
        client = self.client(session)
        self.assertEqual(client.predict("sample-0", "neutral example 0"), 0.8)
        with patch.dict(os.environ, {"METIS_API_KEY": ""}):
            self.assertEqual(client.predict("sample-0", "neutral example 0"), 0.8)
        self.assertEqual(session.post.call_count, 1)
        kwargs = session.post.call_args.kwargs
        self.assertEqual(set(kwargs["json"]), {"state", "model", "questions"})
        self.assertNotIn("y_true", json.dumps(kwargs["json"]))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (10.0, 60.0))
        self.assertEqual(client.ledger_summary()["attempts_spent"], 1)
        self.assertEqual(client.ledger_summary()["succeeded"], 1)
        for path in Path(self.temp.name).iterdir():
            if path.is_file():
                content = path.read_bytes()
                self.assertNotIn(b"offline-test-secret", content)
                self.assertNotIn(b"neutral example 0", content)

    def test_question_mutation_cannot_change_payload_without_cache_hash(self):
        session = successful_session()
        client = self.client(session)
        original = client.question["instructions"]
        client.question["instructions"] = "mutated instruction"
        client.predict("sample-0", "neutral example 0")
        self.assertEqual(session.post.call_args.kwargs["json"]["questions"]["department"]["instructions"], original)

    def test_missing_key_and_unauthorized_input_spend_nothing(self):
        session = successful_session()
        client = self.client(session)
        with patch.dict(os.environ, {"METIS_API_KEY": ""}):
            with self.assertRaises(MissingCredential):
                client.predict("sample-0", "neutral example 0")
        for sample_id, text, model in [("unknown", "neutral example 0", "jev-latest"),
                                      ("sample-0", "altered text", "jev-latest"),
                                      ("sample-0", "neutral example 0", "unapproved")]:
            with self.assertRaises(CohortMismatch):
                client.predict(sample_id, text, model=model)
        session.post.assert_not_called()
        self.assertEqual(client.ledger_summary()["attempts_spent"], 0)

    def test_correct_predictions_duplicate_texts_and_oversized_cohorts_rejected(self):
        rows = cohort()
        rows[0]["y_xgb"] = rows[0]["y_true"]
        with self.assertRaises(CohortMismatch):
            BudgetedJevClient(self.path, rows)
        rows = cohort()
        rows[1]["text_sha256"] = rows[0]["text_sha256"]
        with self.assertRaises(CohortMismatch):
            BudgetedJevClient(self.path, rows)
        with self.assertRaises(CohortMismatch):
            BudgetedJevClient(self.path, cohort(101))
        self.assertFalse(self.path.exists())

    def test_reservation_commits_before_network_and_failures_count(self):
        session = successful_session()
        client = self.client(session, max_attempts=1)
        def fail(*args, **kwargs):
            with sqlite3.connect(self.path) as db:
                self.assertEqual(db.execute("SELECT status FROM attempts").fetchone()[0], "pending")
            raise requests.Timeout("must-not-be-exposed offline-test-secret")
        session.post.side_effect = fail
        with self.assertRaises(JevClientError) as caught:
            client.predict("sample-0", "neutral example 0")
        self.assertNotIn("offline-test-secret", str(caught.exception))
        self.assertEqual(client.ledger_summary()["failed"], 1)
        with self.assertRaises(AttemptAlreadyRecorded):
            client.predict("sample-0", "neutral example 0")
        with self.assertRaises(BudgetExhausted):
            client.predict("sample-1", "neutral example 1")
        self.assertEqual(session.post.call_count, 1)

    def test_interrupt_is_uncertain_and_not_automatically_replayed(self):
        session = successful_session()
        session.post.side_effect = KeyboardInterrupt
        client = self.client(session)
        with self.assertRaises(KeyboardInterrupt):
            client.predict("sample-0", "neutral example 0")
        resumed = self.client(successful_session())
        self.assertEqual(resumed.ledger_summary()["pending_uncertain"], 1)
        with self.assertRaises(AttemptAlreadyRecorded):
            resumed.predict("sample-0", "neutral example 0")
        self.assertEqual(resumed.ledger_summary()["attempts_spent"], 1)

    def test_invalid_provider_responses_count_without_exposing_payload(self):
        invalid = [True, "0.8", float("nan"), float("inf"), -0.1, 1.1, None]
        for index, value in enumerate(invalid):
            with self.subTest(value=value):
                path = Path(self.temp.name) / f"invalid-{index}.sqlite3"
                client = BudgetedJevClient(path, cohort(), session=successful_session(value))
                with self.assertRaises(JevClientError):
                    client.predict("sample-0", "neutral example 0")
                self.assertEqual(client.ledger_summary()["failed"], 1)
        session = successful_session()
        session.post.return_value.status_code = 302
        client = self.client(session)
        with self.assertRaises(JevClientError) as caught:
            client.predict("sample-0", "neutral example 0")
        self.assertIn("302", str(caught.exception))
        self.assertEqual(client.ledger_summary()["attempts_spent"], 1)

    def test_offline_ledger_records_separate_models_and_need_no_credential(self):
        session = successful_session(0.7)
        client = self.client(session, allowed_models=("jev-latest", "custom"))
        client.predict("sample-0", "neutral example 0")
        client.predict("sample-0", "neutral example 0", model="custom")
        with patch.dict(os.environ, {"METIS_API_KEY": ""}):
            base = client.ledger_records()
            tuned = client.ledger_records(model="custom")
        self.assertEqual(base, [{"sample_id": "sample-0", "status": "succeeded", "score": 0.7, "error_code": None}])
        self.assertEqual(tuned, base)
        self.assertEqual(session.post.call_count, 2)
        with self.assertRaises(CohortMismatch):
            client.ledger_records(model="not-authorized")

    def test_offline_records_reject_tampered_score_and_request_checksum(self):
        client = self.client()
        client.predict("sample-0", "neutral example 0")
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE attempts SET score=1.5")
        with self.assertRaises(JevClientError):
            client.ledger_records()
        with self.assertRaises(JevClientError):
            client.predict("sample-0", "neutral example 0")
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE attempts SET score=0.7,request_key='tampered'")
        with self.assertRaises(CohortMismatch):
            client.ledger_records()

    def test_cohort_or_budget_cannot_be_reset_with_existing_ledger(self):
        client = self.client(max_attempts=2)
        client.predict("sample-0", "neutral example 0")
        changed = cohort()
        changed[1]["text_sha256"] = text_sha256("different")
        with self.assertRaises(CohortMismatch):
            BudgetedJevClient(self.path, changed)
        with self.assertRaises(CohortMismatch):
            self.client(max_attempts=3)
        with self.assertRaises(ValueError):
            self.client(max_attempts=HARD_MAX_ATTEMPTS + 1)

    def test_new_model_uses_same_ledger_budget_and_distinct_cache_key(self):
        first = self.client(max_attempts=2)
        first.predict("sample-0", "neutral example 0")
        session = successful_session(0.9)
        second = self.client(session, max_attempts=2, allowed_models=("jev-latest", "verified-custom-model"))
        self.assertEqual(second.predict("sample-0", "neutral example 0", model="verified-custom-model"), 0.9)
        self.assertEqual(second.ledger_summary()["attempts_spent"], 2)
        with self.assertRaises(BudgetExhausted):
            second.predict("sample-1", "neutral example 1", model="verified-custom-model")

    def test_concurrent_clients_do_not_exceed_cap(self):
        BudgetedJevClient(self.path, cohort(20), max_attempts=5, session=successful_session())
        count = 0
        mutex = threading.Lock()
        def task(index):
            nonlocal count
            session = successful_session()
            def record(*args, **kwargs):
                nonlocal count
                with mutex:
                    count += 1
                return successful_session().post()
            session.post.side_effect = record
            client = BudgetedJevClient(self.path, cohort(20), max_attempts=5, session=session)
            try:
                client.predict(f"sample-{index}", f"neutral example {index}")
                return "succeeded"
            except BudgetExhausted:
                return "blocked"
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            outcomes = list(executor.map(task, range(20)))
        self.assertEqual(outcomes.count("succeeded"), 5)
        self.assertEqual(count, 5)
        resumed = BudgetedJevClient(self.path, cohort(20), max_attempts=5, session=successful_session())
        self.assertEqual(resumed.ledger_summary()["attempts_spent"], 5)

    def test_processes_share_the_same_durable_cap(self):
        client = BudgetedJevClient(self.path, cohort(12), max_attempts=3, session=successful_session())
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            outcomes = list(executor.map(process_attempt, [(str(self.path), i) for i in range(12)]))
        self.assertEqual(outcomes.count("succeeded"), 3)
        self.assertEqual(outcomes.count("blocked"), 9)
        self.assertEqual(client.ledger_summary()["attempts_spent"], 3)

    def test_concurrent_duplicate_is_sent_once(self):
        started, release = threading.Event(), threading.Event()
        session = successful_session()
        def blocked(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return successful_session().post()
        session.post.side_effect = blocked
        client = self.client(session)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.predict, "sample-0", "neutral example 0")
            self.assertTrue(started.wait(timeout=5))
            with self.assertRaises(AttemptAlreadyRecorded):
                client.predict("sample-0", "neutral example 0")
            release.set()
            self.assertEqual(first.result(timeout=5), 0.8)
        self.assertEqual(session.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
