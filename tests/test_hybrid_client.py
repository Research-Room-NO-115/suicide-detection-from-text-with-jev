"""Offline checks for the separately authorized probability-routing budget."""
import concurrent.futures
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import requests

from jev_research.client import (
    AttemptAlreadyRecorded, BASE_MODEL, BudgetedJevClient, BudgetExhausted,
    CohortMismatch, JevClientError, ProbabilityHybridJevClient, text_sha256,
)


def rows(size=4, probability=0.5):
    return [{"sample_id": f"hybrid-{index}", "text_sha256": text_sha256(f"neutral routed text {index}"),
             "p_xgb": probability, "source_split": "test"} for index in range(size)]


def session(score=0.8):
    result = Mock()
    result.post.return_value = Mock(status_code=200)
    result.post.return_value.json.return_value = {"answers": {"department": {"score": score}}}
    return result


class HybridClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'hybrid.sqlite3'
        self.key = patch.dict(os.environ, {"METIS_API_KEY": "offline-hybrid-key"})
        self.key.start()

    def tearDown(self):
        self.key.stop()
        self.temp.cleanup()

    def client(self, cohort=None, **kwargs):
        return ProbabilityHybridJevClient(self.path, cohort if cohort is not None else rows(),
                                          interval=(0.2, 0.8), **kwargs)

    def test_routes_correct_and_incorrect_cases_without_gold_labels(self):
        selected = rows()
        selected[0].update(y_true=1, y_xgb=1, error_type='correct')
        selected[1].update(y_true=0, y_xgb=1, error_type='false_positive')
        transport = session()
        client = self.client(selected, session=transport)
        self.assertEqual(client.predict('hybrid-0', 'neutral routed text 0'), 0.8)
        payload = transport.post.call_args.kwargs['json']
        self.assertEqual(set(payload), {'state', 'model', 'questions'})
        self.assertEqual(payload['model'], BASE_MODEL)
        self.assertFalse(transport.post.call_args.kwargs['allow_redirects'])
        with patch.dict(os.environ, {'METIS_API_KEY': ''}):
            self.assertEqual(client.predict('hybrid-0', 'neutral routed text 0'), 0.8)
            self.assertEqual(client.ledger_records()[0]['score'], 0.8)
        self.assertEqual(transport.post.call_count, 1)
        self.assertEqual(client.ledger_summary()['max_attempts'], 5000)

    def test_old_error_study_limits_and_ledger_are_unchanged(self):
        old_rows = [{'sample_id': 'old-1', 'text_sha256': text_sha256('old neutral text'),
                     'y_true': 0, 'y_xgb': 1, 'error_type': 'false_positive'}]
        old_path = Path(self.temp.name) / 'old.sqlite3'
        old = BudgetedJevClient(old_path, old_rows, session=session())
        old.predict('old-1', 'old neutral text')
        original = old.ledger_summary()
        with self.assertRaises(ValueError):
            BudgetedJevClient(old_path, old_rows, max_attempts=101)
        new = self.client(session=session())
        new.predict('hybrid-0', 'neutral routed text 0')
        self.assertEqual(old.ledger_summary(), original)
        with self.assertRaises(CohortMismatch):
            ProbabilityHybridJevClient(old_path, rows(), interval=(0.2, 0.8))
        self.assertEqual(original['max_attempts'], 100)

    def test_interval_boundaries_and_upper_one_are_explicit(self):
        self.client(rows(probability=0.2), session=session())
        for probability in (0.199999, 0.8, 0.9, float('nan'), float('inf'), True, '0.5'):
            with self.subTest(probability=probability), self.assertRaises(CohortMismatch):
                self.client(rows(probability=probability), session=session())
        final = ProbabilityHybridJevClient(Path(self.temp.name) / 'final.sqlite3', rows(probability=1.0),
                                           interval=(0.8, 1.0), session=session())
        self.assertEqual(final.predict('hybrid-0', 'neutral routed text 0'), 0.8)
        for interval in ((0.8, 0.2), (0.2, 0.2), (-0.1, 0.5), (0.5, 1.1),
                         (float('nan'), 1), (False, 1), (0.1,), None):
            with self.subTest(interval=interval), self.assertRaises(CohortMismatch):
                ProbabilityHybridJevClient(self.path, rows(), interval=interval)

    def test_non_test_duplicate_and_oversized_cohorts_are_rejected(self):
        invalid = rows()
        invalid[0]['source_split'] = 'validation'
        with self.assertRaises(CohortMismatch):
            self.client(invalid)
        invalid = rows()
        invalid[1]['text_sha256'] = invalid[0]['text_sha256']
        with self.assertRaises(CohortMismatch):
            self.client(invalid)
        with self.assertRaises(CohortMismatch):
            self.client(rows(5001))
        with self.assertRaises(ValueError):
            self.client(max_attempts=5001)
        with self.assertRaises(ValueError):
            self.client(max_attempts=True)

    def test_interval_and_routing_scores_are_part_of_durable_authorization(self):
        first = self.client(session=session())
        first.predict('hybrid-0', 'neutral routed text 0')
        with self.assertRaises(CohortMismatch):
            ProbabilityHybridJevClient(self.path, rows(), interval=(0.1, 0.9))
        changed = rows()
        changed[0]['p_xgb'] = 0.55
        with self.assertRaises(CohortMismatch):
            self.client(changed)
        # Gold labels never determine routing and are not in the fingerprint.
        changed = rows()
        changed[0].update(y_true=0, y_xgb=0)
        resumed = self.client(changed, session=session())
        self.assertEqual(resumed.cohort_fingerprint, first.cohort_fingerprint)

    def test_hard_5000_cap_blocks_before_the_next_network_request(self):
        selected = rows(5000)
        transport = session()
        client = self.client(selected, session=transport, allowed_models=(BASE_MODEL, 'other-model'))
        # Simulate 4,999 already sent/uncertain attempts without making model calls.
        with sqlite3.connect(self.path) as db:
            db.executemany('''INSERT INTO attempts
                (request_key,sample_id,text_sha256,model,instruction_sha256,status,started_at)
                VALUES (?,?,?,?,?,'pending','offline-test')''',
                [(client._request_key(row['sample_id'], row['text_sha256'], BASE_MODEL),
                  row['sample_id'], row['text_sha256'], BASE_MODEL, client.instruction_fingerprint)
                 for row in selected[:-1]])
        self.assertEqual(client.predict('hybrid-4999', 'neutral routed text 4999'), 0.8)
        with self.assertRaises(BudgetExhausted):
            client.predict('hybrid-4999', 'neutral routed text 4999', model='other-model')
        self.assertEqual(transport.post.call_count, 1)
        self.assertEqual(client.ledger_summary()['attempts_spent'], 5000)
        self.assertEqual(client.ledger_summary()['unique_samples_attempted'], 5000)

    def test_failure_and_interrupt_still_consume_attempts_without_retry(self):
        transport = session()
        transport.post.side_effect = requests.Timeout('private diagnostic must not escape')
        client = self.client(session=transport)
        with self.assertRaises(JevClientError) as caught:
            client.predict('hybrid-0', 'neutral routed text 0')
        self.assertNotIn('private diagnostic', str(caught.exception))
        with self.assertRaises(AttemptAlreadyRecorded):
            client.predict('hybrid-0', 'neutral routed text 0')
        self.assertEqual(client.ledger_summary()['failed'], 1)
        transport.post.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            client.predict('hybrid-1', 'neutral routed text 1')
        self.assertEqual(client.ledger_summary()['pending_uncertain'], 1)
        self.assertEqual(client.ledger_summary()['attempts_spent'], 2)
        self.assertEqual(transport.post.call_count, 2)

    def test_concurrent_worker_clients_share_cap_and_distinct_sessions(self):
        selected = rows(20)
        seed = self.client(selected, max_attempts=5, session=session())
        calls = []
        lock = threading.Lock()
        def task(index):
            transport = session()
            def post(*args, **kwargs):
                with lock: calls.append(index)
                return session().post()
            transport.post.side_effect = post
            client = self.client(selected, max_attempts=5, session=transport)
            try:
                client.predict(f'hybrid-{index}', f'neutral routed text {index}')
                return True
            except BudgetExhausted:
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(task, range(20)))
        self.assertEqual(sum(outcomes), 5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(seed.ledger_summary()['attempts_spent'], 5)


if __name__ == '__main__':
    unittest.main()
