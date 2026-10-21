"""Offline orchestration tests; all provider requests use neutral mocked texts."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from jev_research.client import MissingCredential
from jev_research.probability_hybrid import (
    load_probability_hybrid, prepare_probability_hybrid,
    report_probability_hybrid, run_probability_hybrid,
)


def frame(split, size, offset):
    records = []
    for index in range(size):
        text = f'private neutral probability text {split} {index}'
        truth, probability = index % 2, 0.5
        predicted = int(probability >= 0.5)
        records.append({
            'row_id': offset + index, 'text': text, 'cleaned_text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'p_xgb': probability, 'y_true': truth, 'y_xgb': predicted,
            'error_type': 'correct' if predicted == truth else 'false_positive',
            'source_split': split, 'dataset_sha256': 'd' * 64,
            'baseline_fingerprint': 'b' * 64,
        })
    return pd.DataFrame(records)


class ProbabilityHybridIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'probability-private'
        self.results = self.root / 'probability-public'
        self.old_ledger = self.root / 'previous-study' / 'budget.sqlite3'
        self.old_ledger.parent.mkdir()
        self.old_ledger.write_bytes(b'previous 100-attempt study must remain unchanged')
        self.old_digest = hashlib.sha256(self.old_ledger.read_bytes()).hexdigest()
        self.validation = frame('validation', 12, 0)
        self.test = frame('test', 12, 100)
        self.baseline = SimpleNamespace(
            frames={'validation': self.validation, 'test': self.test},
            metadata={'identity': {'dataset_sha256': 'd' * 64},
                      'baseline_fingerprint': 'b' * 64},
            cache_dir=self.root / 'baseline-cache',
        )
        self.stack = ExitStack()
        # Shrink dispatch-only integration fixtures; production constants and
        # the client's separate 5,000 cap are never increased or bypassed.
        self.stack.enter_context(patch('jev_research.probability_hybrid.TARGET_SIZE', 8))
        self.stack.enter_context(patch('jev_research.probability_hybrid.build_baseline', return_value=self.baseline))
        self.full_test = self.stack.enter_context(
            patch('jev_research.probability_hybrid._load_full_test', return_value=self.test))
        self.stack.enter_context(patch.dict(os.environ, {'METIS_API_KEY': 'offline-probability-secret'}))
        prepare_probability_hybrid('unused.csv', self.artifacts, self.results)
        self.calls = []
        self.lock = threading.Lock()
        self.active = self.max_active = 0

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def transport(self, *, failure=False, delay=0):
        def factory():
            session = Mock()
            def post(*args, **kwargs):
                with self.lock:
                    self.calls.append(kwargs['json'])
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    if delay:
                        time.sleep(delay)
                    if failure:
                        raise requests.Timeout('private diagnostic must not be published')
                    return SimpleNamespace(status_code=200,
                        json=lambda: {'answers': {'department': {'score': 0.8}}})
                finally:
                    with self.lock:
                        self.active -= 1
            session.post.side_effect = post
            return session
        return patch('jev_research.client.requests.Session', side_effect=factory)

    def run_paid_mock(self, **kwargs):
        return run_probability_hybrid(self.artifacts, self.results,
                                      authorized_attempts=8, **kwargs)

    def test_exact_5000_selection_is_label_blind_and_retains_correct_cases(self):
        # Exercise the real requested cohort size without 5,000 mock network calls.
        full_validation, full_test = frame('validation', 6000, 0), frame('test', 6000, 10000)
        self.baseline.frames = {'validation': full_validation, 'test': full_test}
        path, public = self.root / 'full-private', self.root / 'full-public'
        with patch('jev_research.probability_hybrid.TARGET_SIZE', 5000):
            prepare_probability_hybrid('unused.csv', path, public)
            selected, study, _ = load_probability_hybrid(path)
            self.assertEqual(len(selected), 5000)
            self.assertTrue(selected.y_true.eq(selected.y_xgb).any())
            self.assertTrue(selected.y_true.ne(selected.y_xgb).any())
            original_ids = selected.sample_id.tolist()
            original_digest = study['cohort_sha256']
            full_test['y_true'] = 1 - full_test.y_true
            full_test['error_type'] = 'deliberately changed labels must not select cases'
            prepare_probability_hybrid('unused.csv', path, public)
            changed, metadata, _ = load_probability_hybrid(path)
            self.assertEqual(changed.sample_id.tolist(), original_ids)
            self.assertEqual(metadata['cohort_sha256'], original_digest)
            self.assertEqual(metadata['cohort_locked_at_utc'], study['cohort_locked_at_utc'])
        self.assertFalse((path / 'budget.sqlite3').exists())

    def test_prepare_preserves_lock_and_exports_no_private_text_or_credential(self):
        before = json.loads((self.artifacts / 'study.json').read_text())
        prepare_probability_hybrid('unused.csv', self.artifacts, self.results)
        after = json.loads((self.artifacts / 'study.json').read_text())
        self.assertEqual(after, before)
        self.assertEqual(before['target_size'], 8)
        self.assertEqual(before['cohort_selection']['test_labels_used_for_selection'], False)
        for path in self.results.iterdir():
            content = path.read_text()
            self.assertNotIn('private neutral probability text', content)
            self.assertNotIn('offline-probability-secret', content)
        self.assertFalse((self.artifacts / 'budget.sqlite3').exists())

    def test_raw_text_and_manifest_tampering_fail_before_client_or_network(self):
        original = (self.artifacts / 'cohort.csv.gz').read_bytes()
        for field, replacement in (('text', 'modified raw text'), ('p_xgb', 0.7)):
            with self.subTest(field=field):
                table = pd.read_csv(self.artifacts / 'cohort.csv.gz')
                table.loc[0, field] = replacement
                table.to_csv(self.artifacts / 'cohort.csv.gz', index=False)
                with patch('jev_research.probability_hybrid.ProbabilityHybridJevClient') as client:
                    with self.assertRaises(ValueError):
                        self.run_paid_mock()
                    client.assert_not_called()
                (self.artifacts / 'cohort.csv.gz').write_bytes(original)

    def test_invalid_baseline_evidence_stops_before_any_paid_attempt(self):
        with self.transport(), patch('jev_research.probability_hybrid._load_full_test',
                                     side_effect=ValueError('cache checksum failed')):
            with self.assertRaisesRegex(ValueError, 'cache checksum'):
                self.run_paid_mock(workers=2)
        self.assertEqual(self.calls, [])
        # Initialization may create a ledger, but no reservation was spent.
        import sqlite3
        with sqlite3.connect(self.artifacts / 'budget.sqlite3') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0], 0)

    def test_bounded_dispatch_caches_and_preserves_previous_study_budget(self):
        with self.transport(delay=0.02):
            first = self.run_paid_mock(workers=3)
            self.assertEqual(first['attempts_spent'], 8)
            self.assertEqual(first['succeeded'], 8)
            self.assertEqual(len(self.calls), 8)
            self.assertLessEqual(self.max_active, 3)
            for payload in self.calls:
                self.assertEqual(set(payload), {'state', 'model', 'questions'})
            with patch.dict(os.environ, {'METIS_API_KEY': ''}):
                second = self.run_paid_mock(workers=3)
            self.assertEqual(first, second)
            self.assertEqual(len(self.calls), 8)
        self.assertEqual(hashlib.sha256(self.old_ledger.read_bytes()).hexdigest(), self.old_digest)

    def test_failure_stops_then_deliberate_resume_skips_failure_and_falls_back(self):
        with self.transport(failure=True):
            with self.assertRaisesRegex(RuntimeError, 'stopped'):
                self.run_paid_mock(workers=1)
        self.assertEqual(len(self.calls), 1)
        partial = json.loads((self.results / 'hybrid_report.json').read_text())
        self.assertEqual(partial['routing']['jev_success_n'], 0)
        self.assertEqual(partial['routing']['outcome_status_counts']['failed'], 1)
        self.assertEqual(partial['selected_subset']['xgboost'], partial['selected_subset']['hybrid'])
        first_text = self.calls[0]['state']
        with self.transport():
            resumed = self.run_paid_mock(workers=1)
        self.assertEqual(resumed['attempts_spent'], 8)
        self.assertEqual(resumed['succeeded'], 7)
        self.assertEqual(resumed['failed'], 1)
        self.assertEqual(sum(call['state'] == first_text for call in self.calls), 1)
        report = json.loads((self.results / 'hybrid_report.json').read_text())
        self.assertEqual(report['routing']['xgboost_fallback_selected_n'], 1)
        self.assertEqual(report['selected_subset']['n'], 8)

    def test_offline_report_uses_ledger_and_source_not_forged_public_predictions(self):
        with self.transport():
            self.run_paid_mock(workers=2)
        expected = json.loads((self.results / 'hybrid_report.json').read_text())
        path = self.results / 'predictions.json'
        records = json.loads(path.read_text())
        for row in records:
            row.update(y_true=1 - row['y_true'], jev_score=0.0, jev_status='succeeded')
        path.write_text(json.dumps(records))
        with patch('jev_research.client.requests.Session') as network:
            actual = report_probability_hybrid(self.artifacts, self.results)
            for call in network.return_value.post.call_args_list:
                self.fail('Offline reporting unexpectedly used a provider request')
        self.assertEqual(actual, expected)
        refreshed = json.loads(path.read_text())
        self.assertTrue(all(row['jev_score'] == 0.8 for row in refreshed))
        for public in self.results.iterdir():
            self.assertNotIn('private neutral probability text', public.read_text())

    def test_missing_key_and_unapproved_budget_send_nothing(self):
        with self.transport(), patch.dict(os.environ, {'METIS_API_KEY': ''}):
            with self.assertRaises(MissingCredential):
                self.run_paid_mock(workers=2)
        self.assertEqual(self.calls, [])
        for authorization in (100, 5001, True):
            with self.subTest(authorization=authorization), self.assertRaises(ValueError):
                run_probability_hybrid(self.artifacts, self.results, authorized_attempts=authorization)
        for workers in (0, 9, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.run_paid_mock(workers=workers)


if __name__ == '__main__':
    unittest.main()
