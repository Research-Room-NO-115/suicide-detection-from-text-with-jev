"""Mocked inference only. No test sends a provider request."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from jev_research.client import (AttemptAlreadyRecorded, BudgetExhausted,
    CohortMismatch, JevClientError, MissingCredential, text_sha256)
from jev_research.additional_experiment import (
    AdditionalJevClient, prepare_additional_plan, manifest_hash,
    collect_additional_phase, freeze_additional_policies, analyze_additional,
)


def frames_fixture(n_per_stratum=180):
    frames = {}
    row_id = 0
    for split in ['validation', 'test']:
        rows = []
        for p in [.45, .32, .72, .90]:
            for i in range(n_per_stratum):
                text = f'Synthetic fixture {split} {row_id}'
                rows.append({'row_id': row_id, 'text': text, 'text_sha256': text_sha256(text),
                             'p_xgb': p + i / (1000*n_per_stratum), 'source_split': split,
                             'dataset_sha256': 'a'*64, 'baseline_fingerprint': 'b'*64,
                             'y_true': i % 2})
                row_id += 1
        frames[split] = pd.DataFrame(rows)
    return frames


class Response:
    status_code = 200
    def json(self):
        return {'answers': {'department': {'score': .8}}}


class Transport:
    def __init__(self, failure=None):
        self.calls = 0
        self.failure = failure
    def post(self, *args, **kwargs):
        self.calls += 1
        assert kwargs['allow_redirects'] is False
        assert kwargs['json']['model'] == 'jev-latest'
        if self.failure:
            raise self.failure
        return Response()


class AdditionalExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'budget.sqlite3'
        self.frames = frames_fixture()
        self.manifest, self.design = prepare_additional_plan(self.frames, {})
        self.rows = self.manifest.iloc[:4].copy()
        self.transport = Transport()
        self.credentials = patch.dict('os.environ', {'METIS_API_KEY': 'offline-mock-only'})
        self.credentials.start()

    def tearDown(self):
        self.credentials.stop()
        self.temp.cleanup()

    def client(self, **kwargs):
        args = {'run_new_inference': True, 'authorized_manifest_sha256': manifest_hash(self.rows),
                'session': self.transport}
        args.update(kwargs)
        return AdditionalJevClient(self.path, self.rows, **args)

    def text(self, row):
        for frame in self.frames.values():
            hits = frame.loc[frame.text_sha256.eq(row.text_sha256), 'text']
            if len(hits):
                return hits.iloc[0]
        raise AssertionError('Missing fixture')

    def test_manifest_label_blind_unique_disjoint_and_probabilities(self):
        self.assertEqual(self.manifest.study_partition.value_counts().to_dict(), {'evaluation': 300, 'development': 200})
        altered = {name: frame.assign(y_true=1-frame.y_true) for name, frame in self.frames.items()}
        other, _ = prepare_additional_plan(altered, {})
        pd.testing.assert_frame_equal(self.manifest, other)
        self.assertNotIn('y_true', self.manifest)
        self.assertEqual(self.manifest.text_sha256.nunique(), 500)
        np.testing.assert_allclose(self.manifest.inclusion_probability,
                                   self.manifest.stratum_sample_n / self.manifest.stratum_population_n)
        previous = {'historical': self.manifest.iloc[:10]}
        new, _ = prepare_additional_plan(self.frames, previous)
        self.assertFalse(set(new.text_sha256) & set(self.manifest.iloc[:10].text_sha256))
        self.assertFalse(new.overlap_historical.any())
        with self.assertRaises(ValueError):
            prepare_additional_plan(frames_fixture(n_per_stratum=10), {})

    def test_disabled_and_missing_authorization_never_dispatch(self):
        client = self.client(run_new_inference=False)
        row = self.rows.iloc[0]
        with self.assertRaises(JevClientError):
            client.predict(row.sample_id, self.text(row))
        self.assertEqual(client.ledger_summary()['attempts_spent'], 0)
        client = self.client(authorized_manifest_sha256='wrong')
        with self.assertRaises(JevClientError):
            client.predict(row.sample_id, self.text(row))
        self.assertEqual(self.transport.calls, 0)
        result = collect_additional_phase(self.manifest, self.frames, phase='development',
                                          ledger_path=self.path, max_attempts=500)
        self.assertEqual(result['new_paid_attempts'], 0)

    def test_reserve_before_dispatch_cache_resume_and_no_credentials_needed_for_cache(self):
        row = self.rows.iloc[0]
        test = self
        class InspectTransport(Transport):
            def post(self, *args, **kwargs):
                with sqlite3.connect(test.path) as db:
                    test.assertEqual(db.execute('SELECT status FROM attempts').fetchone()[0], 'pending')
                    test.assertEqual(db.execute('SELECT COUNT(*) FROM spending').fetchone()[0], 1)
                return super().post(*args, **kwargs)
        client = self.client(session=InspectTransport())
        self.assertEqual(client.predict(row.sample_id, self.text(row)), .8)
        with patch.dict('os.environ', {'METIS_API_KEY': ''}):
            resumed = self.client(run_new_inference=False)
            self.assertEqual(resumed.predict(row.sample_id, self.text(row)), .8)
            with self.assertRaises(MissingCredential):
                self.client().predict(self.rows.iloc[1].sample_id, self.text(self.rows.iloc[1]))
        self.assertEqual(resumed.ledger_summary()['attempts_spent'], 1)
        self.assertEqual(resumed.ledger_summary()['timed_attempts'], 1)

    def test_failures_and_uncertain_interruptions_count_and_are_not_retried(self):
        row = self.rows.iloc[0]
        self.transport.failure = requests.Timeout('synthetic')
        client = self.client()
        with self.assertRaises(JevClientError):
            client.predict(row.sample_id, self.text(row))
        with self.assertRaises(AttemptAlreadyRecorded):
            self.client().predict(row.sample_id, self.text(row))
        self.assertEqual(client.ledger_summary()['failed'], 1)
        self.assertEqual(self.transport.calls, 1)
        self.transport.failure = KeyboardInterrupt()
        row = self.rows.iloc[1]
        with self.assertRaises(KeyboardInterrupt):
            client.predict(row.sample_id, self.text(row))
        self.assertEqual(client.ledger_summary()['pending_uncertain'], 1)
        with self.assertRaises(AttemptAlreadyRecorded):
            client.predict(row.sample_id, self.text(row))
        self.assertEqual(client.ledger_summary()['attempts_spent'], 2)

    def test_atomic_attempt_cap_across_workers_and_resumption(self):
        # Separate clients share the same durable transaction-protected cap.
        clients = [self.client(max_attempts=2) for _ in range(4)]
        def send(pair):
            i, client = pair
            row = self.rows.iloc[i]
            try:
                client.predict(row.sample_id, self.text(row))
                return 'ok'
            except BudgetExhausted:
                return 'capped'
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(send, enumerate(clients)))
        self.assertEqual(outcomes.count('ok'), 2)
        self.assertEqual(outcomes.count('capped'), 2)
        self.assertEqual(self.client(max_attempts=2).ledger_summary()['attempts_spent'], 2)
        with self.assertRaises(CohortMismatch):
            self.client(max_attempts=3)  # A restart cannot increase the cap.
        with self.assertRaises(ValueError):
            self.client(max_attempts=501)

    def test_atomic_money_cap_retains_failed_reservations_across_restart(self):
        args = {'monetary_cap': '0.03', 'request_cost_upper_bound': '0.02',
                'currency': 'TEST', 'pricing_evidence': 'synthetic verified test bound'}
        self.transport.failure = requests.ConnectionError('mock')
        client = self.client(**args)
        with self.assertRaises(JevClientError):
            client.predict(self.rows.iloc[0].sample_id, self.text(self.rows.iloc[0]))
        resumed = self.client(**args)
        with self.assertRaises(BudgetExhausted):
            resumed.predict(self.rows.iloc[1].sample_id, self.text(self.rows.iloc[1]))
        self.assertEqual(resumed.ledger_summary()['reserved_upper_cost'], .02)
        self.assertIsNone(resumed.ledger_summary()['measured_currency_cost'])
        self.assertEqual(self.transport.calls, 1)
        for bad in [{'monetary_cap': 1}, {'monetary_cap': float('nan'), 'request_cost_upper_bound': .1,
                                       'currency': 'TEST', 'pricing_evidence': 'mock'}]:
            with self.assertRaises(ValueError):
                self.client(**bad)

    def test_identity_changes_rejected_and_http_failure_counts(self):
        client = self.client()
        with self.assertRaises(CohortMismatch):
            client.predict(self.rows.iloc[0].sample_id, 'changed text')
        class HttpFailure(Transport):
            def post(self, *args, **kwargs):
                r = Response()
                r.status_code = 503
                self.calls += 1
                return r
        client = self.client(session=HttpFailure())
        with self.assertRaises(JevClientError):
            client.predict(self.rows.iloc[0].sample_id, self.text(self.rows.iloc[0]))
        self.assertEqual(client.ledger_summary()['failed'], 1)
        changed = self.rows.copy()
        changed.loc[changed.index[0], 'p_xgb'] += .001
        with self.assertRaises(CohortMismatch):
            AdditionalJevClient(self.path, changed)

    def test_evaluation_refuses_without_development_and_sealed_decisions(self):
        with self.assertRaises(ValueError):
            collect_additional_phase(self.manifest, self.frames, phase='evaluation',
                ledger_path=self.path, run_new_inference=True,
                authorized_manifest_sha256=manifest_hash(self.manifest), session=self.transport)
        self.assertEqual(self.transport.calls, 0)
        client = AdditionalJevClient(self.path, self.manifest, run_new_inference=True,
                    authorized_manifest_sha256=manifest_hash(self.manifest), session=self.transport)
        evaluation_row = self.manifest.loc[self.manifest.study_partition.eq('evaluation')].iloc[0]
        with self.assertRaises(CohortMismatch):
            client.predict(evaluation_row.sample_id, self.text(evaluation_row))
        self.assertEqual(self.transport.calls, 0)

    def test_end_to_end_two_phases_freeze_then_report_with_mock_transport(self):
        auth = manifest_hash(self.manifest)
        args = {'ledger_path': self.path, 'run_new_inference': True,
                'authorized_manifest_sha256': auth, 'session': self.transport}
        dev = collect_additional_phase(self.manifest, self.frames, phase='development', **args)
        self.assertEqual(dev['attempts_spent'], 200)
        client = AdditionalJevClient(self.path, self.manifest)
        decision_path = Path(self.temp.name) / 'decisions.json'
        frozen = freeze_additional_policies(self.manifest, self.frames, client, decision_path)
        ev = collect_additional_phase(self.manifest, self.frames, phase='evaluation', frozen_decisions=frozen, **args)
        self.assertEqual(ev['attempts_spent'], 500)
        self.assertEqual(self.transport.calls, 500)  # MOCK requests, zero paid calls.
        resumed = collect_additional_phase(self.manifest, self.frames, phase='evaluation', frozen_decisions=frozen, **args)
        self.assertEqual(resumed['new_paid_attempts_this_run'], 0)
        changed = json.loads(json.dumps(frozen))
        changed['thresholds']['OR'] = .2
        with self.assertRaises(ValueError):
            collect_additional_phase(self.manifest, self.frames, phase='evaluation', frozen_decisions=changed, **args)
        output = Path(self.temp.name) / 'results'
        table = analyze_additional(self.manifest, self.frames, client, frozen, output)
        self.assertEqual(set(table.stratum), {'near_threshold', 'eligible_negative', 'eligible_positive', 'outside_control'})
        self.assertTrue((output / 'additional_predictions.json').is_file())
        with self.assertRaises(ValueError):
            freeze_additional_policies(self.manifest, self.frames, client, decision_path)


if __name__ == '__main__':
    unittest.main()
