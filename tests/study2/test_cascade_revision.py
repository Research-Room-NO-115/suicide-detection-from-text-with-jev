from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jev_research.additional_experiment import manifest_hash
from jev_research.budgeted_experiments import digest, file_digest, save_json
from jev_research.cascade_revision import BASELINE, CASCADE, RANDOM, REVISION, analyze_archive, cascade_arrays, paired_delong
from jev_research.cascade_validation import IDENTITY, _paths, replay, sample_residual, weighted_analysis
from jev_research.client import BASE_MODEL, QUESTION, CohortMismatch, text_sha256
from jev_research.study2 import STUDY
from jev_research.study2_collection import Study2JevClient, collect, write_frozen_package


ROOT = Path(__file__).resolve().parents[2]


def test_frame(outside=20):
    rows = []
    scores = [.15] * outside + [.42] * 8 + [.44] * 8 + [.45] * 10 + [.85] * 10
    for i, score in enumerate(scores):
        text = f'Fixture record {i}'
        rows.append({'row_id': i, 'text': text, 'text_sha256': text_sha256(text), 'p_xgb': score,
                     'source_split': 'test', 'dataset_sha256': 'a' * 64, 'baseline_fingerprint': 'b' * 64,
                     'y_true': i % 2})
    return pd.DataFrame(rows)


def policy():
    return {'architecture': 'asymmetric_or', 'routing_interval': [.4, .45], 'xgb_threshold': .45,
            'calibrated_jev_cutoff': .5, 'calibration_slope': 1., 'calibration_intercept': 0.,
            'model': BASE_MODEL, 'question_sha256': digest(QUESTION)}


def freeze(manifest, design, reference_hash):
    return {'study_id': STUDY, 'revision': REVISION, 'status': 'FROZEN_INTERNAL_PROTOCOL',
            'frozen_at_utc': '2020-01-01T00:00:00+00:00', 'candidate_policy': policy(),
            'reference_file_sha256': reference_hash, 'evaluation_manifest_sha256': manifest_hash(manifest),
            'design_sha256': digest(design), 'gates': dict.fromkeys([
                'primary_source_and_construct_verified', 'dataset_label_provenance_verified',
                'external_transfer_terms_permit', 'complete_sampling_frame_verified', 'frozen_manifest_and_protocol'], True)}


class CascadeRevisionTests(unittest.TestCase):
    def test_archived_threshold_and_historical_arithmetic(self):
        report = analyze_archive(ROOT)
        half = report['threshold_grid'].loc[lambda x: x.threshold.eq(.50)].iloc[0]
        selected = report['threshold_grid'].loc[lambda x: x.selected].iloc[0]
        self.assertEqual((half.n, half.fn, half.fp), (37086, 1385, 1461))
        self.assertEqual((selected.threshold, selected.fn, selected.fp), (.45, 1185, 1650))
        self.assertEqual(report['policy']['routing_interval'], [.40, .45])
        self.assertEqual(report['historical_or']['referrals'], 2184)
        self.assertEqual(report['historical_or']['recovered_positives'], 538)
        table = report['archive_replay'].set_index('policy')
        self.assertEqual((table.loc[CASCADE, 'fn'], table.loc[CASCADE, 'fp'], table.loc[CASCADE, 'requests']), (316, 717, 239))
        self.assertAlmostEqual(table.loc[RANDOM, 'requests'], 239)

    def test_positive_preservation_and_failed_response_fallback(self):
        frame = pd.DataFrame({'p_xgb': [.15, .399, .4, .42, .44, .45, .9],
                              'jev_score': [.9, .99, .9, np.nan, .1, .01, .01],
                              'jev_status': ['succeeded', 'succeeded', 'succeeded', 'failed',
                                             'succeeded', 'succeeded', 'succeeded']})
        arrays, _ = cascade_arrays(frame, policy(), .5)
        np.testing.assert_array_equal(arrays[CASCADE][0], [0, 0, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(arrays[CASCADE][1], [0, 0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(arrays[RANDOM][0], [.5, .5, .5, 0, 0, 1, 1])

    def test_sample_is_label_blind_and_random_budget_matches_population(self):
        frame = test_frame(outside=2000)
        prior_failed = frame.iloc[[2, 2001]].copy()
        exclusions = {'prior_failures': prior_failed}
        sampled, design = sample_residual(frame, exclusions, policy())
        permuted = frame.assign(y_true=1-frame.y_true).sample(frac=1, random_state=1)
        another, _ = sample_residual(permuted, exclusions, policy())
        pd.testing.assert_frame_equal(sampled, another)
        self.assertFalse(sampled.row_id.isin(prior_failed.row_id).any())
        self.assertFalse(sampled.loc[sampled.p_xgb.ge(.45), 'jev_collection'].any())
        self.assertNotIn('y_true', sampled)
        self.assertLess(design['strata'][1]['inclusion_probability'], 1)
        sampled['y_true'] = sampled.row_id % 2
        sampled['jev_score'] = np.where(sampled.jev_collection, .9, np.nan)
        sampled['jev_status'] = np.where(sampled.jev_collection, 'succeeded', 'not_attempted')
        main, _, _ = weighted_analysis(sampled, policy(), design, replicates=40)
        rows = main.set_index('policy')
        self.assertAlmostEqual(rows.loc[CASCADE, 'requests'], design['population_route_n'])
        self.assertAlmostEqual(rows.loc[RANDOM, 'requests'], design['population_route_n'])
        self.assertGreaterEqual(rows.loc[CASCADE, 'recall'], rows.loc[BASELINE, 'recall'])

    def test_delong_agrees_with_auc_and_brute_force_midrank_variance(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        x = np.array([.2, .5, .7, .8, .5, .5])
        z = np.array([.1, .4, .6, .8, .6, .4])
        result = paired_delong(y, x, z)
        self.assertAlmostEqual(result['xgb_roc_auc'], roc_auc_score(y, x))
        self.assertAlmostEqual(result['comparison_roc_auc'], roc_auc_score(y, z))
        pair_matrix = lambda s: (s[y == 1, None] > s[y == 0]).astype(float) + .5 * (s[y == 1, None] == s[y == 0])
        delta = pair_matrix(z) - pair_matrix(x)
        variance = delta.mean(axis=0).var(ddof=1) / 3 + delta.mean(axis=1).var(ddof=1) / 3
        self.assertAlmostEqual(result['standard_error'] ** 2, variance)

    def test_collection_and_readonly_replay_share_one_ledger_and_skip_positives(self):
        frame = test_frame()
        sample, design = sample_residual(frame, {}, policy())
        design['sample_manifest_sha256'] = manifest_hash(sample)
        manifest = sample.loc[sample.jev_collection].copy()
        with tempfile.TemporaryDirectory() as directory:
            work, ledger = _paths(directory)
            work.mkdir(parents=True)
            reference = work / 'reference_private.csv.gz'
            frame.loc[frame.row_id.isin(sample.row_id), ['row_id', 'text_sha256', 'y_true']].to_csv(reference, index=False)
            protocol = freeze(manifest, design, file_digest(reference))
            write_frozen_package(work, protocol, manifest, design)
            save_json(work / 'analysis_sample.json', sample.to_dict('records'))
            source = frame[IDENTITY + ['text']]
            bad = freeze(sample, design, file_digest(reference))
            with self.assertRaisesRegex(CohortMismatch, 'comparator-negative'):
                Study2JevClient(ledger, sample, protocol=bad)
            session = Mock()
            session.post.return_value = Mock(status_code=200)
            session.post.return_value.json.return_value = {'answers': {'department': {'score': .8}}}
            with patch.dict('os.environ', {'METIS_API_KEY': 'offline-fixture'}):
                collect(manifest, source, protocol, design, ledger_path=ledger, run_new_inference=True,
                        workers=1, session_factory=lambda: session)
                collect(manifest, source, protocol, design, ledger_path=ledger, run_new_inference=True,
                        workers=1, session_factory=lambda: session)
            self.assertEqual(session.post.call_count, len(manifest))
            checksum = file_digest(ledger)
            status, outputs = replay(directory)
            self.assertEqual(file_digest(ledger), checksum)
            self.assertEqual(status['actual_new_attempts'], len(manifest))
            self.assertEqual(status['state'], 'replayed')
            self.assertEqual(len(outputs['fresh_population']), 3)
            reference.write_bytes(reference.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'reference labels'):
                replay(directory)


if __name__ == '__main__':
    unittest.main()
