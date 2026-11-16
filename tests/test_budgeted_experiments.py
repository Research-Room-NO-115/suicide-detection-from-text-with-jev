"""Checks of evidence integrity, leakage controls, fallback and paired metrics."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, confusion_matrix

from jev_research.budgeted_experiments import (
    validate_cache, split_cache, choose_threshold, fit_policies, policy_output,
    metrics, transitions, paired_bootstrap, full_population_from_aggregates,
    reproduce, holm,
)

ROOT = Path(__file__).resolve().parents[1]


class CachedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_json(ROOT / 'research/results/jev_probability_hybrid/predictions.json', precise_float=True)
        cls.report = json.loads((ROOT / 'research/results/jev_probability_hybrid/hybrid_report.json').read_text())

    def test_reproduces_real_evidence_and_or_request_short_circuit(self):
        validate_cache(self.frame)
        table = reproduce(self.frame, self.report).set_index('policy')
        self.assertEqual((table.loc['OR', 'fn'], table.loc['OR', 'fp']), (322, 1049))
        self.assertEqual(table.loc['OR', 'prospective_requests'], 2184)
        self.assertEqual(table.loc['AND', 'prospective_requests'], 2816)
        self.assertEqual(table.loc['JEV replacement', 'net_correct'], 732)
        self.assertEqual(table.loc['JEV replacement', 'net_fn_reduction'], 27)
        self.assertAlmostEqual(table.loc['JEV replacement', 'accuracy_change_pp'], 14.64)
        self.assertNotAlmostEqual(table.loc['JEV replacement', 'accuracy_relative_improvement_pct'], 14.64)

    def test_missing_response_retains_xgb_at_half_including_tuned_threshold(self):
        frame = pd.DataFrame({'p_xgb': [.3, .55, .7, .45],
                              'jev_score': [None, None, .01, .99],
                              'jev_status': ['failed', 'pending', 'succeeded', 'succeeded']})
        for spec in [{'kind': 'replacement'}, {'kind': 'weighted', 'jev_weight': .5}]:
            score, prediction, required = policy_output(frame, spec, .1)
            self.assertEqual(prediction[:2].tolist(), [False, True])
            np.testing.assert_equal(score[:2], [.3, .55])
            self.assertTrue(required.all())
        # OR can decide a positive without requiring JEV, even if the historical
        # request failed; AND can decide a negative without a provider result.
        self.assertFalse(policy_output(frame, {'kind': 'or'}, .1)[2].any())
        _, p, need = policy_output(frame, {'kind': 'and'}, .5)
        self.assertEqual(need.tolist(), [False, True, True, False])
        self.assertEqual(p[:2].tolist(), [False, True])
        _, p, _ = policy_output(frame, {'kind': 'replacement', 'interval': [.4, .6]}, .1)
        self.assertFalse(p[0])
        self.assertTrue(p[2])  # No extrapolation: .7 is unrouted and keeps XGB.

    def test_split_is_stable_label_blind_disjoint_and_unique(self):
        whole, development, evaluation = split_cache(self.frame)
        altered = self.frame.sample(frac=1, random_state=2).copy()
        altered['y_true'] = 1 - altered.y_true
        changed, _, _ = split_cache(altered)
        pd.testing.assert_frame_equal(whole[['sample_id', 'partition']], changed[['sample_id', 'partition']])
        self.assertEqual((len(development), len(evaluation)), (2000, 3000))
        for col in ['sample_id', 'row_id', 'text_sha256']:
            self.assertFalse(set(development[col]) & set(evaluation[col]))

    def test_corrupt_cache_is_rejected(self):
        bad = self.frame.copy()
        bad.loc[0, 'row_id'] = -1
        with self.assertRaises(ValueError):
            validate_cache(bad)
        bad = self.frame.copy()
        bad.loc[0, 'jev_status'] = 'failed'
        with self.assertRaises(ValueError):
            validate_cache(bad)
        with self.assertRaises(ValueError):
            validate_cache(pd.concat([self.frame, self.frame.iloc[:1]]))

    def test_threshold_target_is_selected_from_development_only(self):
        _, dev, evaluation = split_cache(self.frame)
        spec = {'kind': 'xgb'}
        choice = choose_threshold(dev, spec, .95)
        _, prediction, _ = policy_output(dev, spec, choice['threshold'])
        self.assertGreaterEqual(metrics(dev.y_true, prediction)['recall'], .95)
        # No larger cutoff can remove a false alarm while meeting the target.
        for threshold in np.unique(dev.p_xgb[dev.p_xgb > choice['threshold']]):
            _, pred, _ = policy_output(dev, spec, threshold)
            result = metrics(dev.y_true, pred)
            self.assertTrue(result['recall'] < .95 or result['fp'] >= choice['development_fp'])
        original = choice.copy()
        evaluation['y_true'] = 1 - evaluation.y_true
        self.assertEqual(original, choose_threshold(dev, spec, .95))

    def test_fusion_is_fitted_to_development_and_serializes_exactly(self):
        _, dev, evaluation = split_cache(self.frame)
        specs, thresholds, _ = fit_policies(dev.iloc[:200])
        before = policy_output(evaluation, specs['Logistic fusion'])[0]
        restored = json.loads(json.dumps(specs))
        np.testing.assert_equal(before, policy_output(evaluation, restored['Logistic fusion'])[0])
        flipped = dev.iloc[:200].copy()
        flipped['y_true'] = 1 - flipped.y_true
        different = fit_policies(flipped)[0]['Logistic fusion']
        self.assertNotEqual(specs['Logistic fusion']['coefficients'], different['coefficients'])

    def test_metrics_and_transitions_consistent_with_sklearn(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1])
        a = np.array([0, 1, 0, 1, 1, 0, 0])
        b = np.array([0, 0, 1, 0, 1, 1, 1])
        m = metrics(y, b)
        self.assertEqual([m[k] for k in ['tn', 'fp', 'fn', 'tp']], confusion_matrix(y, b).ravel().tolist())
        self.assertAlmostEqual(m['f2'], fbeta_score(y, b, beta=2))
        tr = transitions(y, a, b)
        self.assertEqual(tr['net_correct'], int(np.sum(y == b) - np.sum(y == a)))
        ci = paired_bootstrap(y, a, b, replicates=2000)
        self.assertAlmostEqual(ci['recall']['difference'], metrics(y, b)['recall'] - metrics(y, a)['recall'])
        zero = paired_bootstrap(y, a, a, replicates=100)
        self.assertTrue(all(v['ci95'] == [0., 0.] for v in zero.values()))

    def test_full_counts_are_derived_but_no_full_ranking_is_invented(self):
        table = full_population_from_aggregates(self.frame, self.report).set_index('policy')
        self.assertEqual(table.loc['JEV replacement', 'fn'], 1808)
        self.assertEqual(table.loc['JEV replacement', 'fp'], 1182)
        self.assertEqual(table.loc['OR', 'fn'], 1297)
        self.assertEqual(table.loc['OR', 'fp'], 1983)
        self.assertTrue(table.roc_auc.isna().all())
        self.assertTrue(table.average_precision.isna().all())
        self.assertAlmostEqual(table.loc['JEV replacement', 'log_loss'], self.report['full_test_partial_policy']['hybrid']['log_loss'])

    def test_holm_preserves_order_and_controls_family(self):
        np.testing.assert_allclose(holm([.04, .001, .02]), [.04, .003, .04])


if __name__ == '__main__':
    unittest.main()
