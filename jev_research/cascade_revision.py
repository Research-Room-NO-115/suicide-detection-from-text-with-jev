"""Original-validation threshold and cache-only analysis of the asymmetric cascade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import binomtest, norm
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, auc, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from .budgeted_experiments import counts_metrics, digest, metrics, paired_bootstrap, split_cache
from .client import BASE_MODEL, ENDPOINT, QUESTION
from .study2 import SEED, load_historical

REVISION = 'jev_asymmetric_cascade_revision'
BASELINE = 'Word2Vec–XGBoost'
CASCADE = 'Asymmetric cascade'
RANDOM = 'Equal-budget random referral'
CACHE_FLOOR = .25
LOWER_BOUNDS = (.15, .20, .25, .30, .35, .40)
RATING_CUTOFFS = (.40, .50, .60, .70, .80)
LOGIT_EPSILON = 1e-6
MIN_DEVELOPMENT_SUPPORT = 100


def validation_thresholds(bins):
    """Recover exact confusion counts at the archived 0.05 score boundaries.

    Each bin lies entirely on one side of the archive's 0.50 cutoff. Below it,
    errors count positive labels. From 0.50 upward, errors count negative labels.
    The top bin includes 1.00, so a cutoff of 1.00 is not identifiable here.
    """
    bins = bins.sort_values('lower').reset_index(drop=True)
    edges = np.arange(21) / 20
    if len(bins) != 20 or not np.array_equal(bins.lower, edges[:-1]) or not np.array_equal(bins.upper, edges[1:]):
        raise ValueError('Require the complete original 0.05 validation histogram.')
    if list(bins.upper_inclusive) != [False] * 19 + [True]:
        raise ValueError('Unrecognized validation histogram edge convention.')
    values = bins[['n', 'errors']].to_numpy(float)
    if (not np.isfinite(values).all() or (values < 0).any() or
            (values != np.floor(values)).any() or (bins.errors > bins.n).any()):
        raise ValueError('Validation counts must be nonnegative integers and errors cannot exceed records.')
    positive = np.where(bins.lower < .5, bins.errors, bins.n - bins.errors)
    negative = bins.n.to_numpy() - positive
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if not n_pos or not n_neg:
        raise ValueError('Youden selection requires both reference classes.')

    records = []
    for index, threshold in enumerate(edges[:-1]):
        tp, fp = int(positive[index:].sum()), int(negative[index:].sum())
        stats = counts_metrics(n_neg - fp, fp, n_pos - tp, tp)
        # Common denominator n_pos*n_neg gives exact ties despite float rounding.
        numerator = tp * n_neg - fp * n_pos
        records.append({'threshold': threshold, **stats,
                        'youden_j': numerator / (n_pos * n_neg), 'youden_numerator': numerator})
    grid = pd.DataFrame(records)
    winner = grid.sort_values(['youden_numerator', 'tp', 'threshold'], ascending=False).index[0]
    grid['selected'] = grid.index == winner
    return grid


def _logit_features(scores):
    return logit(np.clip(np.asarray(scores, dtype=float), LOGIT_EPSILON, 1 - LOGIT_EPSILON)).reshape(-1, 1)


def _available_ratings(frame):
    ok = frame.jev_status.eq('succeeded').to_numpy()
    values = frame.jev_score.to_numpy(float)
    if (not frame.jev_status.isin(['succeeded', 'failed', 'pending', 'not_attempted']).all() or
            not np.isfinite(values[ok]).all() or ((values[ok] < 0) | (values[ok] > 1)).any() or
            not np.isnan(values[~ok]).all()):
        raise ValueError('Unexpected response status or rating.')
    return ok, values


def _xgb_scores(frame):
    scores = frame.p_xgb.to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('XGBoost scores must lie in [0,1].')
    return scores


def fit_cascade(development, xgb_threshold):
    """Select the lower edge and calibrated cutoff on cached development rows.

    Cross-fitted scores keep each candidate's calibration prediction out of its
    own fit. The final calibration is then fitted on all successful development
    negatives within cache coverage and used without evaluation-time tuning.
    """
    xgb = _xgb_scores(development)
    y = development.y_true.to_numpy(int)
    available, ratings = _available_ratings(development)
    base = xgb >= xgb_threshold
    fit_mask = available & ~base & (xgb >= CACHE_FLOOR)
    features, truth = _logit_features(ratings[fit_mask]), y[fit_mask]
    if (np.bincount(truth, minlength=2) < 5).any():
        raise ValueError('Five calibration folds need at least five records of each class.')
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit = lambda x, t: LogisticRegression(C=1.0, solver='lbfgs').fit(x, t)
    oof = np.full(len(truth), np.nan)
    for train, held_out in folds.split(features, truth):
        oof[held_out] = fit(features[train], truth[train]).predict_proba(features[held_out])[:, 1]
    probability = np.full(len(y), np.nan)
    probability[fit_mask] = oof
    base_f2 = metrics(y, base)['f2']
    rows = []
    for lower in LOWER_BOUNDS:
        if lower < CACHE_FLOOR:
            rows.append({'lower': lower, 'upper_exclusive': xgb_threshold,
                         'status': 'JEV coverage below 0.25 is absent'})
            continue
        route = ~base & (xgb >= lower)
        requests = int(route.sum())
        if requests < MIN_DEVELOPMENT_SUPPORT:
            rows.append({'lower': lower, 'upper_exclusive': xgb_threshold, 'requests': requests,
                         'status': 'below minimum development support'})
            continue
        for cutoff in RATING_CUTOFFS:
            additions = route & (probability >= cutoff)
            y_hat = base | additions
            delta = metrics(y, y_hat)['f2'] - base_f2
            rows.append({'lower': lower, 'upper_exclusive': xgb_threshold, 'requests': requests,
                         'calibrated_cutoff': cutoff, 'positive_recoveries': int((additions & (y == 1)).sum()),
                         'added_false_alarms': int((additions & (y == 0)).sum()), 'delta_f2': delta,
                         'delta_f2_per_100_requests': 100 * delta / requests, 'status': 'scored on development'})
    sweep = pd.DataFrame(rows)
    supported = sweep.loc[sweep.status.eq('scored on development') & sweep.delta_f2.gt(0)]
    if supported.empty:
        raise ValueError('No supported candidate improved development F2. Retain the comparator.')
    best = supported.sort_values(['delta_f2_per_100_requests', 'positive_recoveries', 'requests', 'calibrated_cutoff'],
                                 ascending=[False, False, True, False], kind='stable').iloc[0]
    sweep['selected'] = sweep.index == best.name
    model = fit(features, truth)
    return {
        'architecture': 'asymmetric_or', 'xgb_threshold': float(xgb_threshold),
        'routing_interval': [float(best.lower), float(xgb_threshold)],
        'calibrated_jev_cutoff': float(best.calibrated_cutoff),
        'calibration_slope': float(model.coef_[0, 0]), 'calibration_intercept': float(model.intercept_[0]),
        'calibration_fit_rows': int(fit_mask.sum()), 'calibration_folds': 5, 'calibration_logit_clip': LOGIT_EPSILON,
        'selection_metric': 'positive development F2 gain per prospective request; requires >=100 referrals',
        'fallback': 'retain tuned Word2Vec–XGBoost decision', 'model': BASE_MODEL,
        'endpoint': ENDPOINT, 'question': QUESTION, 'question_sha256': digest(QUESTION),
    }, sweep


def cascade_arrays(frame, policy, random_probability):
    """Replay the same add-only rule under score-based and random referral."""
    if not 0 <= random_probability <= 1:
        raise ValueError('Invalid random-referral probability.')
    lower, upper = policy['routing_interval']
    if upper != policy['xgb_threshold'] or not 0 <= lower < upper <= 1:
        raise ValueError('A cascade interval must end at the fixed comparator threshold.')
    xgb = _xgb_scores(frame)
    available, raw = _available_ratings(frame)
    base = xgb >= upper
    route = ~base & (xgb >= lower)
    # Apply the frozen binary rule even in exterior random controls. Its score
    # calibration was estimated inside coverage and may not transport there.
    q = expit((policy['calibration_slope'] * _logit_features(raw)[:, 0]) + policy['calibration_intercept'])
    can_add = ~base & available & (q >= policy['calibrated_jev_cutoff'])
    assisted = base | (route & can_add)
    random_positive = base.astype(float) + random_probability * can_add
    return {
        BASELINE: (base.astype(float), np.zeros(len(frame))),
        CASCADE: (assisted.astype(float), route.astype(float)),
        RANDOM: (random_positive, random_probability * (~base)),
    }, q


def policy_metrics(y, q, need, weights=None):
    """Binary decisions or marginal expected binary decisions on the same rows."""
    y = np.asarray(y, dtype=float)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    tn = np.dot(weights, (1 - y) * (1 - q))
    fp = np.dot(weights, (1 - y) * q)
    fn = np.dot(weights, y * (1 - q))
    tp = np.dot(weights, y * q)
    m = counts_metrics(tn, fp, fn, tp)
    # counts_metrics has an integer confusion output for ordinary observations.
    # The metrics above use unrounded totals, and we report those totals here.
    m.update({'tn': float(tn), 'fp': float(fp), 'fn': float(fn), 'tp': float(tp),
              'requests': float(np.dot(weights, need)), 'n': float(weights.sum())})
    m['requests_per_1000'] = 1000 * m['requests'] / m['n']
    m['flags_per_1000'] = 1000 * (tp + fp) / m['n']
    return m


def paired_mcnemar(y, base, other, *, comparison):
    y, base, other = np.asarray(y, int), np.asarray(base), np.asarray(other)
    if not np.isin(base, [0, 1]).all() or not np.isin(other, [0, 1]).all():
        raise ValueError('Exact McNemar requires observed, binary paired decisions.')
    rows = []
    for label, mask in [('all: correct classifications', y >= 0), ('positive: sensitivity', y == 1),
                        ('negative: specificity', y == 0)]:
        before, after = base[mask] == y[mask], other[mask] == y[mask]
        gains, losses = int((~before & after).sum()), int((before & ~after).sum())
        rows.append({'comparison': comparison, 'endpoint': label, 'n': int(mask.sum()),
                     'corrected': gains, 'new_errors': losses,
                     'mcnemar_exact_p': float(binomtest(gains, gains + losses, p=.5).pvalue) if gains + losses else 1.})
    return rows


def paired_delong(y, baseline, alternative):
    """Compare continuous ROC curves with paired DeLong influence values.

    For each class and model, get midrank placement via binary searches over
    the other class. The class-specific covariances are calculated on the same
    records for both models, with half credit for score ties.
    """
    y = np.asarray(y)
    scores = np.asarray([baseline, alternative], dtype=float)
    if scores.shape != (2, len(y)) or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Paired DeLong requires complete matched scores and binary labels.')
    positive, negative = y == 1, y == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if min(n_pos, n_neg) < 2:
        raise ValueError('Paired DeLong requires at least two records per class.')

    def placements(values, reference):
        reference = np.sort(reference)
        return (np.searchsorted(reference, values, side='left') +
                np.searchsorted(reference, values, side='right')) / (2 * len(reference))

    pos = np.array([placements(s[positive], s[negative]) for s in scores])
    neg = np.array([1 - placements(s[negative], s[positive]) for s in scores])
    aucs = pos.mean(axis=1)
    covariance = np.cov(pos, ddof=1) / n_pos + np.cov(neg, ddof=1) / n_neg
    contrast = np.array([-1., 1.])
    se = float(np.sqrt(max(0., contrast @ covariance @ contrast)))
    delta = float(contrast @ aucs)
    critical = float(norm.ppf(.975))
    return {'n': len(y), 'positives': n_pos, 'negatives': n_neg, 'xgb_roc_auc': float(aucs[0]),
            'comparison_roc_auc': float(aucs[1]), 'delta_auc': delta, 'standard_error': se,
            'ci_low': delta - critical * se, 'ci_high': delta + critical * se,
            'p_value': float(2 * norm.sf(abs(delta) / se)) if se > 0 else None,
            'scope': 'exploratory; row independence assumed, author clusters unavailable'}


def ranking_audit(frame):
    """Continuous historical scores, with no ROC claim for binary cascades."""
    ratings = frame.jev_score.to_numpy(float)
    xgb = _xgb_scores(frame)
    score = np.where(frame.jev_status.eq('succeeded'), ratings, xgb)
    y = frame.y_true.to_numpy(int)
    rows, curves = [], []
    for name, values in [(BASELINE, xgb), ('Historical Metis score replacement', score)]:
        precision, recall, _ = precision_recall_curve(y, values)
        rows.append({'model': name, 'n': len(y), 'roc_auc': float(roc_auc_score(y, values)),
                     'average_precision': float(average_precision_score(y, values)),
                     'pr_auc_trapezoidal': float(auc(recall, precision))})
        fpr, tpr, _ = roc_curve(y, values)
        curves.append((name, recall, precision, fpr, tpr))
    return pd.DataFrame(rows), paired_delong(y, xgb, score), curves


def historical_cascade_replay(frame, policy):
    y = frame.y_true.to_numpy(int)
    lower, threshold = policy['routing_interval']
    negative = frame.p_xgb.lt(threshold)
    referrals = int((negative & frame.p_xgb.ge(lower)).sum())
    r = referrals / int(negative.sum())
    outputs, calibrated = cascade_arrays(frame, policy, r)
    models = pd.DataFrame([{'policy': name, **policy_metrics(y, pred, needed),
                            'counts': 'expectation' if name == RANDOM else 'observed replay'}
                           for name, (pred, needed) in outputs.items()])
    base, other = outputs[BASELINE][0], outputs[CASCADE][0]
    paired = paired_mcnemar(y, base, other, comparison=f'{CASCADE} vs {BASELINE}')
    replacement = np.where(frame.jev_status.eq('succeeded'), frame.jev_score >= .5, frame.p_xgb >= .5)
    paired += paired_mcnemar(y, frame.p_xgb >= .5, replacement, comparison='Historical score replacement at 0.50')
    return models, pd.DataFrame(paired), paired_bootstrap(y, base, other), calibrated


def historical_random_contrast(frame, policy, *, replicates=4000):
    """Pair cache rows, holding the random-referral policy probability fixed."""
    y = frame.y_true.to_numpy(int)
    lower, upper = policy['routing_interval']
    r = frame.p_xgb.between(lower, upper, inclusive='left').sum() / frame.p_xgb.lt(upper).sum()
    outputs, _ = cascade_arrays(frame, policy, r)
    primary, random = outputs[CASCADE][0], outputs[RANDOM][0]
    states, counts = np.unique(np.column_stack([y, primary - random]), axis=0, return_counts=True)
    draws = np.random.default_rng(SEED).multinomial(len(y), counts / len(y), size=replicates)
    pos, neg = states[:, 0], 1 - states[:, 0]
    delta = states[:, 1]
    sensitivity_differences = (draws @ (pos * delta)) / (draws @ pos)
    false_alarm_differences = 1000 * (draws @ (neg * delta)) / len(y)
    rows = []
    for metric, values, estimate in [
        ('recall_difference', sensitivity_differences, np.sum(y * (primary - random)) / y.sum()),
        ('additional_false_alarms_per_1000', false_alarm_differences,
         1000 * np.mean((1 - y) * (primary - random))),
    ]:
        low, high = np.quantile(values, [.025, .975])
        rows.append({'metric': metric, 'estimate': estimate, 'ci_low': low, 'ci_high': high,
                     'scope': '3,000 reused cache rows; random pool restricted to observed comparator negatives'})
    return pd.DataFrame(rows)


def _read_json(path):
    return json.loads(Path(path).read_text())


def analyze_archive(root):
    root = Path(root)
    results = root / 'research/results'
    previous = results / 'jev_probability_hybrid'
    study = _read_json(previous / 'study.json')
    archive = _read_json(previous / 'hybrid_report.json')
    baseline = _read_json(results / 'jev_error_study/xgboost_baseline.json')
    bins = pd.read_json(previous / 'validation_bins.json', precise_float=True)
    grid = validation_thresholds(bins)
    at_half = grid.loc[grid.threshold.eq(.5)].iloc[0]
    errors = baseline['error_counts']['validation']
    if (at_half.n != study['validation_size'] or at_half.fn != errors['false_negative'] or
            at_half.fp != errors['false_positive'] or study['baseline_fingerprint'] != baseline['baseline_fingerprint']):
        raise ValueError('Validation histogram and baseline archive disagree.')
    frame, _ = load_historical(root)
    _, development, evaluation = split_cache(frame)
    if frame.baseline_fingerprint.unique().tolist() != [study['baseline_fingerprint']]:
        raise ValueError('Cached ratings belong to a different baseline.')
    choice = grid.loc[grid.selected].iloc[0]
    policy, sweep = fit_cascade(development, choice.threshold)
    table, tests, intervals, calibrated = historical_cascade_replay(evaluation, policy)
    rank, delong, pr_curves = ranking_audit(evaluation)
    score_mask = evaluation.p_xgb.lt(choice.threshold) & evaluation.jev_status.eq('succeeded')
    calibration_rows, calibration_curves = [], []
    for label, scores in [('Raw Metis rating', evaluation.jev_score.to_numpy()[score_mask]),
                          ('Sigmoid calibration', calibrated[score_mask])]:
        truth = evaluation.y_true.to_numpy()[score_mask]
        observed, mean_score = calibration_curve(truth, scores, n_bins=8, strategy='quantile')
        calibration_rows.append({'score': label, 'n': len(truth), 'brier': brier_score_loss(truth, scores),
                                 'log_loss': log_loss(truth, scores), 'ap': average_precision_score(truth, scores)})
        calibration_curves.append((label, mean_score, observed))
    development_routing = []
    for row in bins.itertuples(index=False):
        old = development.loc[development.p_xgb.ge(row.lower) & development.p_xgb.lt(row.upper)]
        development_routing.append({'lower': row.lower, 'upper': row.upper, 'validation_n': int(row.n),
                                    'validation_xgb_050_error_rate': row.error_rate,
                                    'validation_positive_rate': (row.errors if row.lower < .5 else row.n - row.errors) / row.n,
                                    'cached_development_n': len(old),
                                    'cached_jev_successes': int(old.jev_status.eq('succeeded').sum())})
    routed = frame.p_xgb.lt(.5)
    recovered = routed & frame.y_true.eq(1) & frame.jev_score.ge(.5) & frame.jev_status.eq('succeeded')
    false_alarms = routed & frame.y_true.eq(0) & frame.jev_score.ge(.5) & frame.jev_status.eq('succeeded')
    _, selected_fp, selected_fn, _ = (archive['selected_subset']['xgboost']['confusion'][key] for key in ['tn', 'fp', 'fn', 'tp'])
    full = archive['full_test_partial_policy']['xgboost']
    full_positive = full['confusion']['tp'] + full['confusion']['fn']
    historic = {'n': len(frame), 'referrals': int(routed.sum()), 'recovered_positives': int(recovered.sum()),
                'added_false_alarms': int(false_alarms.sum()), 'new_missed_positives': 0,
                'selected_misses': selected_fn - int(recovered.sum()),
                'selected_false_alarms': selected_fp + int(false_alarms.sum()),
                'full_frame_n': full['n'], 'full_frame_baseline_recall': full['recall_positive'],
                'full_frame_replayed_recall': (full['confusion']['tp'] + int(recovered.sum())) / full_positive}
    return {'policy': policy, 'threshold_grid': grid, 'routing_sweep': sweep, 'archive_replay': table,
            'mcnemar': tests, 'paired_intervals': intervals, 'ranking': rank, 'delong': delong,
            'pr_curves': pr_curves, 'calibration_curves': calibration_curves,
            'calibration_audit': pd.DataFrame(calibration_rows),
            'random_contrast': historical_random_contrast(evaluation, policy),
            'bin_audit': pd.DataFrame(development_routing), 'historical_or': historic,
            'source_study': study, 'baseline_archive': baseline}
