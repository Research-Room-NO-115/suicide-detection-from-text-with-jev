"""Render the revised analysis, paper tables and figures from archived evidence."""

from __future__ import annotations

from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .budgeted_experiments import digest, file_digest, save_json, save_table
from .cascade_revision import BASELINE, CASCADE, RANDOM, REVISION, analyze_archive
from .cascade_validation import STRATA, replay
from .client import BASE_MODEL, ENDPOINT, QUESTION


def evidence_audit(root, analysis):
    results = root / 'research/results'
    legacy = results / 'jev_probability_hybrid'
    baseline = analysis['baseline_archive']
    phases = json.loads((legacy / 'execution_audit.json').read_text())['phases']
    start = min(datetime.fromisoformat(p['first_started_utc']) for p in phases)
    end = max(datetime.fromisoformat(p['last_finished_utc']) for p in phases)
    tz = ZoneInfo('Asia/Tehran')
    attempts = sum(p['attempts'] for p in phases)
    active = sum((datetime.fromisoformat(p['last_finished_utc']) -
                  datetime.fromisoformat(p['first_started_utc'])).total_seconds() for p in phases)
    model = {'name': 'Metis Jev API', 'provider_domain': 'api.metisai.ir', 'provider_request_alias': BASE_MODEL,
             'endpoint': ENDPOINT, 'prompt': QUESTION, 'prompt_sha256': digest(QUESTION),
             'response_field': 'answers.department.score', 'response_rule': 'finite numeric value in [0,1]',
             'connect_timeout_seconds': 10, 'read_timeout_seconds': 60, 'automatic_retries': 0,
             'immutable_backend_version': None, 'architecture': None, 'parameter_count': None,
             'generation_config': 'Provider defaults; temperature, seed and token logits absent from the exported response contract',
             'historical_started_tehran': start.astimezone(tz).isoformat(),
             'historical_finished_tehran': end.astimezone(tz).isoformat(),
             'historical_attempts': attempts, 'historical_failed_attempts': 4,
             'historical_workers': sorted({phase['workers'] for phase in phases}),
             'retired_name': 'Perplexity Decider has no documented model ID or executed results in the repository.',
             'reproducibility': 'Request parameters are available. An archived response reproduces the analysis under the mutable provider alias.'}
    selected = analysis['archive_replay'].set_index('policy').loc[CASCADE]
    seconds = baseline['coverage']['test']['prediction_seconds']
    resources = {'archive_test_n': baseline['split_counts']['test'], 'archived_test_prediction_seconds': seconds,
                 'archived_prepared_prediction_seconds_per_1000': seconds * 1000 / baseline['split_counts']['test'],
                 'archived_prepared_timing_scope': 'Word2Vec embedding and XGBoost score prediction for pre-tokenized batch',
                 'archived_fit_seconds': baseline['fit']['fit_seconds'], 'archived_xgb_cpu_threads': baseline['identity']['xgb_threads'],
                 'historical_jev_active_phase_wall_seconds': active, 'historical_jev_calendar_wall_seconds': (end - start).total_seconds(),
                 'historical_jev_active_batch_seconds_per_1000_attempts': active * 1000 / attempts,
                 'revised_cached_cascade_requests_per_1000': selected.requests_per_1000,
                 'hypothetical_all_referral_requests_per_1000': 1000,
                 'revised_cached_api_attempt_reduction_pct': 100 * (1 - selected.requests_per_1000 / 1000),
                 'measured_local_end_to_end_inference': None, 'measured_local_peak_rss_mib': None,
                 'observed_provider_input_tokens': None, 'observed_provider_output_tokens': None,
                 'verified_currency_per_1000_estimate': None}
    previous = pd.read_json(results / 'jev_error_study/selection_manifest.json', precise_float=True)
    archive = pd.read_json(legacy / 'predictions.json', precise_float=True)
    others = previous.loc[previous.source_split.eq('test') & ~previous.text_sha256.isin(archive.text_sha256)]
    original_study = analysis['source_study']
    residual = {'original_test_n': baseline['split_counts']['test'], 'old_interval_n': original_study['cohort_selection']['eligible_test_count']}
    residual['unqueried_in_old_interval_before_other_history'] = residual['old_interval_n'] - len(archive)
    residual['additional_prior_test_n'] = len(others)
    residual['additional_prior_test_inside_old_interval_n'] = int(others.p_xgb.between(.25, .8, inclusive='left').sum())
    residual['potential_test_n'] = residual['original_test_n'] - len(archive) - len(others)
    residual['potential_old_interval_n'] = residual['unqueried_in_old_interval_before_other_history'] - residual['additional_prior_test_inside_old_interval_n']
    residual['revised_route_n'] = None
    return model, resources, residual


def _figure_set(analysis, output):
    import matplotlib.pyplot as plt

    figures = output / 'figures'
    figures.mkdir(exist_ok=True)

    def save(fig, filename):
        fig.savefig(figures / f'{filename}.png', dpi=190, bbox_inches='tight')
        fig.savefig(figures / f'{filename}.pdf', bbox_inches='tight')
        plt.close(fig)

    with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), layout='constrained')
        for (label, recall, precision, fpr, tpr), color in zip(analysis['pr_curves'], ['#475569', '#007872']):
            axes[0].plot(fpr, tpr, label=label, color=color, linewidth=1.7)
            axes[1].plot(recall, precision, label=label, color=color, linewidth=1.7)
        axes[0].plot([0, 1], [0, 1], ':', color='#94a3b8')
        baseline = analysis['archive_replay'].iloc[0]
        axes[1].axhline(baseline.positive_n / baseline.n, linestyle=':', color='#94a3b8')
        axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
        axes[0].legend(loc='lower right', fontsize=7)
        fig.suptitle('Continuous-score audit on 3,000 previously inspected records')
        save(fig, 'continuous_scores_roc_pr')

        table = analysis['archive_replay']
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.2), layout='constrained')
        labels = ['XGBoost', 'Cascade', 'Random']
        for ax, column, label in zip(axes, ['misses_per_1000', 'false_alarms_per_1000', 'requests_per_1000'],
                                     ['Missed positives', 'False alarms', 'JEV referrals']):
            values = table[column]
            bars = ax.bar(labels, values, color=['#64748b', '#007872', '#9a5b10'], width=.62)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
            ax.set(ylim=(0, max(values) * 1.22), title=label, ylabel='Per 1,000 records')
        fig.suptitle('Asymmetric routing replay at the shared XGBoost threshold')
        save(fig, 'cached_routing_outcomes')

        fig, ax = plt.subplots(figsize=(5.3, 3.7), layout='constrained')
        for (label, mean_score, observed), color in zip(analysis['calibration_curves'], ['#9a5b10', '#007872']):
            ax.plot(mean_score, observed, marker='o', label=label, color=color)
        ax.plot([0, 1], [0, 1], ':', color='#94a3b8')
        ax.set(xlabel='Mean rating or calibrated score', ylabel='Observed positive label fraction',
               title='Calibration for cached comparator-negative records', xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc='lower right')
        save(fig, 'calibration')


def _markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |',
                      *['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows]])


def _main_table(frame):
    return _markdown_table(['Policy', 'Precision', 'Recall', 'F1', 'F2', 'Requests'], [
        [row.policy, f'{row.precision:.3f}', f'{row.recall:.3f}', f'{row.f1:.3f}', f'{row.f2:.3f}', f'{row.requests:,.0f}']
        for row in frame.itertuples(index=False)])


def _confusion_table(frame):
    return _markdown_table(['Policy', 'TP', 'FP', 'FN', 'TN'], [
        [row.policy, *[f'{x:,.1f}' if row.policy == RANDOM else f'{x:,.0f}' for x in [row.tp, row.fp, row.fn, row.tn]]]
        for row in frame.itertuples(index=False)])


def _local_benchmarks(output):
    records = []
    for path in sorted(output.glob('local_benchmark_*.json')):
        r = json.loads(path.read_text())
        records.append([r['model'], f'{r["validation_threshold"]:.2f}', f'{r["validation"]["precision"]:.3f}',
                        f'{r["validation"]["recall"]:.3f}', f'{r["validation"]["f1"]:.3f}',
                        f'{r["seconds_per_1000_median"]:.3f}', f'{r["observed_inference_peak_rss_mib"]:.1f}'])
    if not records:
        return 'The real-data TF-IDF and end-to-end resource benchmarks await the original training and validation texts.'
    return _markdown_table(['Model', 'Cutoff', 'Precision', 'Recall', 'F1', 's/1,000', 'Peak MiB'], records)


def write_manuscript(analysis, model, resources, residual, fresh, tables, output, affiliation):
    grid = analysis['threshold_grid']
    chosen = grid.loc[grid.selected].iloc[0]
    replay_table = analysis['archive_replay']
    base, cascade, random = (replay_table.set_index('policy').loc[name] for name in [BASELINE, CASCADE, RANDOM])
    random_sensitivity = analysis['random_contrast'].iloc[0]
    random_alarms = analysis['random_contrast'].iloc[1]
    interval = analysis['paired_intervals']['recall']['ci95']
    r_lo, r_hi = analysis['policy']['routing_interval']
    r_score = analysis['policy']['calibrated_jev_cutoff']
    old = analysis['historical_or']
    auc = analysis['delong']
    improvements = analysis['calibration_audit']
    n = int(base.n)
    name_note = affiliation or 'Pending confirmation of official institutional name'
    fresh_text = f'''Fresh sample status: **{fresh['state']}**. The Study 2 ledger shows {fresh['actual_new_attempts']:,} attempted calls and {fresh['unused_allowance']:,} remaining. The inspected environment returned proxy CONNECT 403 for Metis, Kaggle and IEEE, and the complete private score and text frames are absent. The exact dataset-label provenance and applicable transfer terms are still unresolved.'''
    fresh_summary = 'Fresh evaluation and its population-level random comparator are pending in the delivered snapshot.'
    if tables:
        fresh_summary = f"The weighted follow-up covers a residual frame of {fresh['population_n']:,} records."
        fresh_text = f'''The frozen residual frame contains {fresh['population_n']:,} records. The analysis sample contains {fresh['sample_n']:,}. The ledger records {fresh['actual_new_attempts']:,} attempts, {fresh['successes']:,} successful responses and {fresh['failed']:,} failed requests. Paired design-based intervals are available in `fresh_paired.json`.

{_main_table(tables['fresh_population'])}'''
    ranking_table = _markdown_table(['Score', 'ROC-AUC', 'PR-AUC', 'Average precision'], [
        [row.model, f'{row.roc_auc:.3f}', f'{row.pr_auc_trapezoidal:.3f}', f'{row.average_precision:.3f}']
        for row in analysis['ranking'].itertuples(index=False)])
    report = f'''# Asymmetric referral for suicide-related text classification

**Supervisor revision**

**Negar Forouzan affiliation.** {name_note}.

## Abstract

An original-validation threshold of **{chosen.threshold:.2f}** was selected for Word2Vec–XGBoost with Youden’s J. The asymmetric cascade keeps every positive comparator decision and refers only comparator negatives with scores in **[{r_lo:.2f}, {r_hi:.2f})**. On {n:,} previously inspected text records, it recovered **{int(base.fn-cascade.fn)} positive labels**, added **{int(cascade.fp-base.fp)} false alarms**, and required **{int(cascade.requests)} prospective requests**. Recall increased from {base.recall:.2%} to {cascade.recall:.2%}. These results are exploratory. {fresh_summary}

## 1. Model identity and reference labels

The recorded service is **Metis Jev API**, called with provider alias **`{model['provider_request_alias']}`**. The archived inference window was {model['historical_started_tehran'][:10]}, {model['historical_started_tehran'][11:19]} to {model['historical_finished_tehran'][11:19]} Tehran time. The provider’s immutable backend version, architecture and parameter count are unavailable. Requests use generation defaults, parse `answers.department.score`, and count failures without automatic retries. The endpoint, exact scoring instructions and prompt hash are in `model_and_api.json`. “Perplexity Decider” is an undocumented name from the earlier report and has been removed from the comparisons. Candidate open-weight models remain untested.

The repository identifies Kaggle `nikhileswarkomati/suicide-watch` version 14. The full dataset checksum is in `reproducibility.json`. The input is one exported text record. The source pipeline maps the supplied CSV `class` to `suicide=1` or `non-suicide=0`. The possible Reddit and subreddit origin, expert participation, post or author annotation unit and linkage to IEEE document 9591887 require primary-source verification. A subreddit-derived label, if confirmed, measures source membership used as a content proxy. The observed outcome is the supplied text-record label. [Dataset release](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) · [IEEE source supplied by the project owner](https://ieeexplore.ieee.org/document/9591887)

## 2. Comparator and referral rule

Table 1 identifies every cutoff discussed in the feedback. The 0.38 entry is the rounded F1 threshold identifiable in the archived development output. The earlier standalone PDF was unavailable for a cell-by-cell comparison.

**Table 1. XGBoost threshold provenance.**

| Cutoff | Selection data and criterion | Role |
|:---|:---|:---|
| 0.383297 | 2,000 cached development records, maximum F1 | Earlier exploratory baseline, shown as 0.38 |
| 0.44 | Original validation, maximum F1 on the original grid | Centralized notebook’s historical tuned baseline |
| 0.50 | Fixed cutoff specified before historical comparison | Historical replacement and OR accounting |
| **{chosen.threshold:.2f}** | **Original validation, maximum Youden’s J on 0.05 edges** | **Shared cutoff for the revised comparator and cascade** |

The original validation histogram contains {int(chosen.n):,} records. At the archived 0.50 decision, bin errors below 0.50 are positives and bin errors above or at 0.50 are negatives. This recovers the confusion matrix at every eligible 0.05 edge exactly. The rule maximizes sensitivity + specificity − 1, with ties resolved by higher sensitivity and then higher cutoff. The selected validation confusion counts are TP {int(chosen.tp):,}, FP {int(chosen.fp):,}, FN {int(chosen.fn):,} and TN {int(chosen.tn):,}. Validation precision is {chosen.precision:.3f}, recall {chosen.recall:.3f}, F1 {chosen.f1:.3f}, and Youden’s J {chosen.youden_j:.4f}. A finer optimum cannot be inferred from this histogram.

The 51.41% historical error peak in [0.45, 0.50) concerns the 0.50 comparator. These records are **positive** at the revised 0.45 threshold. The validation errors at the historical outer edges are 27.52% for [0.25, 0.30) and 25.93% for [0.75, 0.80). Maximizing the number of possible referrals would mix regions with different error profiles. Under the revised policy, candidate intervals end at 0.45. Intervals starting below 0.25 lack archived JEV scores. On the 2,000-record cached development split, supported intervals and sigmoid decision cutoffs were compared by F2 improvement per referral with a minimum of 100 available referral records. Five-fold cross-fitting used the supplied label as the calibration target. The final sigmoid is fitted on successful historical development responses within the comparator-negative coverage. The chosen routing interval is [{r_lo:.2f}, {r_hi:.2f}), with a calibrated score cutoff of {r_score:.2f}. Neither the raw score nor this cutoff specifies an individual’s clinical risk.

For XGBoost score `p`, the comparator is positive when `p >= {chosen.threshold:.2f}`. Refer only `p` in `[{r_lo:.2f}, {r_hi:.2f})`. A successful JEV decision can add a positive flag. A failed or unavailable response retains the comparator outcome. This add-only rule preserves all comparator detections and can increase false alarms. The primary effectiveness question is the sensitivity gain and accompanying false alarms at fixed request volume.

## 3. Historical results and paired statistics

Table 2 shows the {n:,}-record cache replay. All policies use the same XGBoost cutoff and denominator. The random policy refers uniformly among the observed comparator negatives and uses the identical JEV decision. The equal budgets are expected policy counts on the cache. Precision for the random policy is the ratio of expected confusion totals.

**Table 2. Exploratory policy metrics at a shared XGBoost threshold.**

{_main_table(replay_table)}

**Table 3. Confusion counts, with expected counts for random referral.**

{_confusion_table(replay_table)}

The cascade’s sensitivity difference against XGBoost was {100*(cascade.recall-base.recall):+.2f} percentage points (paired row-bootstrap 95% interval {100*interval[0]:+.2f} to {100*interval[1]:+.2f}). Per 1,000 records the cascade added {cascade.false_alarms_per_1000-base.false_alarms_per_1000:.2f} false alarms and recovered {base.misses_per_1000-cascade.misses_per_1000:.2f} missed positive labels. Against expected random referral within cached negatives, the recall difference was {100*random_sensitivity.estimate:+.2f} points (95% interval {100*random_sensitivity.ci_low:+.2f} to {100*random_sensitivity.ci_high:+.2f}). The difference in false alarms was {random_alarms.estimate:+.2f} per 1,000 (95% interval {random_alarms.ci_low:+.2f} to {random_alarms.ci_high:+.2f}). Those comparisons condition on the observed score-range cache and its fitted policy. They cannot establish the same advantage among all comparator negatives.

The exact McNemar table includes overall errors, sensitivity on positive labels, and specificity on negative labels for the two deterministic rules. Counts and p-values are in `mcnemar.json`. Author clusters are unavailable, so the tests and paired intervals assume record independence for descriptive inference. Population sampling and expected-random totals use paired stratified intervals during fresh replay. The results are associated with dataset labels and previous examination of the cache.

As a historical check, OR at the old 0.50 thresholds would have referred {old['referrals']:,} of {old['n']:,} cached records, recovered {old['recovered_positives']} positives and added {old['added_false_alarms']} false alarms. Applying these exact changes only to the historical selected rows and retaining the 0.50 comparator for the rest gives a full-test replay sensitivity of {old['full_frame_replayed_recall']:.2%} versus {old['full_frame_baseline_recall']:.2%} base. The historical replacement that reduced false alarms from 953 to 248 also lost 511 correctly detected positive labels. It is a secondary audit of the earlier design.

![Historical policy replay and request burden](figures/cached_routing_outcomes.png)

### Score ranking and calibration

For the same cached records, the continuous historical Metis score with XGBoost fallback and the continuous XGBoost score support paired DeLong analysis. The paired ROC-AUC difference was {auc['delta_auc']:+.3f} (95% interval {auc['ci_low']:+.3f} to {auc['ci_high']:+.3f}). Tied ratings receive midrank credit. Binary cascade and expected-random decisions are evaluated at their frozen operating point. PR-AUC below uses trapezoidal integration. Average precision uses the step-weighted definition.

{ranking_table}

![ROC and precision–recall curves for continuous scores](figures/continuous_scores_roc_pr.png)

For {int(improvements.n.iloc[0]):,} successful held-out ratings among cached comparator negatives, Brier loss changes from {improvements.brier.iloc[0]:.3f} to {improvements.brier.iloc[1]:.3f}; log loss changes from {improvements.log_loss.iloc[0]:.3f} to {improvements.log_loss.iloc[1]:.3f}. Calibration applies to this supplied label under this score-region distribution. The verified response field exposes a scalar rating. Token logits and log-probabilities are unavailable in the export. The protocol specifies a qualitative alternative for a separately validated model and corpus.

## 4. Prospective evaluation and independence

The original test includes {residual['original_test_n']:,} records and {residual['old_interval_n']:,} in the old interval. Of the {residual['unqueried_in_old_interval_before_other_history']:,} outside the old 5,000-record study, {residual['additional_prior_test_inside_old_interval_n']} were already selected in the error study. After those known exclusions, {residual['potential_old_interval_n']:,} records could remain in the original interval and {residual['potential_test_n']:,} in the original test frame before further exclusions. The size of the new [{r_lo:.2f}, {r_hi:.2f}) stratum requires complete original scores. The 3,000-record archive replay is evaluation-disjoint from the new fit, but all 5,000 were previously inspected. The residual test is previously unqueried at the record level after exclusions and previously reported in aggregate. A genuinely independent sample must come from an untouched source or time/author cohort.

The frozen residual-study design samples other comparator negatives from the full remaining test frame as random-referral controls. It uses the full reviewed stratum, all comparator positives for local scoring, and one response per sampled comparator negative. Probability weights recover the residual population. Both assisted policies use the same eligible decision and the same population request count. Protocol, model, manifest and reference hashes are sealed before opening evaluation outcomes. Reviewers can audit this method without spending the 10,000-attempt ceiling. All-text dispatch serves as a cost reference only.

{fresh_text}

## 5. Computational accounting and local models

Word2Vec averages a compact skip-gram representation and the CPU XGBoost scorer is retained as a lightweight baseline. Averaging loses much of the order and context used for negation, quotation, temporal reference, and expressions of intent. A local word and character TF-IDF logistic regression comparator uses the same upstream splits and Youden criterion. Benchmark commands time each model’s full warm text-to-score path on original validation texts and record observed process RSS.

{_local_benchmarks(output)}

The archived XGBoost scoring batch took {resources['archived_test_prediction_seconds']:.3f} seconds for {resources['archive_test_n']:,} pre-tokenized records, equivalent to {resources['archived_prepared_prediction_seconds_per_1000']:.6f} seconds per 1,000 in that batch. This measures the embedding and scoring paths only. Archived Metis collection took {resources['historical_jev_active_phase_wall_seconds']:.1f} active wall seconds for {model['historical_attempts']:,} attempts with concurrent workers. Dividing group wall time by attempts reports batch throughput. Single-request timing and RSS were not retained in those exports.

The cache replay estimates **{cascade.requests_per_1000:.1f}** referrals per 1,000 for the cascade and equally many for its random comparator; all-referral would require **1,000**. This is a {resources['revised_cached_api_attempt_reduction_pct']:.1f}% request reduction in this cache, with actual paid attempts recorded separately in `fresh_status.json`. Input and output tokens and verified USD charges are unavailable. USD per 1,000 should be derived from measured token usage and rate-card prices for the relevant request mix. Token, latency and cost reductions need their own measurements.

## 6. Conclusion

The revised architecture protects the tuned comparator’s positive decisions and uses a supported, development-selected uncertainty interval. Archived evidence supports studying the sensitivity and false-alarm trade-off under a lower request budget. Fresh evaluation against verified dataset labels is needed to estimate the population benefit over equal-budget random referral. Clinical claims require a reference and population that measure the clinical target.
'''
    (output / 'manuscript_report.md').write_text(report, encoding='utf-8')


def render_revision(root, *, affiliation=None):
    root = Path(root).resolve()
    output = root / 'research/results' / REVISION
    output.mkdir(parents=True, exist_ok=True)
    analysis = analyze_archive(root)
    model, resources, residual = evidence_audit(root, analysis)
    fresh, fresh_tables = replay(root)
    for name, table in {'original_validation_threshold_grid': analysis['threshold_grid'],
                        'development_routing_sweep': analysis['routing_sweep'], 'validation_bin_audit': analysis['bin_audit'],
                        'cached_policy_comparison': analysis['archive_replay'], 'mcnemar': analysis['mcnemar'],
                        'continuous_ranking': analysis['ranking'], 'calibration_audit': analysis['calibration_audit'],
                        'cached_random_contrasts': analysis['random_contrast'], **fresh_tables}.items():
        save_table(output, name, table)
    for name, obj in {'revised_policy': analysis['policy'], 'delong_continuous_scores': analysis['delong'],
                      'cached_base_contrasts': analysis['paired_intervals'], 'historical_or_check': analysis['historical_or'],
                      'model_and_api': model, 'resource_accounting': resources, 'residual_frame_accounting': residual,
                      'fresh_status': fresh}.items():
        save_json(output / f'{name}.json', obj)
    if not fresh_tables:
        save_table(output, 'fresh_results_pending', pd.DataFrame([{'stratum': stratum, 'n': None, 'status': fresh['state']}
                                                                 for stratum in STRATA]))
    archive = root / 'research/results/jev_probability_hybrid'
    error_study = root / 'research/results/jev_error_study'
    sources = [archive / name for name in ['validation_bins.json', 'predictions.json', 'study.json',
                                           'hybrid_report.json', 'execution_audit.json']]
    sources.extend(error_study / name for name in ['xgboost_baseline.json', 'selection_manifest.json'])
    save_json(output / 'reproducibility.json', {'policy_sha256': digest(analysis['policy']),
              'dataset_sha256': analysis['baseline_archive']['identity']['dataset_sha256'],
              'baseline_fingerprint': analysis['baseline_archive']['baseline_fingerprint'],
              'inputs': {str(p.relative_to(root)): file_digest(p) for p in sources}, 'affiliation': affiliation,
              'analysis_code': {name: file_digest(root / 'jev_research' / name) for name in
                                ['cascade_revision.py', 'cascade_validation.py', 'cascade_report.py']},
              'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn']}})
    _figure_set(analysis, output)
    write_manuscript(analysis, model, resources, residual, fresh, fresh_tables, output, affiliation)
    return analysis, fresh, output
