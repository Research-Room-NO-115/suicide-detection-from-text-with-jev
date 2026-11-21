"""Prepare and replay a frozen residual-test sample for the asymmetric cascade.

The collection command reuses Study 2's sole 10,000-attempt ledger. Preparation
and replay make no HTTP requests. Reference labels are sealed before collection
and read for analysis only after the complete planned attempt set is recorded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm

from .additional_experiment import manifest_hash
from .budgeted_experiments import digest, file_digest, save_json, save_table
from .cascade_revision import BASELINE, CASCADE, RANDOM, REVISION, analyze_archive, cascade_arrays, policy_metrics
from .client import BASE_MODEL, ENDPOINT, QUESTION, text_sha256
from .notebook_pipeline import _read_cache
from .study2 import ATTEMPT_CEILING, SEED, STUDY, contributions, design_totals, summarize_totals
from .study2_collection import collect, ledger_status, write_frozen_package

STRATA = ('uncertain_negative', 'other_negative', 'comparator_positive')
IDENTITY = ['row_id', 'text_sha256', 'p_xgb', 'source_split', 'dataset_sha256', 'baseline_fingerprint']
CONTROL_HALF_WIDTH = .025


def _load_manifest(path):
    # Keep JSON decimal types exactly as written, including census probabilities 1.0.
    return pd.DataFrame(json.loads(path.read_text()))


def control_sample_size(population):
    """Worst-case normal 95% half-width for a binary control rate with FPC."""
    infinite = .25 * (norm.ppf(.975) / CONTROL_HALF_WIDTH) ** 2
    return min(population, max(2, int(np.ceil(population * infinite / (population - 1 + infinite)))))


def sample_residual(test, exclusions, policy, *, seed=SEED):
    """Census the route and local-positive stratum, sample remaining negatives."""
    if test.empty or not test.source_split.eq('test').all() or not set(IDENTITY).issubset(test):
        raise ValueError('A complete original test frame with saved baseline scores is required.')
    for key in ['row_id', 'text_sha256']:
        if test[key].isna().any() or test[key].duplicated().any():
            raise ValueError(f'Invalid test identity in {key}.')
    provenance = ['dataset_sha256', 'baseline_fingerprint']
    if any(test[key].nunique() != 1 for key in provenance):
        raise ValueError('A common baseline and dataset fingerprint are required.')
    # Selection uses only approved identity, text-matching keys and XGBoost scores.
    matching_keys = [key for key in ['cleaned_text', 'author_id'] if key in test]
    pool = test[IDENTITY + matching_keys].copy()
    dataset_hash = test.dataset_sha256.iloc[0]
    exclusion_log = []
    for name, prior in exclusions.items():
        overlap = pool.text_sha256.isin(prior.text_sha256)
        if {'row_id', 'dataset_sha256'}.issubset(prior) and prior.dataset_sha256.eq(dataset_hash).all():
            overlap |= pool.row_id.isin(prior.row_id)
        for key in matching_keys:
            if key in prior:
                known = prior[key].dropna()
                overlap |= pool[key].isin(known[known.astype(str).str.strip().ne('')])
        exclusion_log.append({'source': name, 'removed': int(overlap.sum())})
        pool = pool.loc[~overlap].copy()

    lower, threshold = policy['routing_interval']
    if threshold != policy['xgb_threshold'] or not 0 <= lower < threshold <= 1:
        raise ValueError('The frozen route must end at the XGBoost threshold.')
    if not np.isfinite(pool.p_xgb).all() or not pool.p_xgb.between(0, 1).all():
        raise ValueError('Invalid XGBoost probabilities in the test frame.')
    pool['stratum'] = np.select([pool.p_xgb.ge(threshold), pool.p_xgb.ge(lower)],
                               [STRATA[2], STRATA[0]], default=STRATA[1])
    counts = pool.stratum.value_counts().reindex(STRATA, fill_value=0).astype(int).to_dict()
    if any(n < 2 for n in counts.values()):
        raise ValueError('This design needs at least two records in each residual stratum.')
    quotas = {**counts, STRATA[1]: control_sample_size(counts[STRATA[1]])}
    attempts = quotas[STRATA[0]] + quotas[STRATA[1]]
    if attempts > ATTEMPT_CEILING:
        raise ValueError('The residual-frame collection plan exceeds the Study 2 authorization.')
    rng = np.random.default_rng(seed)
    parts, strata = [], []
    for name in STRATA:
        population = pool.loc[pool.stratum.eq(name), IDENTITY].sort_values('row_id')
        N, n = counts[name], quotas[name]
        indices = np.sort(rng.choice(N, size=n, replace=False)) if n < N else np.arange(N)
        part = population.iloc[indices].copy()
        part['sample_id'] = [text_sha256(f'{d}:{i}:{h}') for d, i, h in
                             zip(part.dataset_sha256, part.row_id, part.text_sha256)]
        strata.append({'stratum': name, 'population_n': N, 'sample_n': n, 'inclusion_probability': n / N,
                       'jev_collection': name != STRATA[2]})
        for key, value in strata[-1].items():
            part[key] = value
        part['study_partition'] = 'evaluation'
        parts.append(part)
    sample = pd.concat(parts, ignore_index=True)
    negative_n = counts[STRATA[0]] + counts[STRATA[1]]
    design = {'seed': seed, 'population_n': len(pool), 'strata': strata, 'collection_n': attempts,
              'excluded': exclusion_log, 'population_negative_n': negative_n,
              'population_route_n': counts[STRATA[0]],
              'random_referral_probability_among_negatives': counts[STRATA[0]] / negative_n,
              'outside_control_planning_half_width': CONTROL_HALF_WIDTH,
              'independence': 'previously unqueried original test records after known exclusions; original test aggregates were reported',
              'grouping': 'author exclusion where identifiers exist; near-duplicate and temporal generalization unresolved'}
    return sample, design


def _paths(root):
    private = Path(root) / 'artifacts' / STUDY
    return private / 'cascade_v2', private / 'budget.sqlite3'


def _cache(root, cache_directory):
    metadata = json.loads((cache_directory / 'metadata.json').read_text())
    reference = json.loads((Path(root) / 'research/results/jev_error_study/xgboost_baseline.json').read_text())
    if (metadata['identity'] != reference['identity'] or
            metadata['baseline_fingerprint'] != reference['baseline_fingerprint'] or
            metadata['split_counts'] != reference['split_counts']):
        raise ValueError('The revised cutoff is valid only for the historical baseline fingerprint and split.')
    return _read_cache(cache_directory, reference['identity'], reference['baseline_fingerprint']).frames


def prepare(root, cache_directory, reference_evidence, transfer_evidence):
    """Freeze model, eligible records and the analysis design before opening labels."""
    root, cache_directory = Path(root), Path(cache_directory)
    reference = reference_evidence
    if not reference.get('verified') or not all(reference.get(k) for k in
           ['release', 'dataset_sha256', 'source_citation', 'annotation_unit', 'label_construction']):
        raise ValueError('Document the dataset release and text-label provenance before fresh collection.')
    if reference.get('reference_type') != 'dataset_text_labels':
        raise ValueError('The revised experiment evaluates the documented dataset text labels.')
    if not transfer_evidence.get('permitted') or not all(transfer_evidence.get(k) for k in
           ['source_terms_sha256', 'interpretation']):
        raise ValueError('Document applicable terms for transmitting these texts to the provider.')
    work, ledger = _paths(root)
    if ledger.exists() or (work / 'protocol.json').exists():
        raise ValueError('Use the existing frozen Study 2 package and ledger; they cannot be replaced on preparation.')

    frames = _cache(root, cache_directory)
    archive = analyze_archive(root)
    if reference['dataset_sha256'] != archive['baseline_archive']['identity']['dataset_sha256']:
        raise ValueError('Reference evidence must match the exact source archive checksum.')
    results = root / 'research/results'
    exclusions = {'train': frames['train'], 'validation': frames['validation']}
    historical_paths = {'old_probability_study': 'jev_probability_hybrid/routing_manifest.json',
                        'old_error_study': 'jev_error_study/selection_manifest.json',
                        'old_optional_study': 'jev_probability_hybrid_budgeted_experiments/additional_manifest.json'}
    test = frames['test'].set_index('row_id', drop=False)
    for name, rel in historical_paths.items():
        path = results / rel
        if path.is_file():
            history = pd.read_json(path, precise_float=True)
            exclusions[name] = history
            if 'author_id' in test:
                matched = test.loc[test.text_sha256.isin(history.text_sha256)]
                exclusions[name] = history.merge(matched[['text_sha256', 'author_id']], on='text_sha256', how='left')
    old = exclusions['old_probability_study'].set_index('row_id', drop=False)
    aligned = test.reindex(old.index)
    if (not np.array_equal(aligned.text_sha256, old.text_sha256) or
            not np.allclose(aligned.p_xgb, old.p_xgb, rtol=0, atol=1e-14)):
        raise ValueError('Restored test scores and historical study identities disagree.')
    sample, design = sample_residual(test.reset_index(drop=True), exclusions, archive['policy'])
    manifest = sample.loc[sample.jev_collection].copy()
    design['sample_manifest_sha256'] = manifest_hash(sample)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    save_json(work / 'analysis_sample.json', sample.to_dict('records'))
    source = test.loc[manifest.row_id, IDENTITY + ['text']]
    source.to_csv(work / 'texts_for_collection.csv.gz', index=False)
    # Seal the reference file without inspecting any outcome values by stratum.
    label_columns = ['row_id', 'text_sha256', 'y_true'] + (['author_id'] if 'author_id' in test else [])
    reference_path = work / 'reference_private.csv.gz'
    test.loc[sample.row_id, label_columns].to_csv(reference_path, index=False)
    for path in [reference_path, work / 'texts_for_collection.csv.gz']:
        path.chmod(0o600)

    protocol = {'study_id': STUDY, 'revision': REVISION, 'status': 'FROZEN_INTERNAL_PROTOCOL',
                'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'candidate_policy': archive['policy'],
                'reference_evidence': reference, 'transfer_evidence': transfer_evidence,
                'reference_file_sha256': file_digest(reference_path),
                'primary_analysis': 'paired sensitivity gain and added false alarms per 1000',
                'noninferiority_margin': None,
                'evaluation_manifest_sha256': manifest_hash(manifest), 'design_sha256': digest(design),
                'gates': {gate: True for gate in ['primary_source_and_construct_verified',
                          'dataset_label_provenance_verified', 'external_transfer_terms_permit',
                          'complete_sampling_frame_verified', 'frozen_manifest_and_protocol']}}
    write_frozen_package(work, protocol, manifest, design)
    return {'status': 'frozen', 'protocol_sha256': digest(protocol), 'collection_n': len(manifest),
            'analysis_sample_n': len(sample), 'design': design}


def weighted_analysis(frame, policy, design, *, replicates=2000):
    """The population request equality uses known frame stratum counts."""
    if frame.sample_id.duplicated().any() or set(frame.stratum) != set(STRATA):
        raise ValueError('The analysis requires distinct records from all frozen strata.')
    if 'author_id' in frame:
        authors = frame.author_id.dropna().astype(str).str.strip()
        if authors[authors.ne('')].duplicated().any():
            raise ValueError('Use paired author-level variance estimation for these repeated author IDs.')
    y = frame.y_true.to_numpy(int)
    outputs, _ = cascade_arrays(frame, policy, design['random_referral_probability_among_negatives'])
    weights = 1 / frame.inclusion_probability.to_numpy(float)
    masks = np.column_stack([contributions(y, *outputs[name]) for name in outputs])
    _, simulations = design_totals(frame, masks, replicates=replicates)
    if not np.isclose(weights.sum(), design['population_n']):
        raise ValueError('Evaluation weights do not recover the frozen residual population.')
    callers = [policy_metrics(y, *outputs[name], weights=weights)['requests'] for name in [CASCADE, RANDOM]]
    if not np.allclose(callers, design['population_route_n']):
        raise ValueError('The two assisted policies do not match the population request budget.')

    table, distributions = [], {}
    for index, name in enumerate(outputs):
        segment = slice(5 * index, 5 * (index + 1))
        metrics = policy_metrics(y, *outputs[name], weights=weights)
        row = {'policy': name, 'sampled_records': len(frame), **metrics}
        distributions[name] = summarize_totals(simulations[:, segment])
        table.append(row)
    main = pd.DataFrame(table)
    by_policy = main.set_index('policy')
    contrasts = []
    for reference in [BASELINE, RANDOM]:
        for metric in ['sensitivity', 'precision', 'specificity', 'false_alarms_per_1000', 'missed_per_1000']:
            index = {'sensitivity': 'recall', 'missed_per_1000': 'misses_per_1000'}.get(metric, metric)
            diff = by_policy.loc[CASCADE, index] - by_policy.loc[reference, index]
            samples = distributions[CASCADE][metric] - distributions[reference][metric]
            finite = samples[np.isfinite(samples)]
            low, high = np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]
            contrasts.append({'contrast': f'{CASCADE} minus {reference}', 'outcome': metric,
                              'estimate': diff, 'ci_low': low, 'ci_high': high,
                              'interval_scope': 'paired stratified finite-frame, fixed policy and responses'})
    by_stratum = []
    for name, (pred, requests) in outputs.items():
        for stratum in STRATA:
            subset = frame.stratum.eq(stratum).to_numpy()
            by_stratum.append({'policy': name, 'stratum': stratum, 'observed_records': int(subset.sum()),
                               **policy_metrics(y[subset], pred[subset], requests[subset], weights[subset])})
    return main, pd.DataFrame(contrasts), pd.DataFrame(by_stratum)


def replay(root):
    """Verify frozen identities and replay complete recorded attempts, read-only."""
    work, ledger = _paths(root)
    status = ledger_status(ledger)
    if not (work / 'protocol.json').is_file():
        return {**status, 'state': 'fresh sample not frozen'}, {}
    protocol = json.loads((work / 'protocol.json').read_text())
    design = json.loads((work / 'design.json').read_text())
    manifest = _load_manifest(work / 'manifest.json')
    sample = _load_manifest(work / 'analysis_sample.json')
    if (protocol['revision'] != REVISION or digest(design) != protocol['design_sha256'] or
            manifest_hash(manifest) != protocol['evaluation_manifest_sha256'] or
            manifest_hash(sample) != design['sample_manifest_sha256']):
        raise ValueError('Frozen asymmetric protocol/sample identity has changed.')
    if status['actual_new_attempts'] < len(manifest):
        return {**status, 'state': 'collection incomplete', 'planned_attempts': len(manifest)}, {}
    with sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        metadata = dict(connection.execute('SELECT key,value FROM metadata'))
        attempts = pd.read_sql_query('SELECT * FROM attempts ORDER BY id', connection)
        elapsed = pd.read_sql_query('SELECT elapsed_seconds FROM spending', connection).elapsed_seconds.dropna()
    if metadata.get('evaluation_decisions_sha256') != digest(protocol) or len(attempts) != len(manifest):
        raise ValueError('The retained ledger does not match the frozen asymmetric protocol.')
    joined = manifest.merge(attempts, on=['sample_id', 'text_sha256'], validate='one_to_one')
    if len(joined) != len(manifest):
        raise ValueError('Recorded outcomes differ from the frozen collection set.')
    for row in joined.itertuples(index=False):
        expected = digest({'sample_id': row.sample_id, 'text_sha256': row.text_sha256,
                           'model': BASE_MODEL, 'question_sha256': digest(QUESTION), 'endpoint': ENDPOINT})
        if (row.request_key != expected or row.model != BASE_MODEL or row.instruction_sha256 != digest(QUESTION) or
                datetime.fromisoformat(row.started_at) < datetime.fromisoformat(protocol['frozen_at_utc'])):
            raise ValueError('A recorded call is incompatible with the sealed request.')

    reference_file = work / 'reference_private.csv.gz'
    if file_digest(reference_file) != protocol['reference_file_sha256']:
        raise ValueError('The sealed reference labels have changed.')
    reference = pd.read_csv(reference_file, keep_default_na=False)
    frame = sample.merge(reference, on=['row_id', 'text_sha256'], validate='one_to_one')
    frame = frame.merge(attempts[['sample_id', 'score', 'status']], on='sample_id', how='left', validate='one_to_one')
    frame = frame.rename(columns={'score': 'jev_score', 'status': 'jev_status'})
    frame['jev_status'] = frame.jev_status.fillna('not_attempted')
    if len(frame) != len(sample) or not frame.y_true.isin([0, 1]).all():
        raise ValueError('The restored reference is inconsistent with the frozen sample.')
    main, paired, strata = weighted_analysis(frame, protocol['candidate_policy'], design)
    return {**status, 'state': 'replayed', 'population_n': design['population_n'],
            'sample_n': len(sample), 'timed_attempts': len(elapsed),
            'median_completed_request_seconds': float(elapsed.median()) if len(elapsed) else None,
            'p95_completed_request_seconds': float(elapsed.quantile(.95)) if len(elapsed) else None,
            'verified_usd_cost': None, 'observed_token_usage': None}, {
                'fresh_population': main, 'fresh_paired': paired, 'fresh_strata': strata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--baseline-cache', type=Path, required=True)
    preparation.add_argument('--reference-evidence', type=Path, required=True)
    preparation.add_argument('--transfer-evidence', type=Path, required=True)
    collection = commands.add_parser('collect')
    collection.add_argument('--workers', type=int, default=4)
    commands.add_parser('status')
    commands.add_parser('replay')
    args = parser.parse_args()
    work, ledger = _paths(args.root)
    if args.command == 'prepare':
        output = prepare(args.root, args.baseline_cache, json.loads(args.reference_evidence.read_text()),
                         json.loads(args.transfer_evidence.read_text()))
    elif args.command == 'collect':
        protocol = json.loads((work / 'protocol.json').read_text())
        if protocol.get('revision') != REVISION:
            raise ValueError('Use the frozen asymmetric-cascade package.')
        output = collect(_load_manifest(work / 'manifest.json'),
                         pd.read_csv(work / 'texts_for_collection.csv.gz', float_precision='round_trip', keep_default_na=False),
                         protocol, json.loads((work / 'design.json').read_text()), ledger_path=ledger,
                         run_new_inference=True, workers=args.workers)
    elif args.command == 'replay':
        output, tables = replay(args.root)
        results = args.root / 'research/results' / REVISION
        results.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            save_table(results, name, table)
        save_json(results / 'fresh_status.json', output)
    else:
        output = ledger_status(ledger)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
