"""Benchmark the two local baselines on the recovered upstream splits only."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
from pathlib import Path
import platform
import statistics
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .budgeted_experiments import metrics, save_json
from .cascade_revision import REVISION, validation_thresholds
from .cascade_validation import _cache
from .notebook_pipeline import DEFAULT_NOTEBOOK, _load_notebook_definitions
from .study2 import SEED, fit_local_text_baseline


def _rss_mib():
    # Current RSS, unlike a lifetime high-water mark, excludes an earlier fit peak.
    with Path('/proc/self/status').open() as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    raise RuntimeError('Linux process RSS is unavailable.')


def _time_batches(predict, texts, *, repeats=7):
    predict(texts)
    gc.collect()
    start_rss = _rss_mib()
    peak_rss = [start_rss]
    stop = Event()

    def monitor():
        while not stop.wait(.005):
            peak_rss[0] = max(peak_rss[0], _rss_mib())

    worker = Thread(target=monitor)
    worker.start()
    normalized = []
    try:
        for _ in range(repeats):
            start = perf_counter()
            predict(texts)
            normalized.append((perf_counter() - start) * 1000 / len(texts))
            peak_rss[0] = max(peak_rss[0], _rss_mib())
    finally:
        stop.set()
        worker.join()
    return {'batch_size': len(texts), 'repeats': repeats, 'seconds_per_1000_median': statistics.median(normalized),
            'seconds_per_1000_p95': float(np.quantile(normalized, .95)),
            'process_rss_before_inference_mib': start_rss, 'observed_inference_peak_rss_mib': peak_rss[0],
            'memory_scope': 'Linux process RSS sampled every 5 ms during the inference loop; includes model/runtime',
            'timing_scope': 'warm CPU text-to-score batches, including preprocessing; excludes fit, I/O and load'}


def _select_validation_cutoff(y, p):
    edges = np.arange(21) / 20
    indices = np.minimum(np.searchsorted(edges, p, side='right') - 1, 19)
    n = np.bincount(indices, minlength=20)
    errors = np.bincount(indices, weights=y != (p >= .5), minlength=20).astype(int)
    bins = pd.DataFrame({'lower': edges[:-1], 'upper': edges[1:], 'n': n, 'errors': errors,
                         'upper_inclusive': [False] * 19 + [True]})
    grid = validation_thresholds(bins)
    return float(grid.loc[grid.selected, 'threshold'].iloc[0])


def benchmark(root, cache_path, model_name):
    frames = _cache(root, cache_path)
    train, valid = frames['train'], frames['validation']
    selection = valid.sort_values('row_id').sample(n=min(1000, len(valid)), random_state=SEED)
    texts = selection.text.tolist()
    y = valid.y_true.to_numpy(int)
    identity = valid.baseline_fingerprint.iloc[0]
    with threadpool_limits(limits=1):
        if model_name == 'word2vec-xgboost':
            definitions = _load_notebook_definitions(DEFAULT_NOTEBOOK)
            local_model = definitions['LocalTextXGB'](seed=SEED)
            local_model.fit(train.cleaned_text.str.split().tolist(), train.y_true.to_numpy())

            def predict(raw):
                documents = [definitions['clean_text'](text).split() for text in raw]
                batch = definitions['SparseTokenBatch'](documents)
                return local_model.predict_batch(batch)[0]

            p_valid = predict(valid.text.tolist())
            # A recreated model can differ across CPU/numeric libraries. Require
            # agreement before attributing its validation metrics to the archive.
            max_score_difference = float(np.max(np.abs(p_valid - valid.p_xgb.to_numpy())))
            if max_score_difference > 1e-6:
                raise ValueError('Rebuilt Word2Vec–XGBoost scores differ from the archived baseline.')
            versions = {name: importlib.metadata.version(name) for name in ['gensim', 'xgboost-cpu']}
        else:
            local_model, _ = fit_local_text_baseline(train, valid)

            def predict(raw):
                return local_model.predict_proba(raw)[:, 1]

            p_valid = predict(valid.text.tolist())
            max_score_difference = None
            versions = {'scikit-learn': importlib.metadata.version('scikit-learn')}
        threshold = _select_validation_cutoff(y, p_valid)
        scores = metrics(y, p_valid >= threshold, p_valid)
        del frames, train, valid, selection
        timing = _time_batches(predict, texts)
    payload = {'model': model_name, 'scope': 'original validation, same 0.05 Youden grid, no fresh-test metrics',
               'upstream_baseline_fingerprint': identity, 'model_versions': versions, 'platform': platform.platform(),
               'cpu_threads': 1, 'seed': SEED, 'validation_threshold': threshold, 'validation': scores,
               'rebuilt_xgb_max_validation_score_difference': max_score_difference, **timing}
    output = Path(root) / 'research/results' / REVISION / f'local_benchmark_{model_name}.json'
    save_json(output, payload)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-cache', type=Path, required=True)
    parser.add_argument('--model', choices=['word2vec-xgboost', 'tfidf-logistic'], required=True)
    args = parser.parse_args()
    print(benchmark(Path.cwd(), args.baseline_cache, args.model))


if __name__ == '__main__':
    main()
