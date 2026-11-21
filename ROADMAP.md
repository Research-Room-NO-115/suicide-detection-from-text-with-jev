# Project roadmap

Research history through **21 November 2026**. Dates use Tehran time. The
project progressed from testing whether Jev could correct individual XGBoost
errors to designing a selective policy that preserves positive decisions and
can be compared with random referral at the same request budget.

## Milestones

| Date | Milestone and commits | What the work established |
| --- | --- | --- |
| 21 Oct | Project initialization · `8e705b2`, `24ecd1e` | Added the original federated IID/non-IID experiment and the first API integration. Later studies reused its centralized Word2Vec–XGBoost pipeline. [Original executed notebook](assets/notebooks/federated_xgboost_distribution_iid_non_iid_executed.ipynb) |
| 21 Oct | Error-only pilot · `a035f18` | On 40 selected XGBoost errors, base Jev recovered 11 of 20 missed positives and corrected 15 of 20 false alarms. The sample estimates correction on selected errors. Jev weight fine-tuning remained pending. [Pilot results](research/results/jev_error_study/report.md) |
| 21 Oct | Probability-routed hybrid · `c03d1be` | Chose `[0.25, 0.80)` on validation data and sampled 5,000 eligible test texts without using their labels. Accuracy rose from 63.74% to 78.38% on those 5,000 and from 91.97% to 93.55% on the 46,342-record partial-routing replay. The policy recovered 538 misses and introduced 511, so the net sensitivity gain was small. [Hybrid results](research/results/jev_probability_hybrid/hybrid_report.md) |
| 16 Nov | Offline policy comparison · `fcc028e` | Compared tuned XGBoost, replacement, OR/AND, fusion and calibration on a 2,000/3,000 development/evaluation partition of the previously inspected cache. At the 95% development recall target, Jev yielded 568 fewer false alarms on the 3,000 records. The paired recall difference was +0.62 percentage points, with a 95% interval of −0.80 to +1.94. [Budgeted experiments notebook](assets/notebooks/jev_probability_hybrid_budgeted_experiments.ipynb) |
| 16 Nov | Study 2 design · `17fd040`, `c2d466b` | Added a source-label audit, an equal-budget random comparator, weighted analysis and a resumable collector under a 10,000-attempt ceiling. Fresh collection remained pending after checks found missing private score/text tables, unresolved label and transfer evidence, and proxy denials. [Study 2 protocol and collection notes](research/study2_collection_and_reporting.md) |
| 21 Nov | Asymmetric cascade after supervisor review · `b5f6d8b` | Selected a common XGBoost cutoff of **0.45** by Youden’s J on the original 37,086-record validation population. The cascade preserves comparator positives and refers negatives in **`[0.40, 0.45)`**. On 3,000 historical records, recall rose from 75.31% to 80.45%, with 83 recovered positives, 25 additional false alarms and 239 prospective calls. The revision added paired statistics, ROC and precision–recall curves, and a TF-IDF benchmark command for use with the recovered data. [Revised notebook](assets/notebooks/jev_asymmetric_cascade_revision.ipynb) · [Revised report](research/results/jev_asymmetric_cascade_revision/manuscript_report.pdf) |

Pull requests #1–#5 integrated these milestones into `master`. PR #5 merged
the asymmetric cascade revision on 21 November at `9dc00ed`.

The cloud workspace was prepared on 16 November with Python 3.12 and CPU
dependencies. The initial 83 tests and both then-current Jev notebooks passed
offline. By the cascade revision, both suites passed with 126 tests in total.
[Environment setup](scripts/setup_environment.sh)

## Request accounting

| Study | Recorded paid attempts | Scope |
| --- | ---: | --- |
| Error-only pilot | 40 / 100 | All 40 succeeded. [Pilot ledger summary](research/results/jev_error_study/inference_budget.json) |
| Probability-routed hybrid | 5,000 / 5,000 | 4,996 successful scores and four fallbacks to XGBoost. [Hybrid ledger summary](research/results/jev_probability_hybrid/inference_budget.json) |
| Optional budgeted follow-up | 0 / 500 planned | The 200-development/300-evaluation design is inactive. [Sampling plan](research/results/jev_probability_hybrid_budgeted_experiments/additional_experiment_plan.json) |
| Study 2 and cascade revision | 0 / 10,000 | One shared ceiling. The 239 calls in the historical replay describe the policy’s prospective request count. [Fresh collection status](research/results/jev_asymmetric_cascade_revision/fresh_status.json) |

## Current position and next milestone

The current comparison uses the shared, original-validation XGBoost cutoff.
At equal request count in the historical replay, cascade recall was 1.15
percentage points above expected random referral among cached comparator
negatives. The false-alarm difference remained uncertain. The cache’s score
coverage and previous exposure limit this result to exploratory analysis.
[Paired contrasts](research/results/jev_asymmetric_cascade_revision/cached_random_contrasts.json)

1. **Recover and verify the evaluation inputs.** Restore the original texts,
   labels and per-record scores. Check release identity, label provenance and
   transfer terms, and confirm usable Metis access. The supplied text labels
   are the current reference. Any expert or clinical claim requires evidence
   for that specific reference.
2. **Freeze and run the residual evaluation.** Exclude previously selected
   records, lock the uncertainty band and the equal-budget random controls,
   then use the resumable collector and shared ledger. The old interval may
   contain 1,062 records after known previous-study exclusions. The size of
   the revised interval requires the actual scores. An untouched source or
   author/time cohort would support independent confirmation.
3. **Report the new paired results and costs.** Compare recall, added false
   alarms and request burden with tuned XGBoost and random referral. Run the
   local TF-IDF and Word2Vec–XGBoost resource benchmarks on the common upstream
   splits once the texts are available. [Current evaluation protocol](research/asymmetric_cascade_protocol.md)
