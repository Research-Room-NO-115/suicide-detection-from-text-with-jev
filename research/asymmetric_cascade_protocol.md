# Asymmetric cascade protocol and response to supervisor feedback

**Revision date** 2026-11-21, Tehran. **Status** Archive analysis executed.
Prospective collection awaits a verified frame. The time-stamped sample,
protocol digest, and reference-file seal are created together by `prepare`
before any new evaluation outcomes are opened.

## Research question and operating rule

Estimate how many additional supplied positive labels an add-only cascade
detects, and how many extra false alarms it generates, at a shared XGBoost
cutoff and a fixed referral budget. Compare against the tuned baseline and
random referral among comparator negatives in the same population. The
intended use is prioritization for contextual review of text records.

- **Comparator.** Word2Vec–XGBoost at **0.45**, maximizing Youden’s J on the
  original 37,086-record Validation grid at increments of 0.05. Ties prefer
  higher sensitivity, then higher threshold. The cached F1 cutoff 0.383297,
  historical original-validation F1 cutoff 0.44, and fixed 0.50 represent
  different experiments. The original history identifies the baseline
  fingerprint and the feature/model parameters.
- **Primary policy.** Preserve all XGBoost positives. Send comparator negatives
  only when their score is in **[0.40, 0.45)**. Promote the flag if the frozen
  calibrated rating is at least **0.40**. Failed, timed-out and uncertain
  requests retain the comparator decision. This guarantees no lost comparator
  positives at the shared cutoff, and allows additional false alarms.
- **Selection.** Only original XGBoost Validation determines its cutoff.
  Historical cache development rows determine the lower edge and rating
  cutoff using positive F2 gain per referral, at least 100 development
  referrals, and five-fold out-of-fold calibration. The final fit uses all
  successful covered comparator negatives on those development rows. Intervals
  with lower edge 0.15 or 0.20 lack cached provider outcomes below 0.25 and
  were excluded from selection. The 3,000-row replay is exploratory.
- **Scoring.** The exact request options and model identifier are recorded in
  `model_and_api.json`. The historical provider returned a rating. Applying
  a logistic sigmoid to its clipped raw logit targets the supplied binary
  labels in the covered score region. A monotone sigmoid does not improve
  rank discrimination by itself. The same frozen binary cutoff is used for
  the random control. Calibration outside coverage remains untested.

## Residual evaluation design

A genuinely independent source or time/author cohort with a compatible
reference is preferred for independent confirmation. The implemented
residual design is an initial evaluation on the original test split after
record-level exclusions. Its earlier aggregate exposure must be disclosed.

1. Reuse the verified original baseline cache with the exact dataset, model
   fingerprint, and split counts. Confirm original Validation/test separation
   and remove all known prior record and text identities, including historical
   failures and earlier error-study selections. Exclude cleaned duplicates
   and previous authors where corresponding identifiers are present. Report
   near-duplicate, author and temporal checks as unavailable when the data
   cannot support them. Current aggregates give 1,067 old-interval records
   after the original cohort subtraction and 1,062 after the known overlap
   with the earlier error study. None supplies the final new-route count.
2. Form the full residual frame and three disjoint strata before reading
   labels. Include the whole uncertainty band and all comparator positives
   for local evaluation. Randomly sample the other comparator negatives from
   the remaining frame with seed 42. For the latter stratum, select the size
   using a worst-case binary rate of 0.5 and a normal 95% precision target of
   2.5 percentage points with finite-population correction. This is a sampling
   choice, not a clinically acceptable change in sensitivity. Each stratum
   has known nonzero inclusion probabilities.
3. Obtain at most one score per selected **negative**. The exterior scores are
   collected solely to estimate the equal-budget random policy. A comparator
   positive contributes to prevalence and confusion totals without a remote
   call. Uniform random referral assigns each comparator negative probability
   `N_route / N_comparator_negatives`. Therefore the expected population
   referral count equals `N_route` even under disproportionate sampling.
   Selection uses no test labels or provider outcomes. The prepared input to
   the collector contains only the frozen negative request records and texts.
4. Report paired weighted confusion totals, sensitivity/precision/F1/F2,
   specificity, false alarms/misses/flags and calls per 1,000, and strata.
   Random results are expected totals under the frozen uniform referral rule.
   Paired finite-frame intervals use stratified bootstrap perturbations with
   finite-population correction. Repeated identifiable authors call for a
   cluster-aware analysis. The cache-only exact McNemar tests use unweighted
   deterministic pairs. DeLong is applied to the paired continuous-score
   historical audit with midrank ties. PR-AUC and AP use separate definitions.
5. Keep the protocol, sample membership, cutoff and prompt frozen when new
   evaluation begins. Attempts, failures, and uncertain responses all draw
   from the already authorized 10,000 Study 2 attempts. Use the existing
   `artifacts/jev_selective_routing_study2_validation/budget.sqlite3` ledger.
   Each required request is sealed and reserved before dispatch. Retries
   on previously attempted records are disabled.

## Source, model and scoring evidence

Use the exact version 14 release and hash from the baseline metadata. Describe
its supplied text labels and report their construction from the data card or
linked source when accessible. The possible Reddit/subreddit origin is an
unverified hypothesis. When the source distinguishes post labels from author
labels, preserve that distinction and the context presented to annotators.
No present reference measures a clinical diagnosis or future suicide outcome.

Report the confirmed API provider and alias, request/response schema, prompt,
historical timestamps, and timeout/retry configuration. Backend architecture,
parameter count, and immutable version require provider documentation. The
previous informal name “Perplexity Decider” has no identifiable checkpoint in
the project. A model comparison requires an actually accessible immutable
checkpoint, license/revision, tokenizer, hardware, dtype and settings.
No transformer weights were available locally for this revision.

The existing API export exposes a scalar rating. A qualitative alternative can
use `no_explicit_evidence`, `ambiguous_or_context_needed`, and
`explicit_current_self_report` with fixed definitions. Review perspective
(author, other person or quotation), temporality, explicit statements of intent,
negation and insufficient context. Use only a validated structured parser and
evaluate binary mappings on separate development data. A supported immutable
local model could also supply token logits for a fixed yes/no label space.
Both alternatives need empirical checks against the documented reference and
neither supplies a clinical probability. The current collection plan keeps the
verified rating contract and applies frozen calibration.

## Commands

The notebook rebuilds the historical figures, statistics and manuscript using
cached data. Run `scripts/build_cascade_revision_notebook.py`, then execute
`assets/notebooks/jev_asymmetric_cascade_revision.ipynb` with `nbclient` from
the repository root. Use the current official affiliation in the notebook
parameter once it is confirmed. To render the manuscript PDF locally:

```bash
cd research/results/jev_asymmetric_cascade_revision
pandoc manuscript_report.md --pdf-engine=xelatex -V geometry:margin=22mm -o manuscript_report.pdf
```

To prepare real evaluation, restore the exact complete original baseline cache
under an ignored path. The cache reader verifies metadata and all three score
tables. If all that is available is the original hash-matching CSV, rebuild the
centralized baseline with `jev_research.notebook_pipeline.build_baseline` and
verify it against the historical fingerprint and cached scores before using it.
Keep source files under `artifacts/` or `data/`.

Record verified dataset reference evidence (`reference_type` set to
`dataset_text_labels`, `verified`, `release`, `dataset_sha256`, `source_citation`,
`annotation_unit`, `label_construction`) and the documented applicable transfer
terms (`permitted`, `source_terms_sha256`, `interpretation`). Attach the papers
or dataset source documents to the private evidence package. Verified dataset
proxy labels suffice for this revised nonclinical question. Claims of clinical
or expert annotation still require evidence of that distinct reference.

```bash
source .venv/bin/activate
study_cache=artifacts/jev_study/baseline/b154407429c3f6d7dd39f87006eb4dd9305d283146ccf27546d57f0d27fa7fd2
python -m jev_research.cascade_validation prepare \
  --baseline-cache "$study_cache" \
  --reference-evidence artifacts/reference.json \
  --transfer-evidence artifacts/transfer_terms.json
python -m jev_research.cascade_validation status
python -m jev_research.cascade_validation collect --workers 4
python -m jev_research.cascade_validation replay
```

`collect` uses the existing authorization after the frozen inputs and usable
Metis access are present. Credentials are read from `METIS_API_KEY` in the
runtime. No credential value belongs in the protocol, notebook, logs, or Git.
Use the same ledger on resume. Replay requires the planned set of attempts to
be accounted for and checks the frozen reference digest before scoring.

Measure local CPU resources on common Validation batches from the restored
training/Validation splits. Each command runs in its own process:

```bash
python -m jev_research.cascade_benchmark --baseline-cache "$study_cache" --model word2vec-xgboost
python -m jev_research.cascade_benchmark --baseline-cache "$study_cache" --model tfidf-logistic
```

The benchmark includes raw-text preprocessing, embedding or TF-IDF, and
prediction. It reports median and p95 batch time normalized to 1,000 records
and sampled observed process RSS. Load/fit duration is a separate quantity.
Archive available `spending.elapsed_seconds` for completed API calls and
report actual collection volume separately from projected referral volume.
Compute tokens and USD per 1,000 from verified usage and prices for that
referral mix. Until those records exist, the comparison with all-referral is
in requests only (1,000 per 1,000 for all-referral).

## Feedback disposition

| Item | Evidence in the revision | Remaining empirical input |
| --- | --- | --- |
| 1. Model identity | API card identifies Metis `jev-latest`, prompt, method and historical time window | Immutable backend configuration and any alternative model checkpoint |
| 2. Comparator cutoff | Separate 0.38/0.44/0.50 history. 0.45 from original Validation Youden grid | Fine-grid optimization would require complete Validation scores |
| 3. Referral rationale | Validation-bin audit and cost-normalized development F2 sweep | New validation responses for score regions below 0.25 |
| 4. Independence | Prior test exposure documented, label-blind residual sampler and frozen manifest code | Original full scores/text and an external cohort for independent confirmation |
| 5. Paired statistics | Exact McNemar, paired DeLong for continuous scores, ROC/PR, AP, intervals and matrices | Weighted fresh results and author identifiers |
| 6. Sensitivity | Same-cutoff invariant and separate recovered positive/added false alarm counts | Fresh estimate under the documented dataset reference |
| 7. Cascade architecture | Negative-only referral, add-only decisions and exact-budget random comparator | Population replay after collection |
| 8. Language model scoring | Development-fit sigmoid and held-out calibration diagnostic. Qualitative categories specified | Provider logits or validated qualitative endpoint, if used |
| 9. Dataset labels | Source version, fingerprint and binary label mapping. Clinical meaning qualified | Primary release documentation and annotation linkage |
| 10. Local baselines | Train-only TF-IDF/logistic and end-to-end timing/RSS commands | Real training/Validation texts and fitted model reconstruction |
| 11. Writing and layout | New English manuscript PDF and aligned numerical tables | Official institutional affiliation and earlier standalone PDF if needed |
| 12. Priorities and cost | Frozen study design, exact attempt accounting, and hypothetical all-referral comparison | Usable network/credentials, complete permissible inputs, verified API token/currency usage |
