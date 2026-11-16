# Selective JEV assistance for suicide-related text flags: development evidence and a validation design

**Manuscript draft dated 2026-11-16. Study 2 results are pending, not a completed validation study.**

## Structured abstract

**Objective.** Evaluate whether a selective external-model policy reduces false-positive flags per 1,000 posts relative to threshold-tuned XGBoost, while estimating sensitivity against a construct-compatible human reference. Preservation requires a defensible margin and confidence bound.

**Methods.** We inspected the two executed notebooks, source pipeline, historical 5,000-case prediction export and earlier error-only cohort. We reproduced paired exploratory comparisons, audited repeated scores, specified a stratified common-sample validation design, and tested atomic collection accounting and weighted paired policy replay. The owner supplied IEEE document 9591887 as the proposed primary annotation source. Its annotation unit, specialist qualifications and record linkage to Kaggle version 14 could not be verified in this environment.

**Results.** In the reused 3,000-post exploratory subset at the 0.90 development recall target, XGBoost and JEV each missed 181 positive posts; false alarms were 1,010 and 410. At the 0.95 target, misses were 74 and 64 and false alarms 1,193 and 625. The paired sensitivity difference at the latter target was +0.62 percentage points (95% conditional row-bootstrap interval -0.80 to +1.94), excluding neither a small loss nor a gain. Of 21 repeated, error-enriched texts, 11 had different scores but only 1 changed a decision at 0.03 (4.76%; Wilson 95% interval 0.85%–22.67%); none changed at 0.08 or 0.50 (upper Wilson limit 15.46%). Study 2 has no collected posts, estimated primary effect, equal-cost random comparison or new expert judgments. Actual additional live attempts: 0; unused authorized allowance: 10000.

**Conclusions.** Cached evidence supports fewer false alarms at similar achieved sensitivity on previously inspected texts. It does not establish expert-reference validity, clinical sensitivity preservation, independent replication or superiority of selective over budget-matched random routing. Those claims require the blocked source verification and fresh collection.

## Methods

### Reference construct and provenance

Primary source supplied by the owner: https://ieeexplore.ieee.org/document/9591887. Article metadata, full text and supplements were not retrievable because the enforced network policy returned proxy 403 responses. No title, DOI, rater count, agreement statistic, ethics approval or license is invented. The owner did not know the paper-to-CSV mapping. The repository identifies Kaggle `nikhileswarkomati/suicide-watch` version 14 and SHA-256 `1ec16b065a2fae36488de16e8abfc326586a576a25901e09467223e545ddd295`, but this is not evidence of specialist labeling. Matched/unmatched source-record counts are not estimable without source records. The provenance table separates missing evidence from negative evidence.

The current model input is a single exported text record. In descriptions of the intended review task, "post" operationally denotes that record; equivalence to a single original source post remains unverified. If the primary paper labels authors after reviewing multiple posts, assigning an author label to each post changes the target and context; even an exact identity join would not establish post-level expert assessment. No future suicide outcome or clinical diagnosis is observed here. Verified existing specialist labels would be retained as the primary reference; bulk reannotation is not proposed as a publication prerequisite.

### Development and primary candidate

All 5,000 historical probability-selected cases are available for development; they are not independent evaluation. Earlier known-error selections remain separate. A conditional candidate uses routing [0.25,0.80), XGBoost cutoff 0.295507907867432, JEV cutoff 0.03, no primary calibration, the original fixed risk-rating question and mutable alias `jev-latest`. An unavailable required response retains the tuned comparator decision. This differs explicitly from Study 1's XGB@0.50 fallback. The development recall target refers to the historical score-range population, not guaranteed global recall. These parameters are a dated candidate, **not a collection-frozen evaluation protocol**, until reference compatibility, transfer terms and sampling data are verified.

The target review task is hypothetical prioritization of posts for qualified human assessment. A flag would request contextual review; it is not a diagnosis, evidence of current intent, a verified future event or an implemented intervention. No acceptable clinical noninferiority margin has been supplied or verified from literature. Sensitivity differences and bounds will be reported; illustrative 1/2/5-percentage-point margins cannot supply the primary preservation claim.

### Sampling, budget and independence

The preferred frame is new authors or a later, compatible expert-labeled collection. If unavailable, use unqueried original test posts after all prior training/development/JEV/error-study and duplicate exclusions. Author/near-duplicate checks require matching source metadata. Aggregate arithmetic leaves a potential 41,323-post old-test frame and 1,062 original-interval posts before further exclusions, radically reducing the residual referral fraction to about 2.57%. These are planning counts, not a recovered sampling frame. Prior use of test labels in baseline aggregate reports remains disclosed.

The tested fallback design uses disjoint outside-low, inside-below-comparator, inside-above-comparator and outside-high score strata, known positive inclusion probabilities, a census of the remaining difficult route where feasible, and probability sampling of both exterior strata. Selection uses scores/identities only. Historical estimates support routed-population precision scenarios; unobserved exterior disagreements prevent a defensible final random-comparison precision calculation. Hence no final sample size or 10,000-row manifest is fabricated. The ceiling is 10,000 cumulative attempts, not a quota. No paid repeats are currently planned because the existing repeat observations should be used first.

### Comparators and statistical analysis

One response per sampled fresh post supports replay of the primary selective policy, tuned XGBoost, one-sided OR and budget-matched random routing. The random policy uses the same JEV cutoff and fallback. Its marginal referral probability equals the **target-frame** selective referral fraction, not the oversampled subset fraction. Expected confusion totals over a uniform fixed-size random referral set identify the expected policy result; ratios are computed from expected totals and are not the average of randomized ratios. A local word/character TF-IDF logistic model is implemented, but no real-data fit was possible without upstream texts. Calibration/fusion remain secondary.

Inverse-inclusion weighted paired totals provide prevalence, sensitivity, precision, specificity, false alarms, misses, flags and requests per 1,000, with transition decomposition and benefit per 100 requests. Stratified paired bootstrap perturbations retain fixed stratum sizes and population weights and apply finite-frame correction, including zero finite-frame sampling variance in a census stratum. These intervals condition on the observed reference labels, provider responses and fixed policy. They do not quantify future-population generalization, annotation error, policy-selection uncertainty or provider variability. Repeated known authors trigger refusal of an unsupported row analysis until an appropriate cluster design is supplied. Ordinary McNemar is not applied to weighted or expected-random confusion totals.

### Focused human review

The package prepares a probability sample of 100 historical cache texts and an enriched sample of 40 transitions, with any overlap disclosed. The probability sample can audit this cache's label fidelity; enriched frequencies cannot estimate prevalence. Model predictions, labels and transition names are omitted from private blinded forms. A provisional descriptive guide covers perspective, temporal context, ideation versus expressed intent, quotation, negation, help-seeking and insufficient evidence/context. Source-paper definitions must ground final coding once retrieved. Qualified reviewers, independent coding, adjudication and scale-appropriate agreement are pending. No language model has generated expert judgments.

## Results

The historical reference checks and repeat audit above are the measured results available here. `study2_main_population_results.json` and `study2_routing_stratum_results.json` explicitly mark fresh estimates **NOT COLLECTED**, rather than filling them with historical results or zeros. No Study 2 sensitivity-preservation criterion has been met or adjudicated, and selective routing versus equal-cost random remains unestimated. The post-flow figure describes these exclusions and blockers. The performance figure is explicitly labeled Study 1, so its favorable false-alarm result cannot be mistaken for replication. The separate 10,000-attempt collection client has only been exercised with mock transport.

## Psychology-focused discussion

Fewer false-positive flags may reduce unnecessary review, but a small or uncertain change in sensitivity must be carried alongside that benefit. An extremely low raw score cutoff such as 0.03 does not imply 3% individual suicide risk. The distinction between ideation, discussion, expressed intent and human-assessed risk is central: improvements against inherited subreddit/category/author labels would answer a different question from improvements against verified post-level specialist judgments. The current documentation does not resolve which construct the reference measures.

Model disagreement can motivate focused contextual review without replacing the existing reference labels. Repeated-score differences should be judged at actual operating points: the 11 numerical discrepancies overstate observed binary instability at 0.08/0.50, but zero changes among 21 nonrepresentative repeats is weak evidence of global stability. A negative fresh routing result, meaningful sensitivity loss or no selective advantage over random would be reported directly. The modest prior incremental fusion benefit does not justify centering complex fusion without new evidence.

## Ethics, privacy and reporting

Source-study approval/exemption, consent, dataset access/license, current-study review and permission to transfer text to JEV remain unverified. A source approval would not automatically cover the present study. No assertion of ethical approval, licensed transfer or clinical readiness is made. Sensitive text, author identifiers, credentials and raw provider payloads stay outside public artifacts. Frozen hashes and aggregate/public-safe manifests support reproducibility. TRIPOD+AI domains relevant to data source, target, reference, validation, missingness, calibration and uncertainty are assessed in the reviewer table; the package does not claim completed clinical-prediction reporting or validation. There is no measured future behavior, diagnosis or intervention benefit.

## Limitations and data/code availability

Required source-paper access, record mapping, complete raw score/text frames, source terms and live authentication are missing. Source annotation unit and expert coverage cannot be verified; no independent Study 2 population or outcomes have been collected. Historical confidence intervals condition on frozen policies and row-level sampling and do not remove prior test reuse. `jev-latest` has no immutable backend revision in the exports. Actual currency cost and historical per-request usage are unavailable.

Code, executed notebook, dated design draft/digest, provenance, historical checks, review sampling and reviewer-readiness table are supplied in this repository. Original private datasets/models/ledgers and new raw text are not redistributed. Collection and fresh offline replay commands are documented in `research/study2_collection_and_reporting.md`. Notebook reruns make zero paid calls. This is a reproducible development-and-protocol package, not a completed expert-reference validation manuscript.
