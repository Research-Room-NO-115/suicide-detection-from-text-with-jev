# Probability-routed XGBoost/Jev hybrid protocol

This study evaluates base Jev as an inference-time replacement for XGBoost on a
fixed subset of score-eligible test texts. It does not train Jev weights. The new
5,000-attempt inference budget was explicitly authorized separately from the
previous 100-attempt error-only study; the original ledger remains unchanged.

## Freeze the probability interval on validation

Use the original full-data centralized Word2Vec/XGBoost baseline, seed 42 and
threshold 0.50. Its dataset, notebook/configuration and cached predictions are
fingerprinted. Reuse the original 64%/16%/20% train/validation/test split.

1. Divide [0,1] into 20 bins of width 0.05. Use [lower,upper), including 1 in the
   final bin. Report validation text counts, error counts, error rates and Wilson
   intervals. Identify both the highest-error-rate and highest-error-count bin;
   the supported rate peak requires at least 100 validation observations.
2. The narrow peak may not supply 5,000 test cases. Search all 210 contiguous
   grid-aligned windows using **validation outcomes only**, maximizing validation
   error rate subject to support of at least
   `ceil(1.10 * 5000 * N_validation / N_test)`.
3. The 10% capacity buffer and 0.05 grid are fixed before Jev outcomes. Break ties
   by smaller interval width and then smaller lower endpoint. Test-label error
   rates are not used. A test-capacity shortage fails preparation; it does not
   silently widen the interval.

For this dataset, the peak diagnostic bin is [0.45,0.50), with 200 errors among 389
validation texts (51.41%). It supplies only 463 test texts. The feasible frozen
critical interval is [0.25,0.80), with 1,648/4,861 validation errors (33.90%) and
6,067 score-eligible test texts. Minimum validation support is 4,402.

## Freeze 5,000 label-blind test identities

Eligibility uses only the frozen score range. Deterministic hash ranking with
seed 42 selects exactly 5,000 distinct test texts. Gold labels and XGBoost
correctness are **not sampling inputs**. Correctly classified texts remain
eligible, allowing the study to measure regressions. Raw text hashes and complete
baseline-cache identity are checked before any paid request.

Public manifests expose IDs, hashes, XGBoost scores, outcome labels and provenance,
without raw texts. The cohort lock timestamp precedes the recorded inference
window. The sampling fingerprint is independent of test labels.

## Execute the operational hybrid

For the selected 5,000, use a successful Jev score at threshold 0.50. Failed,
uncertain or unattempted requests keep the XGBoost score. Every unselected test
case also keeps XGBoost. This includes 1,067 eligible interval members omitted by
the budget. Thus the full-test policy routes only the exact selected manifest;
it does not send every critical-range case to Jev.

One separate SQLite ledger caps all new attempts and unique samples at 5,000.
Reserve before network I/O, count failures/interruptions, disable retries and
redirects, and cache exact successes. Four workers use separate HTTP sessions
and a common ledger. Stop new dispatch after a provider error, finish in-flight
requests and diagnose. A deliberate resume skips successes, failures and
uncertain requests and considers only previously unattempted identities. Never
reset the ledger. The selected comparison always retains all 5,000 cases through
XGBoost fallback.

## Report both populations

Compare the same selected 5,000 texts as the primary population, then all 46,342
test texts under the implemented partial-routing policy. Report confusion counts,
accuracy, precision, recall, specificity, positive/macro F1, ROC-AUC, AP, Brier
score and log loss. Report XGBoost errors corrected, newly introduced errors,
net change, class-specific recovery/regression rates with Wilson intervals and
an exact paired McNemar test.

Required identities:

- Selected accuracy change = net additional correct / 5,000.
- Full-test accuracy change = the same net additional correct / 46,342.
- Corrected/regressed counts and exact McNemar p-values match between the two
  populations because all off-subset predictions are unchanged.
- Missing/provider-failed cases are explicit XGBoost fallbacks, not dropped cases.

## Interpretation limits

The largest observed validation error rate is empirical and selected over many
windows; its interval estimate is descriptive. The locked test comparison is the
outcome evidence, but this test split was already inspected in the original
notebook and previous study, so results are exploratory. Scores produced by
XGBoost and Jev need not be jointly calibrated; mixed-score ranking/calibration
metrics describe the implemented policy. `jev-latest` is a mutable backend alias;
request timestamps are recorded, but an immutable revision is unavailable. These
social-media labels do not establish clinical validity or deployment safety.

## Actual execution outcome

All 5,000 unique texts were attempted. There were 4,996 successful Jev responses
and four HTTP 503 failures, each retained as an XGBoost fallback. No failed
request was resent, and no uncertain request remains. Four dispatch pauses and
deliberate resumptions preserved the ledger. Worker settings were 4, 2, 2, 4, 4
across those phases; the interval, identities, model question and thresholds
never changed. The execution audit records phase counts and UTC timestamps.

The hybrid corrected 1,339 errors and introduced 607, for 732 net corrections.
Of the suicide false negatives, 538 were corrected but 511 new misses were
introduced, leaving only 27 fewer missed cases overall. False alarms fell by 705
net cases. Accuracy and F1 improved, while full-test ROC-AUC slightly declined.
These class-specific flows must accompany aggregate improvement claims.
