"""Build only the new experiment notebook; execute it with nbclient afterwards."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(dedent(source).strip()))


md("""
# Selective JEV assistance under a strict inference budget

**Executed exploratory study of cached predictions; no new paid inference.**

The question is whether JEV improves detection of positive suicide-related texts
beyond threshold tuning and simple combination rules, and how many requests those
rules would need. Accuracy gains are decomposed into missed-positive recovery,
new misses, corrected false alarms, and new false alarms. Positive means the
dataset label `suicide=1`, not a clinical diagnosis.

The historical 5,000 texts were already inspected. A deterministic 2,000/3,000
development/evaluation partition supports honest *within-study* model fitting and
threshold selection; it does **not** turn the reused test set into independent
confirmation. All policies below are exploratory. Previous notebooks and result
directories are preserved.
""")
md(r"""
### Research questions and how to read this notebook

1. **Reproduction:** Do the saved per-example records reproduce the historical
   accuracy gain, and how much of that gain comes from fewer missed positives
   rather than fewer false alarms?
2. **Sensitivity-controlled comparison:** At thresholds chosen for the same
   development recall target, does JEV reduce evaluation false alarms or increase
   precision relative to threshold-tuned XGBoost?
3. **Request efficiency:** Can a short-circuit rule retain useful corrections
   while requiring JEV for only a fraction of the historical routing population?
4. **Score quality:** Does development-only calibration improve held-out score
   losses, and does a two-model combiner add evidence beyond calibrated JEV alone?

These questions organize an exploratory reanalysis, not a prospectively
registered confirmatory trial. Neither small p-values nor a new partition name
remove the prior exposure of these texts. Both recall targets, the unsuccessful
or inconclusive alternatives, and the request trade-offs are reported together.

The notebook separates four evidence levels:

| Evidence level | What can be established here | What it cannot establish |
| --- | --- | --- |
| Saved provider outcomes | Historical statuses and exported risk scores for 5,000 selected cases | Independent authentication without the private ledger or raw provider payloads |
| Recalculated cached analysis | Metrics, transitions and alternative decisions on actual exported rows | New provider measurements or performance outside cache coverage |
| Aggregate-derived full-test results | Counts and additive losses when all unselected predictions are unchanged | Full-test rankings or globally retuned thresholds |
| Optional additional study | Tested sampling and two-phase budget-accounting software | New empirical outcomes while inference is disabled and source artifacts are missing |

Read the reproduction and evidence audit first, then the held-out policy and
matched-sensitivity comparisons. The calibration/routing analyses explain the
trade-offs; they do not override the population and independence limitations.
""")
code(r"""
%matplotlib inline
from pathlib import Path
import json
import sys
import hashlib
import subprocess
import unittest.mock as mock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'jev_research').is_dir() and (p / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT))
from jev_research.budgeted_experiments import (
    run_cached_study, policy_output, metrics, digest, file_digest, save_json, save_table,
)
from jev_research.additional_experiment import write_pending_plan

RUN_NEW_JEV_INFERENCE = False
MAX_NEW_JEV_ATTEMPTS = 500
RANDOM_SEED = 42
AUTHORIZED_ADDITIONAL_MANIFEST_SHA256 = None
RESULTS = ROOT / 'research/results/jev_probability_hybrid_budgeted_experiments'
PRIVATE = ROOT / 'artifacts/jev_probability_hybrid_budgeted_experiments'
FIGURES = RESULTS / 'figures'
for directory in [RESULTS, PRIVATE, FIGURES]:
    directory.mkdir(parents=True, exist_ok=True)

# Defense in depth: the entire default notebook rejects HTTP requests.
# The separately documented, explicit paid runner is outside this offline notebook.
assert RUN_NEW_JEV_INFERENCE is False, 'Use the explicit two-phase runner for an authorized additional study.'
http_guard = mock.patch('requests.sessions.Session.request',
                        side_effect=RuntimeError('HTTP disabled in the default offline notebook'))
http_guard.start()
original_paths = [ROOT / 'assets/notebooks/jev_probability_hybrid_5000.ipynb',
                  *sorted((ROOT / 'research/results/jev_probability_hybrid').glob('*.json')),
                  ROOT / 'assets/notebooks/jev_xgboost_error_rescue.ipynb']
original_fingerprints = {str(p.relative_to(ROOT)): file_digest(p) for p in original_paths}
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 85)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
print('Offline default: zero new paid attempts. Global optional allowance: 500, not activation.')
print('No models are retrained. No original study ledger is opened or changed.')
""")
md("""
## 1. Evidence audit and protocol

The reproducible primary evidence is the committed per-example prediction export
and routing manifest. They can be cross-checked against each other and the saved
report. The private SQLite ledger, raw provider payloads, original CSV, full
baseline prediction tables, and fitted Word2Vec/XGBoost models are absent. Public
exports cannot independently authenticate the underlying provider responses.

A version-14 Kaggle recovery attempt during repository inspection returned a
proxy **403 Forbidden for api.kaggle.com**. Native Git read access worked; the
tracked tree and available workspace contained no missing private artifacts.
This notebook performs no downloads and does not silently retrain a replacement
baseline. Full-population ranking metrics cannot be reconstructed from confusion
counts. Original dataset checksum and baseline fingerprint are retained below.

The source pipeline uses `non-suicide=0, suicide=1`, stratified 64/16/20 splits
(seed 42), raw normalized duplicate removal/conflict checks, then cleaning with
train → validation → test priority for cleaned collisions. Negations are expanded
before punctuation removal. Word2Vec vocabulary/vectors and XGBoost are fitted
on training rows only, according to source inspection. Without private tables we
cannot rerun the original raw/cleaned split-disjointness checks. Author and time
identifiers are not supplied; author/temporal leakage and author-level resampling
are unavailable.

Raw conflicting-label filtering uses labels across the dataset before splitting;
this is a global quality filter, not strictly train-only preprocessing. Learned
features still use training rows only. The risk-rating question and dataset's
binary text labels also need not represent the same clinical construct.

The fixed `jev-latest` question requests a numeric suicidal-intent **risk rating**
from 0 to 1. It is not verified token confidence or a calibrated probability.
The parser accepts finite numeric `answers.department.score` values in [0,1],
rejecting booleans. The client sends the original text without local truncation;
provider-side context limits are unknown. The alias has no immutable recorded
backend revision.
""")
code(r"""
study = run_cached_study(ROOT, RESULTS)
audit, frozen, tables = study['audit'], study['frozen'], study['tables']
all_rows, development, evaluation = study['frame'], study['development'], study['evaluation']
specs, thresholds = frozen['specifications'], frozen['thresholds']
display(pd.DataFrame(audit['sources']))
display(pd.Series({k: audit[k] for k in [
    'source_revision', 'dataset_sha256', 'baseline_fingerprint', 'selected_n',
    'successes', 'failures', 'prior_error_study_overlap_n',
    'prior_overlap_different_scores_n', 'earlier_additional_compatible_cache_n',
    'request_model', 'historical_attempts', 'historical_phase_wall_seconds',
    'new_paid_attempts']}, name='Audit'))
display(Markdown('**Exact fixed scoring question:** ' + audit['question']['instructions']))
print('Historical request dates (UTC):', audit['historical_execution_window_utc'])
print(audit['cost_and_latency'])
""")
md("""
There are **21** overlapping texts between the original error study and the
probability study; **11** have different JEV scores in the two saved runs. We
retain the probability-study response for every common-population comparison,
without cherry-picking the better run. This shows response variability; it does
not identify whether stochastic inference or a mutable backend caused it.
The 19 additional earlier responses come from a gold-label-selected error study;
pooling them would bias the present probability-selected population. They remain
available evidence and are excluded from future new-request sampling, along with
every previously selected sample. No historical failed request is resent.
""")
md("""
## 2. Reproduce the previous result on identical examples

All four failed requests stay in the denominator and keep XGBoost's score and
decision. The following metrics and transitions are recomputed from the actual
5,000 prediction records and asserted against the saved report. OR predicts
positive when either model is positive; AND requires both positives. The 50:50
mean is a fixed score fusion. The last three policies are new **calculations
from historical responses**, not new provider measurements.

**Cost columns are prospective binary-decision request counts.** Historical
acquisition cost was 5,000 attempts for this cache (plus 40 in the earlier study).
Reading the cache now costs zero new requests. OR needs JEV only for eligible
XGB-negative cases; AND needs it only for eligible XGB-positive cases. A future
system that also wants both scores for every example would lose this saving.
Binary OR/AND have no claimed probability metrics in these tables.
""")
md(r"""
### Decision rules, fallback and request accounting

For example $i$, let $x_i$ be its XGBoost score, $j_i$ its available JEV rating,
and $t$ a decision threshold. The original decision is
$b_i=\mathbb{1}[x_i\geq 0.50]$. On routed examples with successful JEV responses:

| Policy | Positive decision | When JEV is operationally needed |
| --- | --- | --- |
| XGBoost | $\mathbb{1}[x_i\geq t]$ | Never |
| Replacement | $\mathbb{1}[j_i\geq t]$ | Every routed example |
| OR | $\mathbb{1}[x_i\geq t\;\mathrm{or}\;j_i\geq t]$ | Only if $x_i<t$ |
| AND | $\mathbb{1}[x_i\geq t\;\mathrm{and}\;j_i\geq t]$ | Only if $x_i\geq t$ |
| Weighted mean | $\mathbb{1}[(1-\alpha)x_i+\alpha j_i\geq t]$ | Every routed example for $0<\alpha<1$ |

If a decision **requires** JEV and its result is failed, pending, or unavailable,
retain $b_i$; do not remove the example, fill in an invented JEV rating, or apply
a tuned cutoff to that fallback decision. A short-circuit decision needs no
fallback when XGBoost already determines the result. Unrouted examples also keep
$b_i$. These details matter when $t\ne0.50$.

For score diagnostics, an unavailable rating is represented by the original
XGBoost score. At a tuned threshold the operational hard fallback may therefore
differ from simply thresholding that diagnostic score. Operational confusion
counts always use the explicit fallback decisions. All 5,000 rows remain in
their population denominator.

The historical attempt count, prospective decision request count and new paid
attempt count answer different questions. In particular, a prospective count of
3,000 does **not** authorize 3,000 new calls: this study makes zero, and the
optional additional study has its own shared hard limit of 500 attempts.
""")
code(r"""
historical = tables['historical_selected_comparison']
display(historical[['policy', 'n', 'accuracy', 'balanced_accuracy', 'precision', 'recall',
                    'specificity', 'f1', 'f2', 'macro_f1', 'fn', 'fp', 'prospective_requests']])
display(historical[['policy', 'corrected', 'introduced', 'net_correct',
                    'fn_recovered', 'fn_introduced', 'fp_corrected', 'fp_introduced',
                    'accuracy_change_pp', 'accuracy_relative_improvement_pct']])
replacement = historical.set_index('policy').loc['JEV replacement']
assert replacement.corrected - replacement.introduced == replacement.net_correct == 732
assert replacement.fn_recovered == 538 and replacement.fn_introduced == 511
assert replacement.fp_corrected == 801 and replacement.fp_introduced == 96
assert historical.set_index('policy').loc['OR', ['fn', 'fp']].tolist() == [322, 1049]
display(Markdown(
    f"Replacement improves selected accuracy by **{replacement.accuracy_change_pp:.2f} percentage points**, "
    f"a **{replacement.accuracy_relative_improvement_pct:.2f}% relative improvement**. "
    'These are different quantities. Only 27 of the 732 net additional correct predictions '
    'come from a net reduction in missed positives; 705 come from fewer false alarms.'
))
""")
code(r"""
from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, name in zip(axes, ['XGB@0.50', 'JEV replacement', 'OR', 'Mean 50:50']):
    row = historical.set_index('policy').loc[name]
    matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
    ConfusionMatrixDisplay(matrix, display_labels=['Negative', 'Positive']).plot(
        ax=ax, colorbar=False, values_format='d', cmap='Blues')
    ax.set_title(name)
fig.suptitle('Historical selected population: the same 5,000 texts', y=1.04)
fig.tight_layout()
fig.savefig(FIGURES / 'historical_confusions.png', dpi=160, bbox_inches='tight')
plt.show()
plt.close(fig)
""")
md("""
### Full-test quantities supported by aggregates

For fixed rules on the **exact same 5,000-row manifest**, unchanged out-of-cache
confusion counts can be added back to obtain population-level counts and rates.
The next table is **derived from public full-test baseline aggregates plus exact
selected-row changes**, not recomputed from 46,342 individual predictions. The
1,067 other eligible texts remain with XGBoost, as do all texts outside the
original interval. Do not interpret this as querying every eligible example.

Brier score and log loss are additive, so they can also be derived for score
policies with the same unchanged outside scores. Full-test ROC-AUC/AP and global
threshold retuning cannot be recovered this way. The saved full-test ROC-AUC
decreases from 0.975169 to 0.974644 and log loss increases from 0.201994 to
0.205766 for the original replacement rule; the AUC figures are report-only.
""")
code(r"""
full_derived = tables['full_population_derived']
display(full_derived[['policy', 'n', 'accuracy', 'precision', 'recall', 'fn', 'fp',
                      'misses_per_1000', 'false_alarms_per_1000', 'prospective_requests',
                      'routing_fraction', 'accuracy_change_pp',
                      'accuracy_relative_improvement_pct', 'brier', 'log_loss']])
assert full_derived.roc_auc.isna().all()
assert full_derived.set_index('policy').loc['JEV replacement', ['fn', 'fp']].tolist() == [1808, 1182]
print('All rows outside the historical manifest remain unchanged. No full-test scores fabricated.')
""")
md("""
## 3. Development/evaluation separation and frozen policies

Sort by SHA-256 of seed 42, a fixed study tag, and stable sample ID; assign the
first 2,000 to development and the remaining 3,000 to evaluation. Labels,
correctness and JEV scores do not enter partition assignment. IDs, row IDs and
raw-text hashes are checked for disjointness. This does not verify semantic,
author or temporal independence.

The following choices use **development only**:

- XGBoost thresholds maximizing F1 and F2, with fixed precision/proximity/tie rules.
- A JEV-weight choice from {0.25, 0.50, 0.75}, minimizing development log loss.
- Sigmoid recalibration of clipped score logits, separately for XGB and JEV.
- A two-feature logistic combiner of the clipped XGB/JEV logits, fixed L2 C=1.

No paid hyperparameter search is used. JEV calibrators/fusion fit successful
development responses only. Every missing evaluation response keeps XGB@0.50,
including in target-recall experiments. XGB-only policies use their selected
thresholds. Unrouted examples also keep XGB@0.50. Threshold candidate values
come only from development scores; evaluation labels cannot select a threshold.
Calibration is conditional on this restricted score-range population.
""")
md(r"""
### Metric definitions and selection objectives

With positive-class counts TP, FN and negative-class counts TN, FP:

\[
\mathrm{recall}=\frac{TP}{TP+FN},\quad
\mathrm{precision}=\frac{TP}{TP+FP},\quad
\mathrm{specificity}=\frac{TN}{TN+FP},\quad
\mathrm{accuracy}=\frac{TP+TN}{N}.
\]

Balanced accuracy is the mean of recall and specificity. Positive-class F1 is
$2TP/(2TP+FP+FN)$, while F2 is $5TP/(5TP+4FN+FP)$ and therefore gives missed
positives greater weight. Macro F1 averages the two class-specific F1 scores.
Misses and false alarms per 1,000 texts are $1000FN/N$ and $1000FP/N$; the
denominator must always be identified as the selected subset or the supported
full population.

The F1/F2 XGB baselines maximize their respective development objectives. For
target recall $r\in\{0.90,0.95\}$, select a development threshold minimizing FP
subject to recall $\geq r$, with the documented deterministic tie rules. An F2
optimum is not required to satisfy a particular sensitivity target, and a target
met on development may be missed on evaluation. No evaluation-label threshold
adjustment is allowed.

Accuracy changes are reported as both percentage points and relative changes:
$100(A_{new}-A_{base})$ versus
$100(A_{new}-A_{base})/A_{base}$. Relative error reduction would be a third,
different quantity and is not substituted for either one.

ROC-AUC measures score ranking across positive/negative examples; average
precision summarizes the precision-recall staircase. Neither is determined by a
single confusion matrix. Brier score and log loss are defined below and penalize
score quality rather than only thresholded decisions. They are diagnostics here,
not proof that raw JEV ratings are calibrated event probabilities.
""")
code(r"""
display(all_rows.groupby('partition').agg(n=('sample_id', 'size'),
        positives=('y_true', 'sum'), successes=('jev_status', lambda x: int(x.eq('succeeded').sum()))))
assert len(development) == 2000 and len(evaluation) == 3000
for column in ['sample_id', 'row_id', 'text_sha256']:
    assert not set(development[column]) & set(evaluation[column])
display(pd.DataFrame([{'policy': name, 'threshold': thresholds[name], **spec}
                      for name, spec in specs.items()]))
display(pd.DataFrame(frozen['selection']['weight_trials']))
print('Frozen policy digest:', digest(frozen))
print('Evaluation remains exploratory despite this disjoint fitting partition.')
""")
md("""
## 4. Policy comparison on the common exploratory evaluation population

These 3,000 examples were not used in the new fits/threshold choices. All
operational metrics include the historical failures. The F2-selected baseline
may select an almost-always-positive rule on this difficult score-range subset;
it is retained honestly. F1 and target-recall baselines provide additional
comparisons. Threshold and calibration changes do not incur provider requests.
""")
code(r"""
comparison = tables['evaluation_comparison']
display(comparison[['policy', 'threshold', 'accuracy', 'balanced_accuracy', 'precision',
                    'recall', 'specificity', 'f1', 'f2', 'macro_f1', 'fn', 'fp',
                    'prospective_requests', 'required_unavailable_fallbacks']])
display(comparison[['policy', 'roc_auc', 'average_precision', 'brier', 'log_loss']])
display(comparison[['policy', 'fn_recovered', 'fn_introduced', 'fp_corrected',
                    'fp_introduced', 'net_correct', 'routing_fraction']])
print('Historical full cache: 5,000 attempts. New paid attempts for every policy above: 0.')
""")
md("""
## 5. Compare sensitivity using development targets of 0.90 and 0.95

For each score rule select the development threshold with the fewest false
alarms subject to the recall target; deterministic ties favor precision and the
higher cutoff. Exact >= ties are respected. A target that cannot be reached
would be marked unachievable. No threshold is moved after seeing evaluation.

**Equal development targets do not guarantee equal evaluation recall.** Compare
achieved recall and its paired difference, not just the target name. These
targets refer to the selected-population development subset; they are not a
claim of full-population 90%/95% sensitivity. Sigmoid XGB recalibration preserves
ranking and therefore cannot improve the attainable threshold trade-off here.
OR/AND short-circuit request counts depend on the chosen cutoff. AND at a very
low threshold can require JEV for almost every cached example, losing the
request saving it has at 0.50.
""")
code(r"""
matched = tables['matched_recall_evaluation']
display(matched[['policy', 'target', 'threshold', 'development_recall', 'development_precision',
                 'recall', 'precision', 'specificity', 'fn', 'fp', 'prospective_requests']])
matched_ci = tables['matched_recall_uncertainty']
display(matched_ci.loc[matched_ci.policy.isin(['JEV replacement', 'OR', 'Logistic fusion']),
    ['policy', 'development_target_recall', 'delta_recall', 'delta_recall_ci_low',
     'delta_recall_ci_high', 'delta_precision', 'delta_precision_ci_low',
     'delta_precision_ci_high', 'net_fp_reduction', 'mcnemar_sensitivity_p']])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
names = ['XGB threshold (target)', 'JEV replacement', 'OR', 'Mean 50:50', 'Logistic fusion']
for ax, target in zip(axes, [.90, .95]):
    rows = matched.loc[matched.target.eq(target)].set_index('policy').loc[names]
    for name, row in rows.iterrows():
        ax.scatter(row.recall, row.precision, s=50)
        ax.annotate(name, (row.recall, row.precision), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Achieved evaluation positive recall', ylabel='Evaluation positive precision',
           title=f'Development recall target {target:.2f}')
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIGURES / 'matched_sensitivity.png', dpi=170, bbox_inches='tight')
plt.show()
plt.close(fig)
""")
md("""
## 6. Routing where JEV adds value, not just where XGB errs

The small analysis-prespecified interval set is the original [0.25,0.80), a
narrow [0.40,0.60), a below-threshold [0.25,0.50), and an above-threshold
[0.50,0.80) interval. Each is combined with replacement, OR, or a fixed mean;
XGB-only is also a candidate. These rules use actual cached responses inside
their interval and XGB fallback everywhere else. They never extrapolate JEV
predictions to the unqueried 1,067 eligible texts or outside original coverage.

Select a rule on development for each hypothetical FN:FP loss ratio 1, 5, and
10; ties favor fewer requests. These are sensitivity-analysis scenarios, not
clinical utilities or currency costs. Evaluation rows for all candidates are
descriptive; the selected-rule table is keyed only to development decisions.
The 0.05-wide bins below are descriptive and are not used to optimize an
elaborate sparse-bin policy.
""")
md(r"""
For a hypothetical false-negative cost multiplier $c$, development selection
minimizes $L_c=c\,FN+FP$. Dividing by the common population size gives loss per
text. This multiplier expresses a scenario about the *relative* consequences of
the two error types; it is neither provider pricing nor an independently
validated clinical preference.

Net additional correct predictions per required request are
$(\mathrm{corrected}-\mathrm{introduced})/N_{requests}$. A region with a high
baseline error rate can still have poor JEV value if corrections are offset by
regressions. This quantity is descriptive for the observed routing population;
it does not estimate the unobserved benefit of requests outside cached coverage.
When no JEV requests are required, this ratio is undefined rather than an
infinite return on investment.
""")
code(r"""
display(tables['routing_development'][['policy', 'baseline_error_rate', 'corrected', 'introduced',
        'net_fn_reduction', 'net_fp_reduction', 'prospective_requests', 'net_correct_per_request',
        'loss_fn_cost_1', 'loss_fn_cost_5', 'loss_fn_cost_10']])
display(tables['routing_selected_evaluation'][['fn_to_fp_cost_ratio', 'policy',
        'recall', 'precision', 'fn', 'fp', 'prospective_requests', 'net_correct_per_request']])
display(tables['score_bins_descriptive'])

bins = tables['score_bins_descriptive'].query("partition == 'evaluation'")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
centers = (bins.lower + bins.upper)/2
axes[0].plot(centers, bins.baseline_error_rate, marker='o', label='XGB error rate')
axes[0].plot(centers, bins.net_correct_per_request, marker='s', label='Replacement net gain/request')
axes[0].axhline(0, color='gray', lw=.8)
axes[0].set(xlabel='XGB score bin center (width 0.05)', ylabel='Rate', title='Baseline errors versus actual net JEV value')
axes[0].legend(fontsize=8)
axes[1].bar(centers-.007, bins.net_fn_reduction, width=.014, label='Net missed-positive reduction')
axes[1].bar(centers+.007, bins.net_fp_reduction, width=.014, label='Net false-alarm reduction')
axes[1].axhline(0, color='gray', lw=.8)
axes[1].set(xlabel='XGB score bin center', ylabel='Net count', title='Exploratory evaluation; all cached bin examples')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / 'routing_value.png', dpi=170)
plt.show()
plt.close(fig)
""")
md("""
## 7. Calibration, simple fusion and ranking boundaries

Reliability diagrams use ten fixed equal-width bins on evaluation only; sparse
bins are shown rather than merged after inspecting outcomes. Brier/log loss are
score-quality diagnostics conditional on coverage, not clinical calibration.
The original raw JEV replacement can improve accuracy while worsening log loss,
because confident mistakes are heavily penalized. Sigmoid calibration and
regularized fusion must earn any gains on the held-out analysis partition.

The boundary diagnostic examines actually observed adjacent 0.05-wide score
bands. We count cross-band ranking inversions after replacing one side's scores.
At the *original* 0.25/0.80 routing boundaries the other side is unobserved, so
inversion rates there are **not estimable**. At 0.40/0.60 both sides exist in the
cache, allowing local diagnostics for the narrower candidate. Such diagnostics
do not recover full-test AUC or prove that calibration fixes every ranking change.
""")
md(r"""
### What is fitted, and what is measured?

For a bounded score $s$, define
$z(s)=\log(\tilde{s}/(1-\tilde{s}))$ with
$\tilde{s}=\mathrm{clip}(s,10^{-6},1-10^{-6})$, and
$\sigma(v)=1/(1+e^{-v})$. The fitted sigmoid is
$q=\sigma(a\,z(s)+b)$; the two-model fusion is
$q=\sigma(w_xz(x)+w_jz(j)+b)$. Coefficients and intercepts are learned on
development labels only using logistic regression with fixed L2 regularization
(`C=1`, inverse regularization strength). The coefficient table and frozen JSON
record the actual fitted parameters. There is no evaluation-driven search for C,
new prompt, or model family.

For diagnostic scores $q_i$ and labels $y_i$, lower values are better for

\[
\mathrm{Brier}=\frac{1}{N}\sum_i(q_i-y_i)^2,\qquad
\mathrm{LogLoss}=-\frac{1}{N}\sum_i\left[y_i\log q_i+(1-y_i)\log(1-q_i)\right].
\]

Log loss clips exact zero/one scores at floating-point machine precision when
evaluating, as in scikit-learn. This is distinct from the $10^{-6}$ clipping
used to construct finite model features. Raw JEV confidence-like extremes can
therefore incur a large log-loss penalty when wrong. A lower fitted loss on
development is not evidence of better calibration; the evaluation losses,
paired intervals and reliability diagram provide the relevant checks here.

A selected raw JEV threshold such as 0.03 is an empirical cutoff for this
restricted dataset and prompt. It must not be read as a calibrated 3% clinical
risk or a recommended intervention threshold. Separately calibrated scores may
have different numerical cutoffs while yielding the same ranking/decisions.
""")
code(r"""
reliability = tables['reliability']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name in ['XGB@0.50', 'JEV replacement', 'Mean 50:50', 'JEV sigmoid', 'Logistic fusion']:
    rows = reliability.loc[reliability.policy.eq(name)]
    axes[0].plot(rows.mean_score, rows.observed_positive_fraction, marker='o', label=name)
axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect reliability')
axes[0].set(xlabel='Mean risk score', ylabel='Observed positive fraction', title='Exploratory held-out reliability')
axes[0].legend(fontsize=8)
score_names = ['XGB@0.50', 'JEV replacement', 'Mean 50:50', 'JEV sigmoid', 'Logistic fusion']
score_rows = comparison.set_index('policy').loc[score_names]
positions = np.arange(len(score_names))
axes[1].bar(positions-.18, score_rows.brier, width=.36, label='Brier')
axes[1].bar(positions+.18, score_rows.log_loss, width=.36, label='Log loss')
axes[1].set_xticks(positions, score_names, rotation=35, ha='right')
axes[1].set(ylabel='Loss (lower is better)', title='Default policy score diagnostics')
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / 'calibration.png', dpi=170)
plt.show()
plt.close(fig)
display(tables['score_loss_uncertainty'])
loss_ci = tables['score_loss_uncertainty']
fusion_ci = loss_ci.loc[loss_ci.policy.eq('Logistic fusion') &
                       loss_ci.reference.eq('JEV sigmoid') & loss_ci.loss.eq('log_loss')].iloc[0]
display(Markdown(
    f"Sigmoid JEV log loss is **{comparison.set_index('policy').loc['JEV sigmoid', 'log_loss']:.4f}**, "
    f"versus **{comparison.set_index('policy').loc['JEV replacement', 'log_loss']:.4f}** before calibration. "
    f"Adding XGB through logistic fusion gives only a **{fusion_ci.difference:+.4f}** log-loss difference "
    f"relative to calibrated JEV (paired 95% CI **{fusion_ci.ci_low:+.4f} to {fusion_ci.ci_high:+.4f}**). "
    'That interval includes zero: the incremental fusion gain is inconclusive. '
    'XGB-only sigmoid calibration also shows no clear proper-score improvement.'
))
display(tables['boundary_ranking'])
""")
md("""
## 8. Paired uncertainty and prespecified analysis comparisons

Use 4,000 paired empirical row-bootstrap resamples (seed 42) for differences in
accuracy, sensitivity, specificity, precision and F2. Multinomial draws of the
eight (truth, baseline prediction, policy prediction) joint cells are exactly
equivalent to row resampling for these metrics. Proper-score loss differences
use 2,000 paired row-bootstrap resamples. Exact two-sided McNemar tests compare
overall correctness; a separate exact paired test on true-positive-class
examples compares recovered versus newly missed positives.

The comparison families are fixed rules versus XGB@0.50, development-F1 XGB and
development-F2 XGB; target-recall comparisons use XGB at the same development
target. Holm adjustment is reported separately for accuracy and sensitivity
within the fixed-policy family and the target-recall family. These are
analysis-prespecified comparisons, not a prospective preregistration. Reported
percentile intervals are pointwise and **conditional on the fitted/frozen policy**:
they omit development-selection variability and do not correct earlier reuse of
the test population. They are not evidence of independent replication. Author
clusters cannot be resampled without author IDs. Small bins/strata have limited
precision; semantic dependence may make row intervals optimistic.
""")
code(r"""
uncertainty = tables['paired_uncertainty']
display(uncertainty.loc[uncertainty.reference.eq('XGB@0.50'),
    ['policy', 'delta_accuracy', 'delta_accuracy_ci_low', 'delta_accuracy_ci_high',
     'delta_recall', 'delta_recall_ci_low', 'delta_recall_ci_high',
     'mcnemar_accuracy_p', 'mcnemar_sensitivity_p', 'mcnemar_sensitivity_holm_p']])
display(uncertainty.loc[uncertainty.reference.eq('XGB threshold (dev F1)'),
    ['policy', 'delta_recall', 'delta_recall_ci_low', 'delta_recall_ci_high',
     'delta_precision', 'delta_precision_ci_low', 'delta_precision_ci_high']])
print('All comparison rows, both p-value adjustments and paired intervals are saved as JSON.')
""")
md("""
## 9. Reproducible disagreement review

The four measured transition categories are available; original text is not.
We freeze eight hash-ranked IDs per category for reproducible review. No raw
text or inferred semantic annotations are published. There is no basis to claim
that negation, quotation, third-person reference, temporal context, ambiguous
labels or truncation explains particular errors from scores alone.

Once matching text is supplied, review these IDs using the rubric below, with
brief redacted excerpts only when necessary. Distinguish an informal single
reviewer's observations from independent blinded annotations. Check original
versus cleaned negation and retained temporal context, and compare original
character length with documented provider limits before attributing truncation.
""")
code(r"""
review = tables['error_review_manifest']
display(review[['transition', 'sample_id', 'row_id', 'p_xgb', 'jev_score', 'annotation_status']])
rubric = {'status': 'pending_original_text', 'sampling_seed': RANDOM_SEED,
          'examples_per_transition': 8,
          'fields_to_annotate': ['negation', 'quotation_or_hypothetical', 'third_person_reference',
                                 'past_vs_current_intent', 'ambiguous_gold_label',
                                 'original_length', 'verified_provider_truncation',
                                 'brief_redacted_excerpt', 'reviewer_id'],
          'claims': 'No semantic category counts or independent annotations have been measured.'}
save_json(RESULTS / 'error_review_protocol.json', rubric)
display(pd.Series(rubric, name='Review protocol'))
""")
md("""
## 10. Optional additional experiment — outcomes pending

The cached analyses leave questions about new examples and uncached score
regions. The additional design is capped at **500 attempts for the entire
study**, including failures, interruptions and development calls. Default Run
All cannot dispatch requests. Allowance does not constitute activation.

Use 200 original-validation examples for development and 300 original-test
examples for evaluation, with the score strata and quotas below. Hash selection
uses scores and stable IDs only. Exclude the union of both previous studies'
selected populations, including failed and error-study tuning samples. Record
source split, stratum counts, quota/count inclusion probabilities, and overlap
flags. No labels/correctness enter sampling. The older baseline test population
was already inspected, so this design remains exploratory; independent future
data would be stronger.

**The actual 500-row manifest is pending** because the full original validation/
test score tables and matching texts are absent. Fabricating IDs or selecting
only the four failures would not implement this design. The generator is tested
on synthetic frames; it refuses undersized strata. When originals are supplied,
it can freeze concrete counts and IDs without model retraining.

The runner reserves a request and any verified upper-bound monetary charge in
one SQLite transaction **before dispatch**. Duplicate successes are free cache
reads; failed and uncertain requests are never automatically replayed. The
default fixed prompt/model is reused; the cap cannot be raised on resumption.
Evaluation requires completed development and frozen policy decisions, whose
digest is sealed into the ledger before evaluation dispatch. Stop on provider
failure and diagnose before deliberate resumption.

No pricing or currency is available. A monetary cap, if supplied later, requires
an externally verified per-attempt maximum price and currency; otherwise the
runner refuses inference. Failed/uncertain maximum-cost reservations are kept.
Reserved upper bounds are **not measured spending**. Per-request completed
elapsed times are recorded, and credentials/payloads are not exported.
""")
code(r"""
pending_plan = write_pending_plan(RESULTS)
display(pd.DataFrame(pending_plan['quotas']).T)
display(pd.Series({k: pending_plan[k] for k in [
    'status', 'manifest_status', 'ledger', 'authorization', 'money', 'new_paid_attempts']}, name='Additional experiment'))
print('Executable helpers: prepare_additional_plan, collect_additional_phase,')
print('freeze_additional_policies, analyze_additional (see research/budgeted_experiments_protocol.md).')
print('No paid runner is called by this notebook.')
""")
md("""
## 11. Verification, including mocked request accounting

Run the repository test suite with mocked inference, not provider credentials.
It checks source-label mapping, sample uniqueness/disjointness, cache integrity,
metrics/transitions, threshold selection, fallback, score-fusion serialization,
and the new financial controls. The synthetic two-phase integration test performs
200 + 300 **mock** requests, freezes decisions between phases, rejects a changed
decision digest, and checks that resumption makes zero new mock requests.
Other tests cover concurrent cap enforcement, absent activation, missing
credentials, HTTP/transport failures, uncertain interruption and monetary
reservations across restart. These synthetic requests are not paid attempts.
""")
code(r"""
completed = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           cwd=ROOT, capture_output=True, text=True, timeout=180)
log = completed.stdout + completed.stderr
(PRIVATE / 'verification.log').write_text(log)
print('\n'.join(log.strip().splitlines()[-5:]))
assert completed.returncode == 0, log[-6000:]
assert 'Ran 101 tests' in log, 'Verify that the expected tests actually executed.'
assert {str(p.relative_to(ROOT)): file_digest(p) for p in original_paths} == original_fingerprints
assert not (PRIVATE / 'budget.sqlite3').exists(), 'Default offline analysis should not create a paid-study ledger.'
verification = {'suite_exit_code': completed.returncode, 'tests_passed': 101,
                'new_paid_attempts': 0, 'default_http_blocked': True,
                'old_artifact_fingerprints_unchanged': True,
                'mock_integration_attempts': 500,
                'optional_additional_real_attempts': 0,
                'optional_additional_outcomes': 'pending'}
save_json(RESULTS / 'verification.json', verification)
save_json(RESULTS / 'original_artifact_fingerprints.json', original_fingerprints)
display(pd.Series(verification, name='Verification'))
""")
md("""
## 12. Interpretation and limitations

The original replacement accuracy improvement is driven predominantly by fewer
false alarms. OR guarantees that a baseline-positive decision is not overridden
negative, so it recovers missed positives without introducing new misses; this
comes with new false alarms. Its request saving is operationally real for binary
decisions, but it is not a claim that every matched-sensitivity trade-off wins.

Threshold tuning is essential: calibrated/fused scores can improve score losses
without proving they dominate a tuned single-model decision rule. The tables
report negative results as well, including the raw replacement's weak sensitivity
gain and the near-redundancy of XGB-only sigmoid calibration at matched targets.
The strongest comparison below uses thresholds fixed at the same development
recall target and reports actual achieved sensitivity, precision and false alarms.

All conclusions are conditional on the selected historical score range. Missing
full predictions prevent global threshold-matched comparisons and new full-score
ranking analysis. Missing original text prevents semantic error annotations.
Missing ledger/payloads prevent independent provider-cache authentication, pricing
reconstruction and per-request historical latency analysis. Unknown authors,
temporal dependence, prior test reuse, mutable `jev-latest` and observed repeat
score variability limit reproducibility claims. No clinical/deployment validity,
independent replication or actual JEV weight fine-tuning is established.
""")
md(r"""
### What would constitute stronger evidence?

The present evidence supports a conditional statement about these historical
texts: some JEV decision rules improve the precision/sensitivity trade-off
relative to simply moving XGBoost's cutoff. It does not establish that JEV
increases recall on a new population, or that the most flexible combiner is
preferable to a calibrated single JEV score. The latter question must consider
the paired loss intervals, not only which point estimate is smallest.

A stronger follow-up would freeze a prompt/model revision and candidate policy
set, obtain an independently collected evaluation population with author/time
metadata, and retain full baseline scores and the request ledger. Development
data would determine thresholds, any calibration and sampling quotas before
evaluation outcomes were opened. Record inclusion probabilities, failures,
usage/pricing and latency so that sensitivity, false alarms and costs can be
compared on the same intended population. An old test example excluded from the
5,000-row manifest is not automatically an independent evaluation example.

The optional 500-attempt design is a practical intermediate step rather than a
replacement for that independent study. Its limited controls can probe cache
coverage and its strata can describe local behavior, but unweighted metrics on
oversampled difficult examples must not be presented as full-dataset results.
""")
code(r"""
target95 = matched.loc[matched.target.eq(.95)].set_index('policy')
x, j = target95.loc['XGB threshold (target)'], target95.loc['JEV replacement']
ci = matched_ci.loc[matched_ci.policy.eq('JEV replacement') &
                    matched_ci.development_target_recall.eq(.95)].iloc[0]
summary = (
    f'On the 3,000-row exploratory evaluation subset, development-target-0.95 '
    f'XGB has recall {x.recall:.2%}, precision {x.precision:.2%}, {int(x.fn)} misses '
    f'and {int(x.fp)} false alarms. JEV at its development-selected threshold has '
    f'recall {j.recall:.2%}, precision {j.precision:.2%}, {int(j.fn)} misses and '
    f'{int(j.fp)} false alarms. This is {int(x.fp-j.fp)} fewer false alarms and '
    f'{100*(j.precision-x.precision):.2f} percentage points higher precision. '
    f'The recall difference is {100*ci.delta_recall:+.2f} percentage points '
    f'(paired conditional 95% CI {100*ci.delta_recall_ci_low:+.2f} to '
    f'{100*ci.delta_recall_ci_high:+.2f}); a sensitivity improvement is not established. '
    f'The JEV rule would require {int(j.prospective_requests)} requests for these '
    f'3,000 cases versus zero for XGB. All analyses reused the existing cache: '
    f'new paid attempts = 0. The lower-cost OR@0.50 rule requires only 2,184 of '
    f'the historical 5,000 requests, with 322 misses and 1,049 false alarms on '
    f'that full selected population. The additional 500-attempt experiment is pending '
    f'original full baseline artifacts and explicit activation.'
)
display(Markdown(summary))
save_json(RESULTS / 'summary.json', {'finding': summary, 'new_paid_attempts': 0,
                                  'independent_confirmation': False,
                                  'matched_comparison_population_n': 3000,
                                  'optional_additional_status': pending_plan['status']})
display(Markdown('### Main comparison at the 0.95 development recall target'))
display(target95.loc[['XGB threshold (target)', 'JEV replacement', 'OR', 'Logistic fusion'],
                     ['threshold', 'recall', 'precision', 'fn', 'fp', 'prospective_requests']])
display(Markdown(
    '**Practical interpretation:** Retain threshold-tuned XGBoost as the zero-request baseline. '
    'Use OR as a transparent candidate when preventing new baseline-positive misses and reducing '
    'request volume matter, while accepting its false-alarm trade-off. Investigate calibrated '
    'JEV for a better sensitivity/precision operating point; the present evidence does not '
    'establish an incremental advantage for logistic fusion over calibrated JEV. '
    'Freeze these candidates before collecting any new evaluation outcomes. '
    'No rule is approved for clinical deployment by this analysis.'
))
(RESULTS / 'report.md').write_text('# Cached budgeted JEV experiments\n\n' + summary +
    '\n\nSee the executed notebook for full policy, routing, calibration, paired uncertainty '
    'and evidence limitations. All tables are saved alongside this report.\n')
http_guard.stop()
print('Completed offline notebook; old artifacts preserved; zero new paid attempts.')
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
})
target = ROOT / 'assets/notebooks/jev_probability_hybrid_budgeted_experiments.ipynb'
nbf.write(notebook, target)
print(target)
