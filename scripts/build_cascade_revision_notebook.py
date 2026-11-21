"""Build the English notebook for the supervisor's asymmetric-cascade revision."""

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))


markdown('''
# Asymmetric cascade for suicide-related text classification

This notebook implements the supervisor revision. The primary question concerns
additional positive text labels detected at a fixed request budget, with the
accompanying false alarms. Word2Vec–XGBoost supplies the first decision. The
Metis Jev API can promote selected comparator negatives to positive.

The archived data support exploratory replay and method development. Fresh
evaluation, a source-verified label definition and provider access remain
pending in the delivered snapshot. A separate collection command uses the
already authorized 10,000-attempt Study 2 ledger.
''')

code('''
from pathlib import Path
import json
import sys

from IPython.display import Image, display
import pandas as pd

root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'jev_research').is_dir())
sys.path.insert(0, str(root))
from jev_research.cascade_report import render_revision

official_affiliation = None
analysis, fresh, output = render_revision(root, affiliation=official_affiliation)
display(pd.Series(fresh, name='Fresh Study 2 status'))
''')

markdown('''
## 1. Identify the actual model and reference

`jev-latest` is the exact request alias for the recorded Metis Jev API. The API
record below supplies the prompt, response parser and historical timing. The
provider's backend architecture, immutable version and parameter count are
unavailable. The earlier name “Perplexity Decider” identifies no verified
checkpoint in this project and has been removed from the model comparison.

One input is one exported text record. The known mapping is the dataset's
`suicide` and `non-suicide` classes. Reddit/subreddit origin and the annotation
linkage to IEEE document 9591887 require source verification. A subreddit-derived
label would reflect a proxy for text content. Clinical diagnosis and individual
future outcomes were outside the observed reference.
''')

code('''
model = json.loads((output / 'model_and_api.json').read_text())
display(pd.Series(model, name='Metis request identity'))
''')

markdown('''
## 2. Reconstruct original Validation counts and fix the comparator

Historical 0.38 is the rounded 0.383297 cache-development F1 cutoff. The
centralized notebook's earlier 0.44 is from original Validation F1 selection.
The historical replacement and OR audits retain their fixed 0.50 rule.

For the new comparison, use Youden's J on the fully identifiable 0.05 grid of
the original 37,086 Validation records. Below the archived 0.50 threshold, the
bin error count is the positive class count. At and above it, the bin error
count is the negative class count. The grid produces an exact confusion table
without using old or new evaluation labels. Apply the chosen comparator to
every revised policy.
''')

code('''
grid = analysis['threshold_grid']
display(grid[['threshold', 'recall', 'specificity', 'precision', 'f1', 'f2', 'youden_j', 'selected']].round(4))
display(grid.loc[grid.selected, ['threshold', 'tn', 'fp', 'fn', 'tp', 'n']])
''')

markdown('''
## 3. Select a supported uncertainty interval and calibrate the score

The historical bin peak of [0.45, 0.50) was measured under a comparator cutoff
of 0.50. At the revised comparator threshold, those records are already
positive. Candidate referral intervals end at the revised XGBoost cutoff.
Intervals starting at 0.15 or 0.20 need outcomes below the 0.25 archive boundary.

Fit sigmoid calibration in five folds on covered comparator negatives in the
2,000-record cached development split. Use out-of-fold scores to choose the
lower edge and calibrated cutoff by positive F2 gain per attempted referral,
with at least 100 development referrals. Refit the final sigmoid on the same
development pool and hold its coefficients fixed. The calibrated target is the
supplied textual label in that score range. The old raw 0.03 and 0.08 operating
points are historical and have no interpretation as clinical probabilities.
''')

code('''
display(pd.Series(analysis['policy'], name='Revised decision rule'))
display(analysis['routing_sweep'].round(5))
display(analysis['calibration_audit'].round(4))
display(Image(filename=str(output / 'figures/calibration.png')))
''')

markdown('''
## 4. Replay paired decisions and compare equal-budget referral

All three policies below use the same XGBoost cutoff and cached denominator.
A successful JEV decision can promote a comparator negative, and an
unavailable score retains the comparator decision. Random referral applies
the same downstream decision to a uniform sample of comparator negatives.
Here the comparison is restricted to observable cached negatives. The
population design uses exterior controls from the entire residual frame.

The 3,000 records were separate from the current development fit. All 5,000
historical records had already been inspected in the earlier studies. Paired
intervals and exact McNemar tests describe this exploratory replay; author
metadata necessary for cluster inference are absent. Expected random totals
are evaluated with paired bootstrap contrasts.
''')

code('''
columns = ['policy', 'precision', 'recall', 'f1', 'f2', 'tp', 'fp', 'fn', 'tn', 'requests']
display(analysis['archive_replay'][columns].round(3))
display(pd.DataFrame(analysis['paired_intervals']).T)
display(analysis['random_contrast'].round(5))
display(analysis['mcnemar'])
display(pd.Series(analysis['historical_or'], name='Historical OR at 0.50'))
display(Image(filename=str(output / 'figures/cached_routing_outcomes.png')))
''')

markdown('''
## 5. Audit the continuous ROC and precision–recall curves

The paired continuous-score comparison uses XGBoost alone and the historical
replacement score with fallback on failed calls. DeLong variance accounts for
paired scores and tied ratings under a record-independence assumption. Average
precision and trapezoidal PR-AUC use separate numerical definitions. The curves
apply to the restricted historical score cohort. The revised cascade produces
binary decisions at a fixed operating point.
''')

code('''
display(analysis['ranking'].round(4))
display(pd.Series(analysis['delong'], name='Paired ROC-AUC audit'))
display(Image(filename=str(output / 'figures/continuous_scores_roc_pr.png')))
''')

markdown('''
## 6. Prepare the independent evaluation and record resources

The 1,067 remaining texts in the original interval require provenance and prior
exclusion checks. The earlier error study accounts for five of them. The old
test's aggregate exposure persists even after disjoint record selection. The
revised sampler takes a census of the new uncertainty band and comparator
positives, plus a probability sample of other comparator negatives. Only
sampled negatives require provider responses. The sample manifest and reference
digest are sealed before collection; labels are read for metrics after the
planned attempts are accounted for.

An actual later source or author/time split would strengthen independence. In
the current environment, the source and full private corpus were unavailable.
The TF-IDF and Word2Vec–XGBoost resource commands will use the restored original
training and Validation splits. The historical prepared-batch timing is already
available. Provider token counts and verified USD costs require usage and
pricing records. All-referral is a hypothetical 1,000 requests per 1,000 texts.
''')

code('''
display(pd.Series(json.loads((output / 'residual_frame_accounting.json').read_text()), name='Residual frame'))
display(pd.Series(json.loads((output / 'resource_accounting.json').read_text()), name='Resource evidence'))
if fresh['state'] == 'replayed':
    display(pd.read_json(output / 'fresh_population.json', precise_float=True))
    display(pd.read_json(output / 'fresh_paired.json', precise_float=True))
    display(pd.read_json(output / 'fresh_strata.json', precise_float=True))
else:
    display(pd.read_json(output / 'fresh_results_pending.json', precise_float=True))
print('Manuscript and reproducibility tables:', output.relative_to(root))
''')

notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
})
destination = ROOT / 'assets/notebooks/jev_asymmetric_cascade_revision.ipynb'
nbformat.write(notebook, destination)
print(destination)
