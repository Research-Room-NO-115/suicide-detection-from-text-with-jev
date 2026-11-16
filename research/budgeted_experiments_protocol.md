# Cached JEV policy comparisons and optional 500-attempt experiment

The executed notebook is
[`jev_probability_hybrid_budgeted_experiments.ipynb`](../assets/notebooks/jev_probability_hybrid_budgeted_experiments.ipynb).
Its supporting tables, scores, manifests and figures are in
[`results/jev_probability_hybrid_budgeted_experiments`](results/jev_probability_hybrid_budgeted_experiments).
The previous notebook, result directories and private ledgers are never overwritten.

## Reproduce the offline study

Use the repository's Python 3.12 environment:

```bash
cd /workspace/suicide-detection-from-text-with-jev
bash scripts/setup_environment.sh
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/artifacts/environment/matplotlib"
export IPYTHONDIR="$PWD/artifacts/environment/ipython"
export JUPYTER_CONFIG_DIR="$PWD/artifacts/environment/jupyter"
export JUPYTER_RUNTIME_DIR="$PWD/artifacts/environment/runtime"
python -m unittest discover -s tests -v
```

Execute the delivered notebook, retaining outputs only in that new notebook:

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path("assets/notebooks/jev_probability_hybrid_budgeted_experiments.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=240, kernel_name="python3",
               resources={"metadata": {"path": str(Path.cwd())}}).execute()
nbformat.write(notebook, path)
```

Default Run All blocks HTTP, performs zero paid calls and trains no baseline.
`scripts/build_budgeted_notebook.py` regenerates the new notebook's source cells;
execute it afterwards if you use the builder, since rebuilding clears outputs.

## Evidence and policy selection

- The public 5,000-row export is matched to the original routing manifest and
  checked against saved metrics. Four failed requests retain XGB@0.50.
- Split rows by SHA-256 of `42:budgeted-cache-split:<sample_id>`; first 2,000 are
  development, remaining 3,000 evaluation. Save the identity manifest and fitted
  specifications. Do not call this independent confirmation: these test texts
  were inspected in the original studies.
- Primary fixed rules: XGB@0.50, replacement, OR, AND, 50:50 mean. Development
  choices: XGB F1/F2 thresholds, a JEV weight from 0.25/0.50/0.75 minimizing log
  loss, sigmoid calibrators on clipped logits, and a two-feature L2 logistic
  combiner (C=1). No evaluation labels fit any of these models or cutoffs.
- At recall targets 0.90 and 0.95, minimize development false alarms subject to
  target recall. Ties prefer precision, then the higher threshold. Show achieved
  evaluation recall: a common development target is not matched actual recall.
- Calibration/fusion fit only successful development JEV responses. Unavailable
  required JEV results keep XGB@0.50 at every threshold. Unrouted examples also
  keep XGB@0.50. OR requires a response only when `p_xgb < policy_threshold`;
  AND only when `p_xgb >= policy_threshold`. Their binary request saving does
  not give both raw scores for all rows, so probability metrics are not claimed
  for these binary rules in the primary comparisons.
- Routing candidates: original [0.25,0.80), narrow [0.40,0.60), below [0.25,0.50),
  above [0.50,0.80), crossed with replacement, OR and fixed mean, plus no JEV.
  Select on development for hypothetical FN:FP loss ratios 1, 5 and 10, breaking
  ties by fewer requests and name. Do not select sparse-bin rules on evaluation.
- Proper-score losses and reliability are conditional on cache coverage. The
  `jev-latest` output is an uncalibrated risk rating, not token confidence.
- Fixed-manifest full-test counts/losses can be derived from unchanged outside
  aggregates. Full ranking metrics and globally tuned thresholds remain
  unavailable without original per-example full-test scores. Never synthesize
  those missing predictions from aggregates or train a replacement silently.
- Use 4,000 paired row bootstrap replicates for confusion-based metric changes
  and 2,000 for proper-score losses, seed 42. Intervals are pointwise conditional
  on frozen policy fits, excluding policy-selection variability. Exact McNemar
  tests address overall correctness and sensitivity separately. Holm families
  are the fixed-policy and target-recall comparisons, separately by endpoint.
  Without author IDs there is no author-level resampling.

## Missing artifacts and pending text review

This checkout has no original version-14 CSV, private cohort/request ledger,
full validation/test score tables or fitted baseline models. Kaggle recovery
failed with a network-policy proxy 403 for `api.kaggle.com`. Git read access
worked and these artifacts are not tracked. The source pipeline can be audited
for class mapping, train-only feature learning and duplicate-handling logic;
the missing private split tables prevent runtime revalidation of the original
raw/cleaned-text separation. No author/time metadata is supplied.

The new error-review manifest contains eight deterministic IDs for each of four
transition categories. Semantic annotations are pending matching text. Do not
infer negation, quotation, third-person reference, temporal context, label
ambiguity or provider truncation from scores. Use the saved review rubric and
brief redacted excerpts if texts later become available. Do not publish raw
social-media text. JEV sends original text without client-side truncation;
provider-side context handling is unknown.

## Optional additional experiment: prepare first, do not dispatch by default

The design is 200 development examples from the original XGB validation split
and 300 evaluation examples from the original test split. Score-only strata:

| Stratum | XGB range | Development | Evaluation |
| --- | --- | ---: | ---: |
| Near threshold | [0.40,0.60) | 100 | 150 |
| Eligible baseline negative | [0.25,0.40) | 60 | 90 |
| Eligible baseline positive | [0.60,0.80) | 20 | 30 |
| Random control | <0.25 or >=0.80 | 20 | 30 |

Hash-rank stable IDs within each stratum; no labels or correctness enter sampling.
Exclude all selected texts from both previous study manifests, including earlier
tuning samples and failed requests. The generator records frame counts, quota /
frame-size inclusion probabilities and overlap flags. Fail on insufficient
support rather than silently modifying quotas. The actual manifest is pending
the original full score frames; the saved design does not invent IDs or counts.
Newly sampled old-test rows remain exploratory despite not appearing in the old
selected manifests. Independent future data would be stronger.

Once the original private baseline cache is supplied, validate and reuse it:

```python
import json
from pathlib import Path
import pandas as pd
from jev_research.notebook_pipeline import _read_cache
from jev_research.additional_experiment import (
    prepare_additional_plan, manifest_hash, AdditionalJevClient,
    collect_additional_phase, freeze_additional_policies, analyze_additional,
)
from jev_research.budgeted_experiments import save_json, save_table

root = Path.cwd()
# Set this to a supplied cache, not a newly trained approximation.
cache = root / "artifacts/jev_study/baseline/b154407429c3f6d7dd39f87006eb4dd9305d283146ccf27546d57f0d27fa7fd2"
metadata = json.loads((cache / "metadata.json").read_text())
assert metadata["baseline_fingerprint"] == cache.name
historical = json.loads((root / "research/results/jev_probability_hybrid/study.json").read_text())
assert metadata["baseline_fingerprint"] == historical["baseline_fingerprint"]
assert metadata["identity"]["dataset_sha256"] == historical["dataset_sha256"]
frames = _read_cache(cache, metadata["identity"], metadata["baseline_fingerprint"]).frames
previous = {
    "probability_study": pd.read_json(root / "research/results/jev_probability_hybrid/routing_manifest.json", precise_float=True),
    "error_study": pd.read_json(root / "research/results/jev_error_study/selection_manifest.json", precise_float=True),
}
manifest, design = prepare_additional_plan(frames, previous)
public = root / "research/results/jev_probability_hybrid_budgeted_experiments"
private = root / "artifacts/jev_probability_hybrid_budgeted_experiments"
private.mkdir(parents=True, exist_ok=True)
save_table(public, "additional_manifest", manifest)
save_json(public / "additional_design.json", design)
```

This prepares a frozen manifest; it does not enable inference. The 19 compatible
earlier error-study responses outside the probability cache are excluded instead
of repurchased. For common-population comparisons, the original probability
response takes precedence on the 21 overlaps, even when another response would
look better. No pending/failed historical request is resent.

## Explicitly authorized execution only

Use **one retained ledger** at
`artifacts/jev_probability_hybrid_budgeted_experiments/budget.sqlite3` for every
new development/evaluation request. Never remove it, switch paths, alter a
manifest or create another ledger to reset the study cap. Keep it across tasks.
No training fees or prompt/model searches are authorized by the attempt limit.

The following configuration is intentionally inert:

```python
RUN_NEW_JEV_INFERENCE = False
MAX_NEW_JEV_ATTEMPTS = 500
AUTHORIZED_MANIFEST_SHA256 = None  # Exact reviewed design digest after explicit authorization.
budget = {}  # No known monetary price/currency; nothing invented.
args = dict(ledger_path=private / "budget.sqlite3",
            run_new_inference=RUN_NEW_JEV_INFERENCE,
            authorized_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
            max_attempts=MAX_NEW_JEV_ATTEMPTS, **budget)

# Disabled returns without dispatch or creating a paid-study ledger.
collect_additional_phase(manifest, frames, phase="development", **args)
```

After the user explicitly authorizes the concrete manifest and activates that
run, provide `METIS_API_KEY` securely through the environment. Never put a key in
notebook outputs, source, manifests or chat. Network access to `api.metisai.ir`
must also be usable; no provider call was made to test this in the offline study.
Set the boolean and exact manifest digest for that authorized run only. Complete
development, then freeze decisions before collecting evaluation:

```python
# Run only in an explicitly authorized context; args must carry its exact digest.
collect_additional_phase(manifest, frames, phase="development", **args)
client = AdditionalJevClient(private / "budget.sqlite3", manifest,
                             max_attempts=MAX_NEW_JEV_ATTEMPTS, **budget)
decisions = freeze_additional_policies(manifest, frames, client,
                                     public / "additional_frozen_decisions.json")
collect_additional_phase(manifest, frames, phase="evaluation",
                         frozen_decisions=decisions, **args)
analyze_additional(manifest, frames, client, decisions, public)
```

The ledger reserves requests before dispatch, disables automatic retries and
redirects, caches successes, counts failures and uncertain interruptions, and
seals the decision digest before evaluation. A provider failure stops the runner;
diagnose it before deliberately resuming unattempted rows. Failed/uncertain rows
remain fallback rows rather than being resent. A configured allowance alone
does not enable a request.

If the user supplies a monetary cap, configure `monetary_cap`,
`request_cost_upper_bound`, `currency`, and `pricing_evidence` together in `budget`.
The bound must come from verified provider pricing that covers any charge for
one request with this configuration, including failures. Unknown/token-dependent
unbounded prices cannot enforce a hard monetary cap; the client refuses a cap
without the bound. The cap and bound are represented in exact millionths and
maximum-charge reservations commit atomically with attempts. Reservations are
never released for failures/interruptions. These retained upper-bound amounts
are not measured currency spending; actual currency cost remains unknown with
the available response parser. Use the same monetary configuration on restart.

Additional reports provide stratum results and explicitly label unweighted
oversampled-subset metrics. They do not estimate full-dataset performance.
Inverse-inclusion weighting would target only the frame after exclusions and
would need design-aware uncertainty before any population claim.

## Validation evidence

The delivered default notebook runs 101 tests: 83 existing and 18 new. All
inference is mocked. A 200+300 mock integration run checks the full preparation,
development, freeze, evaluation and reporting flow. Other tests check concurrent
budget reservation, zero dispatch without activation, cap persistence, cache
reuse, failed/uncertain calls, optional monetary caps and changed decisions.
These test transport calls are not paid attempts. Check `verification.json`
and the executed notebook for actual run evidence.
