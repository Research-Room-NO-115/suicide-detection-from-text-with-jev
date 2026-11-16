# Study 2: source verification, frozen sampling and authorized live collection

The project owner has authorized **up to 10,000 additional real JEV attempts**
for this follow-up. This authorization is already granted. It does not need to
be requested again. Historical 100/5,000-attempt ledgers and the previous optional
500-attempt design retain their original scopes and caps; do not increase or
reset them. All Study 2 requests must use one retained ledger:
`artifacts/jev_selective_routing_study2_validation/budget.sqlite3`.

Current status: **no Study 2 live collection**. Source-reference compatibility,
data transfer terms, the full original text/score frame and live credentials
remain unresolved. The executed notebook and manuscript draft label this state
explicitly. Cached development findings must not be represented as fresh Study 2
or as a validation against expert post-level risk judgments.

## Source and reference requirements

The owner supplied <https://ieeexplore.ieee.org/document/9591887> and did not know
the record mapping or annotation unit. IEEE, Kaggle and Crossref returned proxy
403 responses. The repository contains neither the paper/supplement nor an
expert-to-CSV mapping. Do not invent the paper's title, DOI, raters, agreement,
collection period, approval or dataset license.

Read the primary source and release documentation when access is available.
Verify whether the source annotation unit is a post or an author with multiple
posts, what context raters saw, the construct definition, specialist coverage and
the exact version/record linkage. `verify_post_mapping` checks row IDs, text hashes,
source record IDs and unchanged labels; author labels inherited by individual
posts are explicitly incompatible with a claim of independently rated posts.
Matching identities does not itself verify rater qualifications or procedures.

The environment draft adds `ieeexplore.ieee.org`, `www.kaggle.com`,
`api.kaggle.com`, `api.crossref.org`, `doi.org`, and `api.metisai.ir` while retaining
the existing package-manager presets. It declares `METIS_API_KEY` for the JEV
destination. Saving that draft did not apply networking or supply a credential.
Use environment settings to enter the key securely, review/save the settings and
publish the environment as required by the platform. Never place keys in chat,
Git, notebook outputs or raw logs. Retry source/API operations only after a
meaningful configuration change. Document any further download redirect domain
actually required by the source service rather than replacing unknown policy.

Applicable terms must permit sending text to JEV. Source-study ethics approval,
current-study review and permission for this external transfer are different
claims. Unknown terms are not reported as a verified prohibition, but they are
not treated as established permission either. No bulk new expert annotation is
required just for publication when compatible original specialist labels exist.

## Development evidence and candidate protocol

`run_offline` recomputes the old 3,000-row comparisons, audits the 21 repeated
texts, uses all historical 5,000 rows for conditional operating-point planning,
and saves a dated internal **design draft** and digest. It does not call this a
public preregistration or a frozen collection protocol. Fallback in the proposed
Study 2 policy is the *tuned comparator*, explicitly different from Study 1's
fixed 0.50 fallback. No primary calibration/fusion is used; those remain secondary.

No defensible clinical noninferiority margin is currently available. The report
therefore estimates paired sensitivity differences and intervals; a
nonsignificant difference is not preservation. The decision helper requires an
external justification to use a margin and a strict lower-bound comparison.
Illustrative 1/2/5-point margins are sensitivity scenarios, not accepted clinical
tolerances. A one-sided 97.5% lower bound corresponds to the lower end of the
reported two-sided 95% interval for this decision.

## Sampling and precision before spending

Prefer a separately collected, construct-compatible expert-labeled population
with unused authors/time periods. The implemented fallback samples unqueried
original test posts, disclosing their prior aggregate baseline exposure. Full
original test scores and exact texts are required; no replacement baseline is
trained silently. If rebuilding is necessary, fit/select upstream and record
the new fingerprint instead of calling it the historical model.

Exclude all training/development sets, historical JEV cohorts, earlier
error-study selections and failed requests. Where present, the sampler excludes
matching raw hashes, cleaned text and known authors. Near-duplicate and temporal
checks need the actual source records and must be documented; do not describe
the current hash checks as proof of semantic or author independence.

Aggregate arithmetic leaves **41,323** potential old-test records and **1,062**
eligible [0.25,0.80) records before further exclusions. The old 5,000-row cache
cannot supply their IDs, outside-range JEV scores or the final sampling frame.
Its removal also changes the target routing fraction to about 2.57%, so using
the old 13.09% interval fraction would produce an unfair random comparator.

The generator `build_sample` uses four mutually exclusive score strata: below
0.25, [0.25, frozen XGB cutoff), [cutoff,0.80), and at/above 0.80. It can census
the residual difficult interval and probability-sample both exterior strata,
recording `N_h`, `n_h`, and inclusion probabilities. Selection discards labels
and correctness before hash ranking. Every stratum needed for the population
claim has nonzero support. The final sample size is not fixed merely because
10,000 attempts are allowed. Illustrative routed-population precision scenarios
and author design-effect scenarios are supplied, while a full-population random
comparison needs exterior-region assumptions or evidence. No paid repeats or
pilot are presently planned; any justified adaptation must be versioned before
collection, excluded from primary evaluation and charged to the same ceiling.

One response per sampled fresh post supports every policy without repeated paid
queries. Actual research acquisition attempts are separate from prospective
requests under a policy. Random referral uses the **population** selective
referral fraction, identical downstream cutoff and fallback. Expected confusion
totals average the uniform fixed-budget random policy; precision is the ratio of
those expected totals, not an asserted expected value of randomized precision.

## Freeze a concrete package

After source verification, complete data access and a design-aware precision
justification, use these helpers from the repository root:

```python
import json
from pathlib import Path
from jev_research.study2 import build_sample
from jev_research.study2_collection import freeze_for_collection, write_frozen_package

result = Path('research/results/jev_selective_routing_study2_validation')
draft = json.loads((result / 'protocol.json').read_text())

# Supply validated original/full frames and all training/development/prior
# selections in `exclusions`; they are not synthesized from public aggregates.
# justified_n <= 10000 must be determined from the complete frame and precision.
manifest, design = build_sample(test_frame, exclusions,
    draft['candidate_policy'], requested_n=justified_n)

# These objects contain verifiable source/mapping and terms evidence, not keys.
# reference_evidence includes primary-source hash, source citation, construct,
# qualifications, reviewer identity and a compatible post-level mapping report.
frozen = freeze_for_collection(draft, manifest, design,
    reference_evidence=reference_evidence,
    transfer_evidence=transfer_evidence,
    precision_justification=precision_justification,
    output=result / 'frozen_protocol_v1.json')
write_frozen_package(result / 'frozen_v1', frozen, manifest, design)
```

`write_frozen_package` retains full numeric round-trip precision for digest-bound
manifests. A 15-decimal DataFrame export can change a digest on reloading; do not
replace this serialization with a lossy export. Keep private selected source text
and reference labels at
`artifacts/jev_selective_routing_study2_validation/evaluation_private.csv.gz`.
No raw text, author IDs, credentials or provider payloads belong in the public
package.

## Separate resumable live command

Status is read-only and makes no provider requests:

```bash
source .venv/bin/activate
python -m jev_research.study2_collection --status
```

Once the concrete package and prerequisites above are verified, activate the
already authorized collection with:

```bash
python -m jev_research.study2_collection --collect \
  --protocol research/results/jev_selective_routing_study2_validation/frozen_v1/protocol.json \
  --manifest research/results/jev_selective_routing_study2_validation/frozen_v1/manifest.json \
  --design research/results/jev_selective_routing_study2_validation/frozen_v1/design.json \
  --private-source artifacts/jev_selective_routing_study2_validation/evaluation_private.csv.gz \
  --workers 4
```

The one persistent ledger reserves attempts atomically before dispatch. Success,
HTTP failure, timeout and uncertain interruption all consume the same cap.
Successful exact requests are cache reads; failures/uncertain rows are never
resent. On an error, scheduling stops; already in-flight bounded requests still
count. Diagnose before deliberately resuming the same command. Never delete or
replace a ledger, change manifests or raise a stored cap to reset allowance.
The inherited client supports conservative monetary reservations if a monetary
cap and verified per-request upper bound/currency are supplied programmatically;
no price or currency is invented here. Routine notebooks never invoke this
command. The current real ledger does not exist because zero calls were made.

## Cache-only replay, figures, manuscript and tests

The new executed notebook is
`assets/notebooks/jev_selective_routing_study2_validation.ipynb`. Its builder is
`scripts/build_study2_notebook.py`; rebuild only that notebook, then execute it
with nbclient, Python 3, repository root as the working directory, timeout 240.
`replay_fresh_if_complete` reads the frozen package and SQLite ledger read-only,
verifies request/protocol hashes and freeze timestamps, and refuses to issue
final results from an incomplete collection. Missing provider outcomes remain
predefined fallback cases once every selected attempt is accounted for.

Population estimates use inverse-inclusion weighted paired totals. Bootstrap
perturbations preserve strata/weights and finite-frame correction. Author
clusters require a compatible prespecified design; the code refuses ordinary
row inference with repeated identifiable authors. No ordinary exact McNemar
test is applied to weighted or expected-random counts. These intervals do not
include provider variability, reference-label uncertainty or policy selection.

The manuscript report, population and difficult-stratum tables, two PNG/PDF/SVG
figures and reviewer-readiness table are under the new result directory. While
Study 2 is blocked, figure 2 is explicitly historical development evidence;
uncollected Study 2 results are never drawn as zero effects. Human-review forms
and the blind linkage key stay in ignored private artifacts. The public
probability and enriched samples are clearly labeled historical, and new
qualified coding remains pending. Coding definitions must be finalized against
the actual paper; no generated specialist judgments are supplied.

Run the legacy suite and the explicit new suite separately:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests/study2 -v
```

The separate Study 2 directory intentionally has no `__init__.py`, preserving
the historical notebook's recorded 101-test discovery contract. The new suite
tests sampling/exclusions, population budget matching despite oversampling,
finite-frame uncertainty, label-unit mismatch, source/transfer gates, frozen
hashes, cap enforcement, failure/resume and a complete mocked collection plus
read-only replay. All test transports are synthetic; none are paid requests.
