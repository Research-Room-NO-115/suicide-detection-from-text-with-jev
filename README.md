# Jev on XGBoost suicide-detection errors

This research asks how often Jev corrects suicide cases missed by the original
Word2Vec + XGBoost model. It reproduces the notebook's **centralized** baseline,
selects **100 actual XGBoost mistakes**, prepares separate weight-training data,
and evaluates base Jev on an untouched subset under a **100-attempt inference cap**.
All code, notebooks, and reports are in English.

In the recorded run, **base Jev recovered 11/20 missed suicide cases (55%; 95%
Wilson CI 34.2%–74.2%)** and corrected **15/20 false alarms (75%)**. That is 26/40
corrections on the balanced error sample. All 40 paid attempts succeeded; 60
remain under the study cap. The full XGBoost test accuracy was 91.97%.

**True Jev weight fine-tuning has not been performed.** The available repository
API supports inference, but no supported provider training API, training access,
or approved training-fee cap was available. Prepared training files are not a
fine-tuned model. See [training status](research/results/jev_error_study/training_status.json).

## Recorded study

Open [the executed research notebook](assets/notebooks/jev_xgboost_error_rescue.ipynb)
or [the measured results](research/results/jev_error_study/report.md). Its default
Run All reads committed results and makes no paid calls.

The source experiment remains at
[the original notebook](assets/notebooks/federated_xgboost_distribution_iid_non_iid_executed.ipynb).
Its pipeline definitions and hyperparameters are reused without modifying it.
Only one full-data centralized model is fitted; rerunning all 360 federated models
is unnecessary for this comparison.

## Environment and execution

Use Python 3.12 and the CPU-only XGBoost dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m unittest discover -s tests -v
```

The original public dataset is
[Kaggle Suicide and Depression Detection](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).
This run used version 14, `Suicide_Detection.csv`, SHA-256
`1ec16b065a2fae36488de16e8abfc326586a576a25901e09467223e545ddd295`.
Provide the CSV locally or download that version through `kagglehub`:

```python
from pathlib import Path
import kagglehub
root = Path(kagglehub.dataset_download('nikhileswarkomati/suicide-watch/versions/14'))
print(next(root.rglob('Suicide_Detection.csv')))
```

Then prepare the reproducible study. The first run trains the full centralized
model; later runs verify and reuse its cache:

```bash
jev-research prepare --csv /path/to/Suicide_Detection.csv
```

For explicit paid execution, configure `METIS_API_KEY` securely in the process
environment. No key is stored in the code. Then run:

```bash
jev-research evaluate-base
jev-research report
```

`evaluate-base` queries only the 40 prespecified evaluation errors. Successful
results are reused without another paid attempt. `report` is offline and rebuilds
results from the private cohort and verified SQLite ledger, not editable public
score files. A fresh clone can read published results without private data; it
cannot resume this exact study without its original cohort and ledger.

Use `--artifacts PATH` and `--results PATH` **before** the subcommand to select
paths explicitly. The default private directory is `artifacts/jev_study`; the
public directory is `research/results/jev_error_study`.

## Study design

The original raw stratified split is 64% XGBoost training, 16% validation, and 20%
test, seed 42. Cleaning and deduplication preserve train → validation → test
priority. The fixed XGBoost and Jev classification thresholds are both **0.50**.
The source notebook, dataset, configurations, dependencies, and private cache
contents are fingerprinted.

Selection occurs before Jev calls:

| Jev role | Source XGBoost partition | False negatives | False positives | Total |
| --- | --- | ---: | ---: | ---: |
| Prepared training | Validation errors | 20 | 20 | 40 |
| Prepared validation | Validation errors | 10 | 10 | 20 |
| Held-out evaluation | Test errors | 20 | 20 | 40 |

Both error classes prevent an all-positive training set from making an
always-positive response look useful. Selection is deterministic by hash;
insufficient error counts cause a clear failure rather than an altered sample.
Raw and cleaned texts remain disjoint across source partitions. Jev receives the
original text and fixed scoring question, without gold labels or XGBoost error
metadata. Public manifests contain only row IDs, hashes, labels, probabilities,
and provenance; raw text stays in ignored local artifacts.

The primary outcome is **missed-suicide recovery among the 20 held-out XGBoost
false negatives**. False-alarm correction is reported separately. Reports give
Wilson 95% intervals and explicit missing/failed denominators. The reporting code
also supports a future paired base/tuned comparison and exact McNemar test, but
no such comparison is claimed while the tuned arm is absent.

These are gold-label-selected errors and an intentionally balanced FN/FP mixture.
Their correction rate is not overall population accuracy, an operational cascade
improvement, or clinical validation. Small stratum samples create wide intervals.
The source notebook already inspected this XGBoost test split, and this study now
publishes its selected cases and base outcomes. Keep future training choices on
training/validation only and describe a later tuned comparison on these reused
cases as exploratory; it is not a fresh confirmatory test. `jev-latest` is also a
mutable provider alias with no recorded immutable backend revision. Later backend
changes could confound a comparison performed at a different time.

## Cost and credential controls

A durable SQLite ledger reserves every attempt **before** network I/O. One study
is capped at 100 paid inference attempts and 100 unique authorized samples.
Failures and uncertain interrupted requests count. Automatic retries and HTTP
redirects are disabled. Exact successful responses are cached; failed or uncertain
requests are not automatically replayed. Concurrency tests cover threads and
processes. Stop and diagnose a provider failure before further paid work.

Keep the original ledger across restarts. **Do not delete it or create a replacement
to reset the cap.** It is intentionally private and ignored by Git. Base and future
tuned evaluation must share it. Forty base plus forty tuned evaluation calls would
use 80 attempts. Training fees are separate and have not been authorized.

`main.py` now exposes the CLI instead of sending unbudgeted demonstration texts.
The compatibility `API.test.Jev` wrapper requires an authorized sample ID and a
shared `BudgetedJevClient`. Provider authentication is read from `METIS_API_KEY`.
The former embedded key was removed from current source; historical Git commits
still contain it and should be handled through the provider's key rotation process.

## Completing actual weight fine-tuning

The private `train.jsonl` and `validation.jsonl` contain provider-neutral
`sample_id`, `text`, and binary `label` records. Their
[schema](research/training_record.schema.json) is not a verified provider upload
format. No training job or training spend is fabricated.

To complete the requested fine-tuning, obtain the provider's supported upload,
training-job, and custom-model inference specification; confirm training access;
and approve a separate maximum training spend. Convert the prepared records to
that documented schema. Record the actual training job/configuration, model IDs,
manifest hashes, completed status, and spend. Keep evaluation rows out of all
training/model selection, then evaluate the trained model on the same 40 held-out
cases using the existing ledger. Prompt optimization is not weight fine-tuning.
