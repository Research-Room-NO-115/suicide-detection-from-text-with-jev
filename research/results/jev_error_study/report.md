# Jev correction of held-out XGBoost errors

Selected errors: 40; complete paired responses: 0; unresolved pairs: 40.

The prespecified score threshold is 0.50.

## Available model responses

These rates use each model's available responses. All selected denominators and unresolved outcomes remain explicit; rates from different response subsets are not a paired comparison.

| Outcome | Base Jev | Fine-tuned Jev (not run) |
| --- | --- | --- |
| Missed-suicide recovery (XGBoost FN) | 11/20 (55.0%; 95% Wilson CI 34.2%–74.2%); selected n=20 | Not estimable (n=0); selected n=20 |
| False-alarm correction (XGBoost FP) | 15/20 (75.0%; 95% Wilson CI 53.1%–88.8%); selected n=20 | Not estimable (n=0); selected n=20 |
| All sampled errors | 26/40 (65.0%; 95% Wilson CI 49.5%–77.9%); selected n=40 | Not estimable (n=0); selected n=40 |

Weight fine-tuning was not run. No tuned-model behavior or benefit has been established.

## Primary paired comparison

Rates below use the same complete response pairs for both models.

| Outcome | Base Jev | Fine-tuned Jev (not run) | Paired / selected |
| --- | --- | --- | --- |
| Missed-suicide recovery (XGBoost FN) | Not estimable (n=0) | Not estimable (n=0) | 0 / 20 |
| False-alarm correction (XGBoost FP) | Not estimable (n=0) | Not estimable (n=0) | 0 / 20 |
| All sampled errors | Not estimable (n=0) | Not estimable (n=0) | 0 / 40 |

No complete response pairs are available. Model improvement, regression, and paired statistical significance cannot be estimated.

## Unresolved responses and all-selected bounds

| Model | Observed / selected | Failed | Missing | Corrected | Possible all-selected correction rate |
| --- | --- | --- | --- | --- | --- |
| Base Jev | 40 / 40 | 0 | 0 | 26 | 65.0%–65.0% |
| Fine-tuned Jev (not run) | 0 / 40 | 0 | 40 | 0 | 0.0%–100.0% |

These bounds assign every unresolved outcome first to incorrect, then to correct. They are missing-outcome bounds, not statistical confidence intervals.

## Interpretation limits

- Gold labels select XGBoost errors; this is oracle-selected error analysis, not a deployable cascade evaluation.
- Correction rates are conditional on the sampled error mixture; deliberate FN/FP balancing does not estimate natural prevalence.
- The primary comparison uses complete response pairs. Missing and failed outcomes remain explicit, with worst/best bounds over all selected samples.
- Wilson 95% intervals describe binomial uncertainty within the sampled errors; small stratum counts produce wide intervals.
- Social-media dataset labels do not establish clinical validity or deployment performance.
- The original notebook inspected the source test split, and base outcomes are now public. A later tuned comparison reuses inspected cases; freeze all tuning choices on training/validation and describe it as exploratory.
- jev-latest is a mutable provider alias without a recorded immutable revision; later backend changes may confound comparisons performed at different times.
