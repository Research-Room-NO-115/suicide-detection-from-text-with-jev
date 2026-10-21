# Probability-routed Jev hybrid versus XGBoost

Selected test cases: 5000; successful Jev responses: 4996; selected XGBoost fallbacks: 4.

Both models use threshold 0.50. Every selected case remains in the primary comparison, including failed or missing requests.

## Primary: identical selected test cases

Population n=5000.

| Metric | XGBoost | Hybrid |
| --- | ---: | ---: |
| Accuracy | 63.74% | 78.38% |
| Balanced accuracy | 63.28% | 79.26% |
| Positive precision | 66.16% | 88.40% |
| Positive recall | 68.42% | 69.41% |
| Specificity | 58.15% | 89.11% |
| Positive F1 | 0.672685 | 0.777618 |
| Macro F1 | 0.633137 | 0.783633 |
| ROC-AUC | 0.684172 | 0.889443 |
| Average precision | 0.709576 | 0.889352 |
| Brier score | 0.223236 | 0.164997 |
| Log loss | 0.637583 | 0.672543 |

Corrected XGBoost errors: 1339; newly introduced errors: 607; net additional correct cases: 732; accuracy change: +14.64%.
Two-sided exact McNemar p-value: 3.67801e-63.

Corrections must be considered alongside newly introduced errors within each class:

| Error class | XGBoost errors | Corrected | Newly introduced | Hybrid errors | Net reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Missed suicide cases (FN) | 860 | 538 | 511 | 833 | +27 |
| False alarms (FP) | 953 | 801 | 96 | 248 | +705 |

| Error outcome | Count / eligible | Rate | 95% Wilson interval |
| --- | ---: | ---: | --- |
| Missed-suicide recovery | 538/860 | 62.56% | 59.27%–65.73% |
| False-alarm correction | 801/953 | 84.05% | 81.59%–86.24% |
| Regression among XGBoost-correct cases | 607/3187 | 19.05% | 17.72%–20.45% |

## Full test: partial routing of the exact selected manifest

Population n=46342.

| Metric | XGBoost | Hybrid |
| --- | ---: | ---: |
| Accuracy | 91.97% | 93.55% |
| Balanced accuracy | 91.97% | 93.55% |
| Positive precision | 91.88% | 94.76% |
| Positive recall | 92.09% | 92.21% |
| Specificity | 91.85% | 94.89% |
| Positive F1 | 0.919864 | 0.934671 |
| Macro F1 | 0.919684 | 0.935470 |
| ROC-AUC | 0.975169 | 0.974644 |
| Average precision | 0.974449 | 0.973709 |
| Brier score | 0.059086 | 0.052802 |
| Log loss | 0.201994 | 0.205766 |

Corrected XGBoost errors: 1339; newly introduced errors: 607; net additional correct cases: 732; accuracy change: +1.58%.
Two-sided exact McNemar p-value: 3.67801e-63.

Corrections must be considered alongside newly introduced errors within each class:

| Error class | XGBoost errors | Corrected | Newly introduced | Hybrid errors | Net reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Missed suicide cases (FN) | 1835 | 538 | 511 | 1808 | +27 |
| False alarms (FP) | 1887 | 801 | 96 | 1182 | +705 |

| Error outcome | Count / eligible | Rate | 95% Wilson interval |
| --- | ---: | ---: | --- |
| Missed-suicide recovery | 538/1835 | 29.32% | 27.28%–31.44% |
| False-alarm correction | 801/1887 | 42.45% | 40.24%–44.69% |
| Regression among XGBoost-correct cases | 607/42620 | 1.42% | 1.32%–1.54% |

## Routing outcomes

| Outcome | Count |
| --- | ---: |
| succeeded | 4996 |
| failed | 4 |
| pending | 0 |
| not_attempted | 0 |

## Interpretation limits

- The selected subset contains probability-eligible cases, not gold-label-selected errors. Corrections and newly introduced errors are both counted.
- The full-test result applies only to partial routing of the exact selected manifest; it does not route every case in the critical interval unless the manifest includes them all.
- Failed, pending, and unattempted selected requests fall back to XGBoost and remain in every operational denominator.
- The original notebook and earlier study inspected the test split; this reused-test comparison is exploratory. Interval selection and all policy choices must use validation data only.
- Hybrid ROC-AUC, AP, Brier score, and log loss use piecewise XGBoost/Jev scores. Those scores are not jointly calibrated, so these metrics describe this policy rather than calibrated clinical probabilities.
- jev-latest is a mutable provider alias without a recorded immutable backend revision; later backend changes can affect reproducibility.
- Social-media labels do not establish clinical validity or deployment performance. Confidence intervals are conditional on this selected evaluation population.
