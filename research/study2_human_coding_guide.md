# Focused human review: provisional coding guide

**Status: source construct, matching texts and qualified reviewers pending.**
This guide does not claim to reproduce the unread IEEE paper's instructions.
Finalize its construct definitions after primary-source verification. Retain
verified source labels as the reference when they answer the question; do not
generate expert labels with JEV or another language model.

## Two samples and their interpretation

The prepared probability sample contains 100 of the historical 5,000 texts, with
conditional inclusion probability 0.02. It supports an audit of this cache's
label fidelity, not full-dataset or fresh Study 2 prevalence. The deliberately
enriched sample contains up to ten cases in each historical transition category:
recovered positive, lost positive, corrected false alarm and new false alarm.
It supports examining distinctions, not estimating their population frequencies.
Record overlap between samples. Never pool the enriched cases into a prevalence
estimate without a valid sampling design.

Public sampling records carry existing stable IDs/hashes only. Blinded forms,
the private linkage key and an unexported blind-ID salt are under ignored
`artifacts/jev_selective_routing_study2_validation/human_review/`. The forms omit
reference labels, model names, scores and transition categories. Raters must not
receive the linkage key or prediction exports before independent coding.
Text is not available now; no substantive annotations have been completed.

## Proposed descriptive fields

Use **present / absent / insufficient evidence** unless a justified alternative
scale is specified. These codes describe language/context, not clinical risk.

| Field | Operational coding instruction |
| --- | --- |
| Person/reference | Identify first-person experience, third-person reference, mixed perspective or insufficient context. Quoting another person is not automatically a first-person disclosure. |
| Temporal context | Distinguish present/current, explicitly past, hypothetical/future or unclear time; do not infer current intent from a past event. |
| Ideation versus intent | Apply the verified source definition when available. Do not infer intent from generic suicide-related discussion; otherwise code insufficient evidence. |
| Quotation/hypothetical language | Record explicit quotation, reported speech, fictional discussion or hypothetical framing; retain uncertainty. |
| Negation | Record scope-changing negation and whether the negated proposition concerns self-harm/intent or a different statement. |
| Help-seeking | Record explicit requests for support, information or assistance separately from an inferred risk label. |
| Context sufficiency | State whether a single post contains enough context for the intended source construct. Do not replace missing author history with speculation. |
| Source construct code | Use the source study's actual categories and unit only after verification. A post does not receive an author-level expert rating merely by inheritance. |
| Ambiguity/insufficient evidence | Preserve unresolved interpretations; do not force a confident binary judgment to match a model. |

## Coding and adjudication procedure

Where qualified reviewers are available, use two independent reviewers, a small
source-informed calibration exercise outside evaluation, independent blinded
forms and a documented adjudication step after the independent decisions are
locked. Record qualifications and reviewer/adjudicator identifiers privately.
Do not substitute model consensus or LLM prose for qualified human coding.

Report agreement per field with the actual independent-coding denominator,
raw agreement, Cohen's kappa for nominal two-rater codes and weighted kappa only
for a justified ordinal scale. For more raters, use an appropriate multi-rater
coefficient with missingness handled explicitly. Show uncertainty and describe
prevalence effects; do not report post-adjudication consensus as independent
inter-rater agreement. Preserve insufficient-evidence cases and explain any
exclusions. The source paper's reported agreement and this new audit's agreement
are separate quantities.

Prefer aggregates. Where permitted and necessary, use brief carefully redacted
or paraphrased examples; do not publish raw sensitive posts, authors or a
reidentifiable key. Source data terms, current-study review and external
processing permission remain to be verified.
