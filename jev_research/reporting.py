"""Aggregate a paired Jev comparison on held-out, oracle-selected XGBoost errors.

Only aggregate data are exported. A missing response is an unresolved outcome,
not a successful correction and not a reason to silently shrink a denominator.
The primary paired comparison is explicitly conditional on complete response pairs.
"""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping


ERROR_TYPES = ("false_negative", "false_positive")
FAILED_STATUSES = frozenset(
    {"failed", "error", "timeout", "invalid_response", "budget_exhausted", "unknown"}
)


def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    """Return a Wilson score interval; an empty denominator has no interval."""
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("Counts must be integers, not booleans.")
    if not isinstance(successes, Integral) or not isinstance(total, Integral):
        raise ValueError("Counts must be integers.")
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("Counts must satisfy 0 <= successes <= total.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one.")
    if not total:
        return None
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def exact_mcnemar_pvalue(base_only_correct: int, tuned_only_correct: int):
    """Two-sided exact binomial McNemar test on discordant pairs."""
    for count in (base_only_correct, tuned_only_correct):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Discordant pair counts must be nonnegative integers.")
    total = base_only_correct + tuned_only_correct
    if not total:
        return 1.0
    smaller = min(base_only_correct, tuned_only_correct)
    return min(1.0, 2.0 * sum(math.comb(total, k) for k in range(smaller + 1)) / (2**total))


def _missing(value):
    return value is None or type(value).__name__ in {"NAType", "NaTType"} or (
        isinstance(value, Real) and math.isnan(float(value))
    )


def _score(value, field):
    if _missing(value):
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a numeric score, not a boolean or string.")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1].")
    return score


def _records(frame):
    if hasattr(frame, "to_dict"):
        records = frame.to_dict(orient="records")
    else:
        records = [dict(record) for record in frame]
    if not records:
        raise ValueError("A comparison requires at least one selected evaluation sample.")
    if len(records) > 100:
        raise ValueError("The research cohort may contain at most 100 unique error samples.")
    return records


def _failure(record, arm):
    status = str(record.get(f"{arm}_status", "")).lower()
    error = record.get(f"{arm}_error")
    has_error = not _missing(error) and bool(error)
    return status in FAILED_STATUSES or has_error


def _validate(frame, threshold):
    records = _records(frame)
    seen = set()
    validated = []
    for record in records:
        identifier = record.get("sample_id")
        if isinstance(identifier, bool) or not isinstance(identifier, (str, Integral)) or not str(identifier).strip():
            raise ValueError("Every evaluation sample needs a nonempty unique sample_id.")
        identifier = str(identifier)
        if identifier in seen:
            raise ValueError("Duplicate sample_id in evaluation cohort.")
        seen.add(identifier)
        label = record.get("y_true")
        if isinstance(label, bool) or not isinstance(label, Real) or label not in (0, 1):
            raise ValueError("y_true must be numeric 0 or 1.")
        label = int(label)
        error_type = record.get("error_type")
        if error_type not in ERROR_TYPES:
            raise ValueError("Only false_negative and false_positive XGBoost errors are eligible.")
        if (error_type == "false_negative") != (label == 1):
            raise ValueError("error_type and y_true disagree.")
        if "jev_partition" in record and record["jev_partition"] != "evaluation":
            raise ValueError("Tuning samples cannot enter the held-out comparison report.")
        if "source_split" in record and record["source_split"] != "test":
            raise ValueError("Evaluation samples must come from the original XGBoost test split.")
        probability_field = "p_xgb" if "p_xgb" in record else "xgb_probability"
        if probability_field in record:
            probability = _score(record[probability_field], probability_field)
            if probability is None or int(probability >= 0.5) == label:
                raise ValueError("An evaluation row is not an XGBoost error at the fixed 0.50 threshold.")
        row = {"sample_id": identifier, "label": label, "error_type": error_type}
        for arm in ("base", "tuned"):
            score = _score(record.get(f"{arm}_score"), f"{arm}_score")
            failed = _failure(record, arm)
            if score is not None and failed:
                raise ValueError(f"{arm} has both a score and failure status; resolve stale results first.")
            row[f"{arm}_score"] = score
            row[f"{arm}_failed"] = failed
            row[f"{arm}_correct"] = None if score is None else int(score >= threshold) == label
        validated.append(row)
    return validated


def _rate(successes, total):
    return {
        "corrected": successes,
        "n": total,
        "rate": successes / total if total else None,
        "wilson_95": wilson_interval(successes, total),
    }


def _pair_outcomes(rows):
    counts = {"both_correct": 0, "base_only_correct": 0, "tuned_only_correct": 0, "both_incorrect": 0}
    for row in rows:
        base, tuned = row["base_correct"], row["tuned_correct"]
        key = "both_correct" if base and tuned else "base_only_correct" if base else "tuned_only_correct" if tuned else "both_incorrect"
        counts[key] += 1
    improvements, regressions = counts["tuned_only_correct"], counts["base_only_correct"]
    return {
        **counts,
        "n": len(rows),
        "improvements": improvements,
        "regressions": regressions,
        "net_additional_corrections": improvements - regressions,
        "net_correction_rate_change": (improvements - regressions) / len(rows) if rows else None,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(regressions, improvements) if rows else None,
    }


def _paired_summary(selected_rows):
    paired = [row for row in selected_rows if row["base_score"] is not None and row["tuned_score"] is not None]
    return {
        "selected_n": len(selected_rows),
        "paired_n": len(paired),
        "excluded_unresolved_pairs": len(selected_rows) - len(paired),
        "base": _rate(sum(row["base_correct"] for row in paired), len(paired)),
        "tuned": _rate(sum(row["tuned_correct"] for row in paired), len(paired)),
        "paired_outcomes": _pair_outcomes(paired),
    }


def _all_selected_summary(rows, arm):
    available = [row for row in rows if row[f"{arm}_score"] is not None]
    corrected = sum(row[f"{arm}_correct"] for row in available)
    failed = sum(row[f"{arm}_score"] is None and row[f"{arm}_failed"] for row in rows)
    unresolved = len(rows) - len(available)
    return {
        "selected_n": len(rows),
        "observed_n": len(available),
        "corrected": corrected,
        "failed_n": failed,
        "missing_n": unresolved - failed,
        "unresolved_n": unresolved,
        "observed": _rate(corrected, len(available)),
        "observed_correction_rate": corrected / len(available) if available else None,
        # These are deterministic missing-outcome bounds, not confidence intervals.
        "possible_all_selected_rate": [corrected / len(rows), (corrected + unresolved) / len(rows)] if rows else None,
    }


def _metadata(value):
    """Reject obvious text/credential payloads instead of copying them to reports."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key = str(key)
            lowered = key.lower()
            forbidden = ("text", "prompt", "payload", "authorization", "api_key", "token", "password", "secret")
            if any(part in lowered for part in forbidden) and not lowered.endswith(("_sha256", "_hash", "_tokens", "_token_count")):
                raise ValueError(f"Metadata field {key!r} may contain text or credentials; use aggregate audit metadata only.")
            result[key] = _metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real) and math.isfinite(float(value)):
        return float(value)
    raise ValueError("Report metadata must contain finite JSON scalar values, lists, or mappings.")


def build_comparison_report(frame, *, threshold: float = 0.5, metadata: Mapping[str, Any] | None = None):
    """Build a JSON-safe report from held-out error rows with paired Jev scores.

    Required columns: sample_id, y_true, error_type, base_score, tuned_score.
    Missing score columns/None/NaN mean unresolved responses. Optional p_xgb (or
    xgb_probability), source_split, and jev_partition enforce provenance when
    supplied. Per-arm *_status and *_error identify failures separately from
    unattempted/missing responses. Invalid numeric scores raise, rather than
    being silently discarded. The threshold is prespecified at 0.50.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or float(threshold) != 0.5:
        raise ValueError("The prespecified Jev comparison threshold must remain 0.50.")
    rows = _validate(frame, float(threshold))
    strata = {error: [row for row in rows if row["error_type"] == error] for error in ERROR_TYPES}
    strata["all_errors"] = rows
    primary = {name: _paired_summary(subset) for name, subset in strata.items()}
    arm_summaries = {arm: {name: _all_selected_summary(subset, arm) for name, subset in strata.items()} for arm in ("base", "tuned")}
    limitations = [
        "Gold labels select XGBoost errors; this is oracle-selected error analysis, not a deployable cascade evaluation.",
        "Correction rates are conditional on the sampled error mixture; deliberate FN/FP balancing does not estimate natural prevalence.",
        "The primary comparison uses complete response pairs. Missing and failed outcomes remain explicit, with worst/best bounds over all selected samples.",
        "Wilson 95% intervals describe binomial uncertainty within the sampled errors; small stratum counts produce wide intervals.",
        "Social-media dataset labels do not establish clinical validity or deployment performance.",
        "The original notebook inspected the source test split, and base outcomes are now public. A later tuned comparison reuses inspected cases; freeze all tuning choices on training/validation and describe it as exploratory.",
        "jev-latest is a mutable provider alias without a recorded immutable revision; later backend changes may confound comparisons performed at different times.",
    ]
    if not strata["false_positive"]:
        limitations.append("No false-positive controls were selected; specificity and resistance to an always-positive predictor cannot be assessed.")
    if not strata["false_negative"]:
        limitations.append("No false-negative samples were selected; missed-suicide recovery cannot be estimated.")
    return {
        "schema_version": 1,
        "analysis_population": "held_out_oracle_selected_xgboost_errors",
        "comparison_status": "complete" if primary["all_errors"]["paired_n"] == len(rows) else "partial" if primary["all_errors"]["paired_n"] else "unavailable_no_complete_pairs",
        "threshold": 0.5,
        "selected_counts": {name: len(subset) for name, subset in strata.items()},
        "primary_paired": primary,
        "all_selected_by_arm": arm_summaries,
        "metadata": _metadata(metadata or {}),
        "limitations": limitations,
    }


def _format_rate(result):
    if not result["n"]:
        return "Not estimable (n=0)"
    low, high = result["wilson_95"]
    return f"{result['corrected']}/{result['n']} ({result['rate']:.1%}; 95% Wilson CI {low:.1%}–{high:.1%})"


def render_markdown(report):
    """Render English aggregate findings without raw texts or row-level scores."""
    primary = report["primary_paired"]
    total = primary["all_errors"]
    training_status = report.get("metadata", {}).get("training_status", "not_recorded")
    tuned_label = "Fine-tuned Jev (not run)" if training_status == "blocked_not_run" else "Fine-tuned Jev"
    lines = [
        "# Jev correction of held-out XGBoost errors",
        "",
        f"Selected errors: {total['selected_n']}; complete paired responses: {total['paired_n']}; unresolved pairs: {total['excluded_unresolved_pairs']}.",
        "",
        "The prespecified score threshold is 0.50.",
        "",
        "## Available model responses",
        "",
        "These rates use each model's available responses. All selected denominators and unresolved outcomes remain explicit; rates from different response subsets are not a paired comparison.",
        "",
        f"| Outcome | Base Jev | {tuned_label} |",
        "| --- | --- | --- |",
    ]
    for name, label in (("false_negative", "Missed-suicide recovery (XGBoost FN)"), ("false_positive", "False-alarm correction (XGBoost FP)"), ("all_errors", "All sampled errors")):
        base = report["all_selected_by_arm"]["base"][name]
        tuned = report["all_selected_by_arm"]["tuned"][name]
        lines.append(f"| {label} | {_format_rate(base['observed'])}; selected n={base['selected_n']} | {_format_rate(tuned['observed'])}; selected n={tuned['selected_n']} |")
    if training_status == "blocked_not_run":
        lines += ["", "Weight fine-tuning was not run. No tuned-model behavior or benefit has been established."]
    lines += [
        "",
        "## Primary paired comparison",
        "",
        "Rates below use the same complete response pairs for both models.",
        "",
        f"| Outcome | Base Jev | {tuned_label} | Paired / selected |",
        "| --- | --- | --- | --- |",
    ]
    for name, label in (("false_negative", "Missed-suicide recovery (XGBoost FN)"), ("false_positive", "False-alarm correction (XGBoost FP)"), ("all_errors", "All sampled errors")):
        item = primary[name]
        lines.append(f"| {label} | {_format_rate(item['base'])} | {_format_rate(item['tuned'])} | {item['paired_n']} / {item['selected_n']} |")
    paired = total["paired_outcomes"]
    pvalue = "Not estimable" if paired["exact_mcnemar_pvalue"] is None else f"{paired['exact_mcnemar_pvalue']:.6g}"
    paired_statement = (
        f"Paired improvements: {paired['improvements']}; regressions: {paired['regressions']}; net additional corrections: {paired['net_additional_corrections']}. Two-sided exact McNemar p-value: {pvalue}."
        if total["paired_n"] else
        "No complete response pairs are available. Model improvement, regression, and paired statistical significance cannot be estimated."
    )
    lines += [
        "",
        paired_statement,
        "",
        "## Unresolved responses and all-selected bounds",
        "",
        "| Model | Observed / selected | Failed | Missing | Corrected | Possible all-selected correction rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, label in (("base", "Base Jev"), ("tuned", tuned_label)):
        item = report["all_selected_by_arm"][arm]["all_errors"]
        low, high = item["possible_all_selected_rate"]
        lines.append(f"| {label} | {item['observed_n']} / {item['selected_n']} | {item['failed_n']} | {item['missing_n']} | {item['corrected']} | {low:.1%}–{high:.1%} |")
    lines += ["", "These bounds assign every unresolved outcome first to incorrect, then to correct. They are missing-outcome bounds, not statistical confidence intervals.", "", "## Interpretation limits", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def write_report(report, directory):
    """Write aggregate report.json and report.md; no sample texts are exported."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = directory / "report.json", directory / "report.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
