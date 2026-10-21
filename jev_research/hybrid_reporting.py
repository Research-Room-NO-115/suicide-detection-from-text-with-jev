"""Compare a probability-routed Jev hybrid with XGBoost on identical cases.

Missing provider outcomes fall back to XGBoost. They never disappear from the
denominator. Only aggregate metrics and redacted configuration are exported.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .reporting import wilson_interval


def _records(value: Any, name: str) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        records = value.to_dict(orient="records")
    else:
        records = [dict(row) for row in value]
    if not all(isinstance(row, Mapping) for row in records):
        raise ValueError(f"{name} must contain records.")
    return records


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric, not a boolean or string.")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return value


def _row_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("row_id must be a nonnegative integer.")
    return int(value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _checksum(value: Any) -> str:
    value = _identifier(value, "text_sha256")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("text_sha256 must be a lowercase SHA-256 digest.")
    return value


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key = str(key)
            lowered = key.lower()
            sensitive = ("text", "prompt", "payload", "authorization", "api_key", "token", "password", "secret")
            aggregate = lowered.endswith(("_sha256", "_hash", "_count", "_counts", "_n", "_tokens"))
            if any(word in lowered for word in sensitive) and not aggregate:
                raise ValueError(f"Metadata field {key!r} may contain raw content or credentials.")
            result[key] = _safe_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real) and math.isfinite(float(value)):
        return float(value)
    raise ValueError("Metadata must contain finite JSON scalars, lists, or mappings.")


def _rate(corrected: int, n: int) -> dict[str, Any]:
    return {"corrected": corrected, "n": n, "rate": corrected / n if n else None,
            "wilson_95": wilson_interval(corrected, n)}


def _metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, confusion_matrix, f1_score,
        log_loss, precision_score, roc_auc_score,
    )

    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    positive, negative = tp + fn, tn + fp
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    defined_class_recalls = [value for value in (recall, specificity) if value is not None]
    correct = tp + tn
    two_classes = bool(positive and negative)
    return {
        "n": len(truth), "positive_n": positive, "negative_n": negative,
        "correct": correct, "accuracy": correct / len(truth),
        "accuracy_wilson_95": wilson_interval(correct, len(truth)),
        "balanced_accuracy": sum(defined_class_recalls) / len(defined_class_recalls),
        "precision_positive": float(precision_score(truth, predicted, zero_division=0)),
        "recall_positive": recall, "specificity": specificity,
        "f1_positive": float(f1_score(truth, predicted, zero_division=0)),
        "f1_macro": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)) if two_classes else None,
        "average_precision": float(average_precision_score(truth, scores)) if two_classes else None,
        "brier_score": float(brier_score_loss(truth, scores)),
        "log_loss": float(log_loss(truth, scores, labels=[0, 1])),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _population(truth: np.ndarray, xgb: np.ndarray, hybrid: np.ndarray) -> dict[str, Any]:
    from scipy.stats import binomtest

    xgb_prediction, hybrid_prediction = xgb >= 0.5, hybrid >= 0.5
    xgb_correct, hybrid_correct = xgb_prediction == truth, hybrid_prediction == truth
    corrected = int(np.sum(~xgb_correct & hybrid_correct))
    regressed = int(np.sum(xgb_correct & ~hybrid_correct))
    discordant = corrected + regressed
    false_negative = (truth == 1) & ~xgb_prediction
    false_positive = (truth == 0) & xgb_prediction
    corrected_false_negatives = int(np.sum(false_negative & hybrid_correct))
    corrected_false_positives = int(np.sum(false_positive & hybrid_correct))
    introduced_false_negatives = int(np.sum((truth == 1) & xgb_correct & ~hybrid_correct))
    introduced_false_positives = int(np.sum((truth == 0) & xgb_correct & ~hybrid_correct))
    if corrected_false_negatives + corrected_false_positives != corrected:
        raise ValueError("Class-specific correction counts do not match total corrections.")
    if introduced_false_negatives + introduced_false_positives != regressed:
        raise ValueError("Class-specific introduced errors do not match total regressions.")
    return {
        "n": len(truth), "xgboost": _metrics(truth, xgb), "hybrid": _metrics(truth, hybrid),
        "paired": {
            "n": len(truth), "both_correct": int(np.sum(xgb_correct & hybrid_correct)),
            "xgboost_only_correct": regressed, "hybrid_only_correct": corrected,
            "both_incorrect": int(np.sum(~xgb_correct & ~hybrid_correct)),
            "corrected_xgboost_errors": corrected, "introduced_errors": regressed,
            "corrected_false_negatives": corrected_false_negatives,
            "corrected_false_positives": corrected_false_positives,
            "introduced_false_negatives": introduced_false_negatives,
            "introduced_false_positives": introduced_false_positives,
            "net_false_negative_reduction": corrected_false_negatives - introduced_false_negatives,
            "net_false_positive_reduction": corrected_false_positives - introduced_false_positives,
            "net_additional_correct": corrected - regressed,
            "accuracy_change": (corrected - regressed) / len(truth),
            "exact_mcnemar_pvalue": float(binomtest(min(corrected, regressed), discordant, 0.5).pvalue)
            if discordant else 1.0,
        },
        "error_recovery": {
            "false_negative": _rate(corrected_false_negatives, int(np.sum(false_negative))),
            "false_positive": _rate(corrected_false_positives, int(np.sum(false_positive))),
            "all_xgboost_errors": _rate(corrected, int(np.sum(~xgb_correct))),
            "regression_among_xgboost_correct": _rate(regressed, int(np.sum(xgb_correct))),
        },
    }


def build_hybrid_report(full_test: Any, selected: Any, outcomes: Any, *,
                        threshold: float = 0.5, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Report two paired populations, retaining failed/missing routes as fallback.

    ``full_test`` requires row_id/y_true/p_xgb. ``selected`` requires sample_id,
    row_id/text_sha256/p_xgb/source_split='test'. Optional selected labels and
    provenance must match the full table. ``outcomes`` uses ledger-style
    sample_id/status/score records. Only succeeded outcomes replace XGBoost.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or threshold != 0.5:
        raise ValueError("The prespecified classification threshold must remain 0.50.")
    full_records = _records(full_test, "full_test")
    selected_records = _records(selected, "selected")
    outcome_records = _records(outcomes, "outcomes")
    if not full_records or not selected_records or len(selected_records) > 5000:
        raise ValueError("Require a nonempty test population and between one and 5,000 selected cases.")

    full_by_row = {}
    truth, xgb = [], []
    for index, row in enumerate(full_records):
        identifier = _row_id(row.get("row_id"))
        if identifier in full_by_row:
            raise ValueError("Duplicate row_id in full test table.")
        label = row.get("y_true")
        if isinstance(label, bool) or not isinstance(label, Real) or label not in (0, 1):
            raise ValueError("y_true must be numeric 0 or 1.")
        if "source_split" in row and row["source_split"] != "test":
            raise ValueError("The full comparison must contain test rows only.")
        probability = _probability(row.get("p_xgb"), "p_xgb")
        full_by_row[identifier] = (index, row)
        truth.append(int(label))
        xgb.append(probability)

    selected_by_sample = {}
    selected_indexes = []
    seen_hashes, seen_rows = set(), set()
    for row in selected_records:
        sample = _identifier(row.get("sample_id"), "sample_id")
        identifier = _row_id(row.get("row_id"))
        digest = _checksum(row.get("text_sha256"))
        if sample in selected_by_sample or identifier in seen_rows or digest in seen_hashes:
            raise ValueError("Selected samples, row IDs, and text checksums must be unique.")
        if identifier not in full_by_row or row.get("source_split") != "test":
            raise ValueError("Every selected case must belong to the full test population.")
        index, source = full_by_row[identifier]
        probability = _probability(row.get("p_xgb"), "selected p_xgb")
        if not math.isclose(probability, xgb[index], rel_tol=0, abs_tol=1e-15):
            raise ValueError("Selected probability does not match its full-test row.")
        if "y_true" in row and (isinstance(row["y_true"], bool) or row["y_true"] != truth[index]):
            raise ValueError("Selected label does not match its full-test row.")
        for field in ("text_sha256", "dataset_sha256", "baseline_fingerprint", "sample_id"):
            if field in source and field in row and source[field] != row[field]:
                raise ValueError(f"Selected {field} provenance differs from its full-test row.")
        selected_by_sample[sample] = index
        selected_indexes.append(index)
        seen_rows.add(identifier)
        seen_hashes.add(digest)

    outcome_by_sample = {}
    status_counts = {status: 0 for status in ("succeeded", "failed", "pending", "not_attempted")}
    hybrid = np.asarray(xgb, dtype=float).copy()
    for row in outcome_records:
        sample = _identifier(row.get("sample_id"), "sample_id")
        if sample not in selected_by_sample or sample in outcome_by_sample:
            raise ValueError("Outcome IDs must be unique members of the locked selected cohort.")
        status = row.get("status")
        if status not in status_counts:
            raise ValueError("Outcome status must be succeeded, failed, pending, or not_attempted.")
        score = row.get("score")
        missing_score = score is None or (isinstance(score, Real) and math.isnan(float(score)))
        if status == "succeeded":
            if missing_score:
                raise ValueError("A succeeded outcome must have a valid score.")
            hybrid[selected_by_sample[sample]] = _probability(score, "Jev score")
        elif not missing_score:
            raise ValueError("An unresolved outcome cannot also carry a stale Jev score.")
        outcome_by_sample[sample] = status
        status_counts[status] += 1
    status_counts["not_attempted"] += len(selected_records) - len(outcome_by_sample)

    truth_array = np.asarray(truth, dtype=int)
    xgb_array = np.asarray(xgb, dtype=float)
    selected_indexes = np.asarray(selected_indexes, dtype=int)
    successes = status_counts["succeeded"]
    fallback = len(selected_records) - successes
    return {
        "schema_version": 1,
        "analysis_population": "probability_selected_test_cases_including_correct_and_incorrect_xgboost_predictions",
        "primary_population": "selected_subset",
        "coverage_status": "complete_jev_coverage" if not fallback else "partial_jev_coverage_with_xgboost_fallback",
        "threshold": 0.5,
        "routing": {
            "selected_n": len(selected_records), "full_test_n": len(full_records),
            "jev_success_n": successes, "xgboost_fallback_selected_n": fallback,
            "unrouted_full_test_n": len(full_records) - len(selected_records),
            "selected_fraction_of_full_test": len(selected_records) / len(full_records),
            "successful_fraction_of_full_test": successes / len(full_records),
            "outcome_status_counts": status_counts,
            "policy": "For the exact selected manifest, use a successful Jev score; otherwise retain XGBoost. All unselected test rows retain XGBoost.",
        },
        "selected_subset": _population(truth_array[selected_indexes], xgb_array[selected_indexes], hybrid[selected_indexes]),
        "full_test_partial_policy": _population(truth_array, xgb_array, hybrid),
        "metadata": _safe_metadata(metadata or {}),
        "limitations": [
            "The selected subset contains probability-eligible cases, not gold-label-selected errors. Corrections and newly introduced errors are both counted.",
            "The full-test result applies only to partial routing of the exact selected manifest; it does not route every case in the critical interval unless the manifest includes them all.",
            "Failed, pending, and unattempted selected requests fall back to XGBoost and remain in every operational denominator.",
            "The original notebook and earlier study inspected the test split; this reused-test comparison is exploratory. Interval selection and all policy choices must use validation data only.",
            "Hybrid ROC-AUC, AP, Brier score, and log loss use piecewise XGBoost/Jev scores. Those scores are not jointly calibrated, so these metrics describe this policy rather than calibrated clinical probabilities.",
            "jev-latest is a mutable provider alias without a recorded immutable backend revision; later backend changes can affect reproducibility.",
            "Social-media labels do not establish clinical validity or deployment performance. Confidence intervals are conditional on this selected evaluation population.",
        ],
    }


def _number(value: Any, *, percentage: bool = False) -> str:
    if value is None:
        return "Not estimable"
    return f"{value:.2%}" if percentage else f"{value:.6f}"


def render_hybrid_markdown(report: Mapping[str, Any]) -> str:
    routing = report["routing"]
    lines = [
        "# Probability-routed Jev hybrid versus XGBoost", "",
        f"Selected test cases: {routing['selected_n']}; successful Jev responses: {routing['jev_success_n']}; selected XGBoost fallbacks: {routing['xgboost_fallback_selected_n']}.", "",
        "Both models use threshold 0.50. Every selected case remains in the primary comparison, including failed or missing requests.", "",
    ]
    for key, title in (("selected_subset", "Primary: identical selected test cases"),
                       ("full_test_partial_policy", "Full test: partial routing of the exact selected manifest")):
        population = report[key]
        lines.extend([f"## {title}", "", f"Population n={population['n']}.", "",
                      "| Metric | XGBoost | Hybrid |", "| --- | ---: | ---: |"])
        for metric, label in (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                              ("precision_positive", "Positive precision"), ("recall_positive", "Positive recall"),
                              ("specificity", "Specificity"), ("f1_positive", "Positive F1"),
                              ("f1_macro", "Macro F1"), ("roc_auc", "ROC-AUC"),
                              ("average_precision", "Average precision"), ("brier_score", "Brier score"),
                              ("log_loss", "Log loss")):
            percentage = metric in {"accuracy", "balanced_accuracy", "precision_positive", "recall_positive", "specificity"}
            lines.append(f"| {label} | {_number(population['xgboost'][metric], percentage=percentage)} | {_number(population['hybrid'][metric], percentage=percentage)} |")
        paired = population["paired"]
        lines.extend(["", f"Corrected XGBoost errors: {paired['corrected_xgboost_errors']}; newly introduced errors: {paired['introduced_errors']}; net additional correct cases: {paired['net_additional_correct']}; accuracy change: {paired['accuracy_change']:+.2%}.",
                      f"Two-sided exact McNemar p-value: {paired['exact_mcnemar_pvalue']:.6g}.", "",
                      "Corrections must be considered alongside newly introduced errors within each class:", "",
                      "| Error class | XGBoost errors | Corrected | Newly introduced | Hybrid errors | Net reduction |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for error, label in (("false_negative", "Missed suicide cases (FN)"),
                             ("false_positive", "False alarms (FP)")):
            confusion_key = "fn" if error == "false_negative" else "fp"
            lines.append(
                f"| {label} | {population['xgboost']['confusion'][confusion_key]} | "
                f"{paired[f'corrected_{error}s']} | {paired[f'introduced_{error}s']} | "
                f"{population['hybrid']['confusion'][confusion_key]} | {paired[f'net_{error}_reduction']:+d} |")
        lines.extend(["", "| Error outcome | Count / eligible | Rate | 95% Wilson interval |",
                      "| --- | ---: | ---: | --- |"])
        for error, label in (("false_negative", "Missed-suicide recovery"), ("false_positive", "False-alarm correction"),
                             ("regression_among_xgboost_correct", "Regression among XGBoost-correct cases")):
            item = population["error_recovery"][error]
            interval = item["wilson_95"]
            interval_label = f"{interval[0]:.2%}–{interval[1]:.2%}" if interval else "Not estimable"
            lines.append(f"| {label} | {item['corrected']}/{item['n']} | {_number(item['rate'], percentage=True)} | {interval_label} |")
        lines.append("")
    lines.extend(["## Routing outcomes", "", "| Outcome | Count |", "| --- | ---: |"])
    lines.extend(f"| {status} | {count} |" for status, count in routing["outcome_status_counts"].items())
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def write_hybrid_report(report: Mapping[str, Any], directory: str | Path) -> dict[str, Path]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "hybrid_report.json"
    markdown_path = destination / "hybrid_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(render_hybrid_markdown(report))
    return {"json": json_path, "markdown": markdown_path}
