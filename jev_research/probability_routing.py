"""Select a validation-defined probability interval and label-blind test routing.

The interval is selected using validation outcomes only. Test labels are neither
read nor used when choosing or ranking routed texts. The caller must separately
verify validation/test disjointness before invoking these single-frame helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import numbers
from typing import Any, Mapping

import numpy as np
import pandas as pd


DIAGNOSTIC_MIN_SUPPORT = 100
PRIMARY_THRESHOLD = 0.5


class InsufficientRoutingSupport(ValueError):
    """The fixed interval/protocol cannot supply the requested exact cohort."""


@dataclass
class ProbabilityIntervalResult:
    interval: dict[str, Any]
    diagnostics: pd.DataFrame
    candidates: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class RoutingCohortResult:
    cohort: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def manifest(self) -> pd.DataFrame:
        """Return whitelisted provenance/outcomes without any raw text."""
        columns = [
            "sample_id", "row_id", "dataset_sha256", "baseline_fingerprint",
            "text_sha256", "source_split", "p_xgb", "y_xgb", "y_true",
            "error_type", "sample_priority", "cohort_order",
        ]
        return self.cohort[[name for name in columns if name in self.cohort]].copy()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    if "p_xgb" not in frame or frame.empty:
        raise ValueError("A nonempty p_xgb probability column is required.")
    if not frame.p_xgb.map(lambda value: isinstance(value, numbers.Real)
                           and not isinstance(value, (bool, np.bool_))).all():
        raise ValueError("XGBoost probabilities must be numeric, not boolean or string.")
    values = frame.p_xgb.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("XGBoost probabilities must be finite and in [0,1].")
    return values


def _fingerprint_column(frame: pd.DataFrame, column: str, *, required: bool) -> str | None:
    if column not in frame:
        if required:
            raise ValueError(f"Missing routing provenance: {column}.")
        return None
    values = frame[column].unique()
    if (len(values) != 1 or not isinstance(values[0], str) or len(values[0]) != 64
            or any(char not in "0123456789abcdef" for char in values[0])):
        raise ValueError(f"Ambiguous or malformed {column} provenance.")
    return values[0]


def _wilson(errors: int, n: int) -> list[float | None]:
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = errors / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _diagnostic_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lower": float(row["lower"]), "upper": float(row["upper"]),
        "upper_inclusive": bool(row["upper_inclusive"]),
        "n": int(row["n"]), "errors": int(row["errors"]),
        "error_rate": None if not row["n"] else int(row["errors"]) / int(row["n"]),
        "wilson_95": _wilson(int(row["errors"]), int(row["n"])),
    }


def select_probability_interval(
    validation_df: pd.DataFrame, *, target_size: int = 5000,
    test_size: int, coverage_buffer: float = 1.10, bin_width: float = 0.05,
) -> ProbabilityIntervalResult:
    """Find the highest validation-error-rate feasible contiguous grid window.

    Minimum support is ``ceil(buffer * target_size * N_validation / N_test)``.
    All grid-aligned windows are compared; ties prefer a narrower window and
    then a lower endpoint. Neither test probabilities nor test labels are inputs.
    The selected interval is never expanded automatically after test inspection.

    Diagnostic bins include empty/low-support bins, with a separate supported
    peak using a fixed minimum of 100 validation observations. Intervals use
    [lower,upper), except that a final upper endpoint of 1 includes probability 1.
    """
    target_size = _positive_integer(target_size, "target_size")
    test_size = _positive_integer(test_size, "test_size")
    if target_size > test_size:
        raise InsufficientRoutingSupport("target_size exceeds the entire test population.")
    if (isinstance(coverage_buffer, bool) or not isinstance(coverage_buffer, numbers.Real)
            or not math.isfinite(coverage_buffer) or coverage_buffer < 1):
        raise ValueError("coverage_buffer must be finite and at least 1.")
    if (isinstance(bin_width, bool) or not isinstance(bin_width, numbers.Real)
            or not math.isfinite(bin_width) or not 0 < bin_width <= 1):
        raise ValueError("bin_width must divide the unit probability interval.")
    n_bins = round(1 / bin_width)
    if not math.isclose(n_bins * bin_width, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("bin_width must divide the unit probability interval.")
    probabilities = _probabilities(validation_df)
    if "y_true" not in validation_df:
        raise ValueError("Validation gold labels are required to identify errors.")
    if not validation_df.y_true.map(lambda value: isinstance(value, numbers.Real)
                                   and not isinstance(value, (bool, np.bool_))
                                   and value in (0, 1)).all():
        raise ValueError("Validation labels must be binary numeric values.")
    predictions = (probabilities >= PRIMARY_THRESHOLD).astype(int)
    if "y_xgb" in validation_df and not np.array_equal(predictions, validation_df.y_xgb.to_numpy()):
        raise ValueError("Validation predictions disagree with the fixed 0.50 threshold.")
    if "source_split" in validation_df and not validation_df.source_split.eq("validation").all():
        raise ValueError("Only validation rows may select the routing interval.")
    provenance = {
        name: _fingerprint_column(validation_df, name, required=False)
        for name in ("dataset_sha256", "baseline_fingerprint")
    }
    provenance = {name: value for name, value in provenance.items() if value is not None}
    errors = predictions != validation_df.y_true.to_numpy(dtype=int)
    # searchsorted preserves the stated interval convention at exact boundaries.
    edges = np.array([index / n_bins for index in range(n_bins + 1)])
    bins = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    error_counts = np.bincount(bins, weights=errors.astype(int), minlength=n_bins).astype(int)
    diagnostic_records = []
    for index in range(n_bins):
        record = _diagnostic_record({
            "lower": edges[index], "upper": edges[index + 1],
            "upper_inclusive": index == n_bins - 1,
            "n": counts[index], "errors": error_counts[index],
        })
        record["supported_peak_eligible"] = record["n"] >= DIAGNOSTIC_MIN_SUPPORT
        diagnostic_records.append(record)
    # Treat the declared decimal buffer exactly, so e.g. ceil(1.1 * 400)
    # remains 440 instead of becoming 441 through binary rounding noise.
    support = Fraction(str(float(coverage_buffer))) * target_size * len(validation_df) / test_size
    minimum = math.ceil(support)
    if minimum > len(validation_df):
        raise InsufficientRoutingSupport("The fixed coverage buffer requires more validation rows than exist.")
    cumulative_n = np.r_[0, np.cumsum(counts)]
    cumulative_errors = np.r_[0, np.cumsum(error_counts)]
    candidates = []
    eligible = []
    for lower_index in range(n_bins):
        for upper_index in range(lower_index + 1, n_bins + 1):
            n = int(cumulative_n[upper_index] - cumulative_n[lower_index])
            count = int(cumulative_errors[upper_index] - cumulative_errors[lower_index])
            row = {
                "lower": float(edges[lower_index]), "upper": float(edges[upper_index]),
                "upper_inclusive": upper_index == n_bins,
                "n": n, "errors": count, "error_rate": count / n if n else None,
                "width": (upper_index - lower_index) / n_bins,
                "minimum_support": minimum, "support_eligible": n >= minimum,
            }
            candidates.append(row)
            if row["support_eligible"]:
                # Exact rational ordering avoids manufactured floating-point ties.
                eligible.append(((-Fraction(count, n), upper_index - lower_index, lower_index), row))
    if not eligible:
        raise InsufficientRoutingSupport("No validation-supported probability window is available.")
    chosen = min(eligible, key=lambda item: item[0])[1]
    nonempty = [row for row in diagnostic_records if row["n"]]
    supported = [row for row in nonempty if row["supported_peak_eligible"]]
    rank_rate = lambda row: (-Fraction(row["errors"], row["n"]), row["lower"])
    peak = min(supported, key=rank_rate) if supported else None
    raw_peak = min(nonempty, key=rank_rate)
    count_peak = min(nonempty, key=lambda row: (-row["errors"], row["lower"]))
    interval = {
        "lower": chosen["lower"], "upper": chosen["upper"],
        "upper_inclusive": chosen["upper_inclusive"],
        "target_size": target_size, "test_population_size": test_size,
        **provenance,
    }
    metadata = {
        "selection_schema_version": 1, "selection_source": "validation_only",
        "validation_size": len(validation_df), "test_population_size": test_size,
        "target_size": target_size, "coverage_buffer": float(coverage_buffer),
        "bin_width": float(bin_width), "minimum_validation_support": minimum,
        "diagnostic_min_support": DIAGNOSTIC_MIN_SUPPORT,
        "highest_error_rate_bin": _diagnostic_record(peak) if peak else None,
        "raw_highest_error_rate_bin": _diagnostic_record(raw_peak),
        "highest_error_count_bin": _diagnostic_record(count_peak),
        "interval": interval,
        "selected_validation": _diagnostic_record(chosen),
        "candidate_windows": len(candidates),
        "supported_candidate_windows": len(eligible),
        "selection_objective": "Maximize validation error rate, subject to fixed minimum validation support.",
        "tie_breaking": "Narrower grid window, then lower endpoint; exact rational error rates.",
        "interval_convention": "[lower,upper), with probability 1 included when upper=1",
        "test_capacity_policy": "Fail if fewer than target_size test probabilities fall inside the frozen interval; never auto-expand.",
        **provenance,
    }
    return ProbabilityIntervalResult(interval, pd.DataFrame(diagnostic_records), pd.DataFrame(candidates), metadata)


def select_routing_cohort(
    test_frame: pd.DataFrame, interval: Mapping[str, Any] | ProbabilityIntervalResult,
    *, target_size: int = 5000, seed: int = 42,
) -> RoutingCohortResult:
    """Hash-rank exactly target_size test texts inside the frozen interval.

    Test labels, predictions, and correctness flags are deliberately not read.
    Such columns, if present, are merely carried through for later evaluation.
    This differs from the earlier oracle-selected XGBoost-error-only experiment.
    """
    target_size = _positive_integer(target_size, "target_size")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    if isinstance(interval, ProbabilityIntervalResult):
        interval = interval.interval
    if not isinstance(interval, Mapping):
        raise ValueError("Provide the frozen validation-selected interval.")
    try:
        lower, upper = interval["lower"], interval["upper"]
    except KeyError as exc:
        raise ValueError("An interval needs lower and upper endpoints.") from exc
    if (any(isinstance(value, bool) or not isinstance(value, numbers.Real) for value in (lower, upper))
            or not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper <= 1):
        raise ValueError("Interval endpoints must satisfy 0<=lower<upper<=1.")
    lower, upper = float(lower), float(upper)
    inclusive = upper == 1.0
    if "upper_inclusive" in interval:
        declared_inclusive = interval["upper_inclusive"]
        if not isinstance(declared_inclusive, (bool, np.bool_)) or bool(declared_inclusive) != inclusive:
            raise ValueError("Only an upper endpoint of 1 may be inclusive.")
    if "target_size" in interval and interval["target_size"] != target_size:
        raise ValueError("The routing size differs from the frozen interval-selection protocol.")
    if "test_population_size" in interval and interval["test_population_size"] != len(test_frame):
        raise ValueError("The test population differs from the frozen interval-selection protocol.")
    required = {"row_id", "text_sha256", "source_split", "dataset_sha256", "baseline_fingerprint"}
    if required - set(test_frame.columns):
        raise ValueError(f"Missing test routing provenance: {sorted(required - set(test_frame.columns))}")
    probabilities = _probabilities(test_frame)
    if not test_frame.source_split.eq("test").all():
        raise ValueError("Only original test rows may be routed.")
    if test_frame.row_id.duplicated().any() or not test_frame.row_id.map(
        lambda value: isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("Original test row IDs must be distinct integers.")
    if test_frame.text_sha256.duplicated().any() or not test_frame.text_sha256.map(
        lambda value: isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    ).all():
        raise ValueError("Original test text checksums must be distinct SHA-256 values.")
    provenance = {
        name: _fingerprint_column(test_frame, name, required=True)
        for name in ("dataset_sha256", "baseline_fingerprint")
    }
    for column, value in provenance.items():
        if column in interval and interval[column] != value:
            raise ValueError(f"The test source disagrees with interval {column} provenance.")
    if "text" in test_frame:
        if not test_frame.text.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("Original test texts must be nonempty strings.")
        if not test_frame.text.map(_hash).equals(test_frame.text_sha256):
            raise ValueError("Test text checksums do not match; no routing cohort was produced.")
    mask = (probabilities >= lower) & ((probabilities <= upper) if inclusive else (probabilities < upper))
    eligible = test_frame.loc[mask].copy()
    if len(eligible) < target_size:
        raise InsufficientRoutingSupport(
            f"Frozen interval contains {len(eligible)} test texts, fewer than requested {target_size}; automatic expansion is forbidden."
        )
    eligible["sample_id"] = eligible.apply(
        lambda row: _hash(f"{row['dataset_sha256']}:{int(row['row_id'])}:{row['text_sha256']}"), axis=1,
    )
    eligible["sample_priority"] = eligible.apply(
        lambda row: _hash(f"{int(seed)}:probability_hybrid:{row['dataset_sha256']}:"
                          f"{row['baseline_fingerprint']}:{int(row['row_id'])}:{row['text_sha256']}"), axis=1,
    )
    selected = eligible.sort_values(["sample_priority", "sample_id"], kind="mergesort").head(target_size).reset_index(drop=True)
    selected["cohort_order"] = np.arange(target_size, dtype=int)
    # Bind selection to only label-blind fields; changing test labels cannot
    # affect the selected identities, order, or selection fingerprint.
    fingerprint_fields = ["sample_id", "row_id", "text_sha256", "p_xgb", "sample_priority", "cohort_order"]
    selection_identity = {
        "interval": {"lower": lower, "upper": upper, "upper_inclusive": inclusive},
        "seed": int(seed), "target_size": target_size,
        "records": selected[fingerprint_fields].to_dict(orient="records"),
        **provenance,
    }
    fingerprint = _hash(json.dumps(selection_identity, sort_keys=True, separators=(",", ":"), allow_nan=False))
    metadata = {
        "routing_schema_version": 1, "sampling_seed": int(seed),
        "target_size": target_size, "selected_count": len(selected),
        "test_population_size": len(test_frame), "eligible_test_count": len(eligible),
        "unqueried_interval_count": len(eligible) - target_size,
        "full_test_routing_fraction": target_size / len(test_frame),
        "interval_routing_fraction": target_size / len(eligible),
        "interval": {"lower": lower, "upper": upper, "upper_inclusive": inclusive},
        "selection_sha256": fingerprint,
        "sampling_rule": "Hash-ranked test texts inside the frozen validation interval, without gold-label or correctness filtering.",
        "hybrid_rule": "Use Jev only on the selected cohort; retain XGBoost for every other test row, including unqueried interval members.",
        "test_labels_used_for_selection": False,
        **provenance,
    }
    return RoutingCohortResult(selected, metadata)
