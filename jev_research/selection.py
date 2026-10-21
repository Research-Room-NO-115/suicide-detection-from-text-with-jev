"""Deterministic, leakage-aware selection of at most 100 XGBoost errors."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd


ERROR_TYPES = ("false_negative", "false_positive")
MANIFEST_COLUMNS = (
    "sample_id", "row_id", "text_sha256", "y_true", "p_xgb", "y_xgb",
    "error_type", "source_split", "dataset_sha256", "baseline_fingerprint",
    "jev_partition", "jev_role", "sample_priority", "cohort_order",
)


class InsufficientErrorsError(ValueError):
    """The requested exact cohort cannot be built without weakening the protocol."""


@dataclass
class SelectionResult:
    cohort: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def manifest(self) -> pd.DataFrame:
        """Public provenance only: never raw or cleaned social-media text."""
        return selection_manifest(self.cohort)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sources(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    required = {
        "row_id", "text", "text_sha256", "y_true", "p_xgb", "y_xgb", "error_type",
        "source_split", "dataset_sha256", "baseline_fingerprint",
    }
    for name, frame in (("validation", validation), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing baseline columns in {name}: {sorted(missing)}")
        if frame.empty or frame["row_id"].duplicated().any() or frame["text_sha256"].duplicated().any():
            raise ValueError(f"Baseline {name} rows and text hashes must be nonempty and unique.")
        if not frame["source_split"].eq(name).all():
            raise ValueError(f"Only original {name} rows may populate its selection source.")
        if not frame["y_true"].isin([0, 1]).all() or not frame["y_xgb"].isin([0, 1]).all():
            raise ValueError("XGBoost labels and predictions must be binary.")
        if not all(isinstance(value, (int, np.integer)) for value in frame["row_id"]):
            raise ValueError("Original dataset row IDs must be integers.")
        probabilities = frame["p_xgb"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("XGBoost probabilities must be finite and in [0,1].")
        predicted = (probabilities >= 0.5).astype(int)
        if not np.array_equal(predicted, frame["y_xgb"].to_numpy()):
            raise ValueError("Selection requires the fixed XGBoost threshold 0.50.")
        expected = np.where(
            (frame["y_true"] == 1) & (predicted == 0), "false_negative",
            np.where((frame["y_true"] == 0) & (predicted == 1), "false_positive", "correct"),
        )
        if not np.array_equal(expected, frame["error_type"].to_numpy()):
            raise ValueError("XGBoost error_type disagrees with its labels/predictions.")
        if not frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("Every selectable source row needs its nonempty original text.")
        if not frame["text"].map(_digest).equals(frame["text_sha256"]):
            raise ValueError("Source text hashes do not match the original text.")
        for column in ("dataset_sha256", "baseline_fingerprint"):
            values = frame[column].unique()
            if (len(values) != 1 or not isinstance(values[0], str) or len(values[0]) != 64
                    or any(character not in "0123456789abcdef" for character in values[0])):
                raise ValueError(f"The baseline has ambiguous {column} provenance.")
    if set(validation["row_id"]) & set(test["row_id"]):
        raise ValueError("Validation and test row IDs overlap; Jev tuning would leak evaluation data.")
    if set(validation["text_sha256"]) & set(test["text_sha256"]):
        raise ValueError("Validation and test raw texts overlap.")
    if "cleaned_text" in validation and "cleaned_text" in test:
        if set(validation["cleaned_text"]) & set(test["cleaned_text"]):
            raise ValueError("Validation and test cleaned texts overlap.")
    for column in ("dataset_sha256", "baseline_fingerprint"):
        if validation[column].iloc[0] != test[column].iloc[0]:
            raise ValueError(f"Validation and test belong to different {column} values.")


def _stratified_take(
    pool: pd.DataFrame, size: int, *, seed: int, tag: str,
    minimum_each: int = 1, leave_each: int = 0,
) -> pd.DataFrame:
    counts = {error: int(pool["error_type"].eq(error).sum()) for error in ERROR_TYPES}
    if size < 2 * minimum_each or any(count < minimum_each + leave_each for count in counts.values()):
        raise InsufficientErrorsError(
            f"{tag} needs both false negatives and false positives "
            f"(at least {minimum_each + leave_each} each); available: {counts}."
        )
    if len(pool) < size + 2 * leave_each:
        raise InsufficientErrorsError(f"{tag} requests {size} errors, but only {len(pool)} are available.")
    # The study deliberately balances error strata, rather than reproducing the
    # baseline's natural error mixture. An odd requested size gives FN one extra
    # row. Never silently relax a quota because an error stratum is too small.
    allocations = {"false_negative": (size + 1) // 2, "false_positive": size // 2}
    for error, requested in allocations.items():
        if requested < minimum_each or counts[error] < requested + leave_each:
            raise InsufficientErrorsError(
                f"{tag} requires {requested} {error} rows and {leave_each} reserved rows; "
                f"only {counts[error]} are available. Balanced quotas are not relaxed."
            )
    ranked = pool.copy()
    ranked["sample_priority"] = ranked.apply(
        lambda row: _digest(
            f"{seed}:{tag}:{row['dataset_sha256']}:{row['baseline_fingerprint']}:"
            f"{int(row['row_id'])}:{row['text_sha256']}"
        ), axis=1,
    )
    selected = []
    for error in ERROR_TYPES:
        candidates = ranked.loc[ranked["error_type"].eq(error)].sort_values(
            ["sample_priority", "text_sha256", "row_id"], kind="mergesort"
        )
        selected.append(candidates.head(allocations[error]))
    return pd.concat(selected, ignore_index=True).sort_values(
        ["sample_priority", "text_sha256", "row_id"], kind="mergesort"
    ).reset_index(drop=True)


def partition_tuning_rows(
    cohort: pd.DataFrame, *, train_size: int = 40, validation_size: int = 20, seed: int = 42,
) -> pd.DataFrame:
    """Split only original-XGBoost-validation errors into Jev train/validation."""
    result = cohort.copy()
    tuning = result.loc[result["jev_partition"].eq("tuning")]
    if train_size + validation_size != len(tuning) or train_size < 2 or validation_size < 2:
        raise ValueError("Jev train and validation sizes must partition tuning rows, with both labels each.")
    if not tuning["source_split"].eq("validation").all():
        raise ValueError("Jev training/validation rows must come from the XGBoost validation split.")
    training = _stratified_take(
        tuning, train_size, seed=seed, tag="jev_training", minimum_each=1, leave_each=1
    )
    training_ids = set(training["sample_id"])
    result["jev_role"] = np.where(
        result["jev_partition"].eq("evaluation"), "evaluation",
        np.where(result["sample_id"].isin(training_ids), "train", "validation"),
    )
    for role, count in (("train", train_size), ("validation", validation_size)):
        selected = result.loc[result["jev_role"].eq(role)]
        if len(selected) != count or set(selected["error_type"]) != set(ERROR_TYPES):
            raise InsufficientErrorsError(f"Jev {role} must retain both labels with exactly {count} rows.")
    return result


def selection_manifest(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed text-free public manifest schema."""
    missing = set(MANIFEST_COLUMNS) - set(cohort.columns)
    if missing:
        raise ValueError(f"Incomplete cohort manifest columns: {sorted(missing)}")
    return cohort.loc[:, MANIFEST_COLUMNS].copy()


def select_error_cohort(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_samples: int = 100,
    tuning_size: int = 60,
    evaluation_size: int = 40,
    seed: int = 42,
) -> SelectionResult:
    """Choose exactly 60 tuning + 40 evaluation errors by default.

    Tuning uses XGBoost validation errors only; evaluation uses XGBoost test
    errors only. The default is an artificial 50/50 mixture: 30 FN + 30 FP for
    tuning, and 20 FN + 20 FP for evaluation. This conditional error study cannot
    estimate an operational cascade's accuracy or its natural error mixture.
    """
    integers = (max_samples, tuning_size, evaluation_size, seed)
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in integers):
        raise ValueError("Sample limits, sizes, and seed must be integers.")
    if max_samples < 1 or max_samples > 100 or tuning_size + evaluation_size > max_samples:
        raise ValueError("The study permits at most 100 unique XGBoost error samples.")
    if tuning_size < 4 or evaluation_size < 2:
        raise ValueError("Tuning needs at least four rows and evaluation at least two, retaining both labels.")
    _validate_sources(validation_df, test_df)
    pools = {
        "tuning": validation_df.loc[validation_df["error_type"].isin(ERROR_TYPES)].copy(),
        "evaluation": test_df.loc[test_df["error_type"].isin(ERROR_TYPES)].copy(),
    }
    selected = []
    for partition, size, minimum in (("tuning", tuning_size, 2), ("evaluation", evaluation_size, 1)):
        rows = _stratified_take(pools[partition], size, seed=seed, tag=partition, minimum_each=minimum)
        rows["jev_partition"] = partition
        rows["sample_id"] = rows.apply(
            lambda row: _digest(f"{row['dataset_sha256']}:{int(row['row_id'])}:{row['text_sha256']}"),
            axis=1,
        )
        selected.append(rows)
    cohort = pd.concat(selected, ignore_index=True)
    training_size = max(2, min(tuning_size - 2, math.floor(tuning_size * 2 / 3)))
    cohort = partition_tuning_rows(
        cohort, train_size=training_size, validation_size=tuning_size - training_size, seed=seed
    )
    cohort["cohort_order"] = np.arange(len(cohort), dtype=int)
    manifest = selection_manifest(cohort)
    manifest_hash = _digest(manifest.to_json(orient="records", double_precision=15))
    metadata = {
        "selection_schema_version": 1,
        "seed": int(seed),
        "max_unique_samples": int(max_samples),
        "selected_samples": len(cohort),
        "sizes": {"tuning": int(tuning_size), "evaluation": int(evaluation_size)},
        "jev_role_counts": cohort["jev_role"].value_counts().to_dict(),
        "available_error_counts": {
            name: {error: int(pool["error_type"].eq(error).sum()) for error in ERROR_TYPES}
            for name, pool in pools.items()
        },
        "selected_error_counts": {
            name: {error: int(cohort.loc[cohort["jev_partition"].eq(name), "error_type"].eq(error).sum())
                   for error in ERROR_TYPES}
            for name in ("tuning", "evaluation")
        },
        "dataset_sha256": cohort["dataset_sha256"].iloc[0],
        "baseline_fingerprint": cohort["baseline_fingerprint"].iloc[0],
        "manifest_sha256": manifest_hash,
        "sampling": "Hash-ranked, balanced FN/FP quotas; artificial 50/50 error mixture fixed before Jev calls.",
        "limitations": [
            "Gold-label-selected XGBoost errors: conditional error correction, not deployment accuracy.",
            "Both false negatives and false positives are retained to avoid all-positive fine-tuning.",
            "Error strata are deliberately balanced; the combined correction rate is not the baseline's natural error-mixture rate.",
            "Jev tuning derives only from XGBoost validation; test errors are evaluation only.",
            "Small held-out error sample gives wide uncertainty; report strata-specific denominators.",
        ],
    }
    return SelectionResult(cohort, metadata)


def validate_error_cohort(
    cohort: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_samples: int = 100,
    tuning_size: int = 60,
    evaluation_size: int = 40,
    seed: int = 42,
) -> None:
    """Reject mutated/reassigned/reordered selections before resuming paid work.

    Text-free persisted manifests are accepted. When text columns are supplied,
    they must also exactly match their original source rows.
    """
    expected = select_error_cohort(
        validation_df, test_df, max_samples=max_samples,
        tuning_size=tuning_size, evaluation_size=evaluation_size, seed=seed,
    ).cohort
    missing = set(MANIFEST_COLUMNS) - set(cohort.columns)
    if missing or len(cohort) != len(expected) or cohort["sample_id"].duplicated().any():
        raise ValueError("Persisted cohort is incomplete or contains duplicate samples.")
    for column in MANIFEST_COLUMNS:
        left = cohort[column].reset_index(drop=True)
        right = expected[column].reset_index(drop=True)
        if column == "p_xgb":
            same = np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=0, atol=1e-14)
        else:
            same = left.equals(right)
        if not same:
            raise ValueError(f"Persisted cohort differs from deterministic selection: {column}")
    for column in ("text", "cleaned_text"):
        if column in cohort and not cohort[column].reset_index(drop=True).equals(expected[column].reset_index(drop=True)):
            raise ValueError(f"Persisted cohort {column} differs from the original source.")
