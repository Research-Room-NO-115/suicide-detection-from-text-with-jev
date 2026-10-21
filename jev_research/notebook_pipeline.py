"""Replay the notebook's centralized baseline without its 360-client sweep.

Only the supplied dataset and ignored artifact directory are written. Raw text is
retained in local caches so that later Jev requests can use the original text;
these caches must never be committed to Git.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "assets/notebooks/federated_xgboost_distribution_iid_non_iid_executed.ipynb"
)
CACHE_SCHEMA_VERSION = 1
PRIMARY_THRESHOLD = 0.50
SPLIT_SEED = 42


@dataclass
class BaselineResult:
    frames: dict[str, pd.DataFrame]
    metadata: dict[str, Any]
    cache_dir: Path
    from_cache: bool = False


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_notebook_definitions(path: Path) -> dict[str, Any]:
    """Execute original configuration/definitions, excluding experiments and I/O."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "jev_research.notebook_baseline"}
    # These cells implement the source pipeline; calls and top-level experiment
    # loops are excluded. No source definition or hyperparameter is rewritten.
    for index in (2, 4, 7, 8, 11):
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Expected notebook code cell {index}; source layout changed.")
        tree = ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
        allowed = (ast.FunctionDef, ast.ClassDef)
        if index == 2:
            allowed += (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)
        tree.body = [node for node in tree.body if isinstance(node, allowed)]
        exec(compile(tree, f"{path}:cell-{index}", "exec"), namespace)
    expected = {
        "validate_raw_data", "clean_partitions", "assert_disjoint", "SparseTokenBatch",
        "LocalTextXGB", "train_test_split", "W2V_CONFIG", "XGB_CONFIG",
    }
    missing = expected - namespace.keys()
    if missing:
        raise ValueError(f"The original notebook pipeline is incomplete: {sorted(missing)}")
    if namespace["SEED"] != SPLIT_SEED:
        raise ValueError("The notebook split seed changed; review the study protocol.")
    return namespace


def _versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "gensim", "xgboost-cpu", "xgboost"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _validate_frames(frames: dict[str, pd.DataFrame], fingerprint: str, dataset_hash: str) -> None:
    seen_rows: set[int] = set()
    seen_raw: set[str] = set()
    seen_cleaned: set[str] = set()
    required = {
        "row_id", "text", "cleaned_text", "y_true", "p_xgb", "y_xgb", "error_type",
        "text_sha256", "source_split", "dataset_sha256", "baseline_fingerprint",
    }
    for name in ("train", "validation", "test"):
        frame = frames[name]
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"Incomplete baseline cache table: {name}")
        row_ids = frame["row_id"].to_numpy()
        if not np.issubdtype(row_ids.dtype, np.integer) or len(set(row_ids)) != len(row_ids):
            raise ValueError(f"Baseline row IDs must be distinct integers: {name}")
        if set(frame["y_true"].unique()) != {0, 1}:
            raise ValueError(f"Both classes must remain in baseline split {name}.")
        probabilities = frame["p_xgb"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Invalid baseline probabilities: {name}")
        predicted = (probabilities >= PRIMARY_THRESHOLD).astype(int)
        if not np.array_equal(predicted, frame["y_xgb"].to_numpy()):
            raise ValueError(f"Baseline predictions disagree with the fixed threshold: {name}")
        errors = np.where(
            (frame["y_true"] == 1) & (predicted == 0), "false_negative",
            np.where((frame["y_true"] == 0) & (predicted == 1), "false_positive", "correct"),
        )
        if not np.array_equal(errors, frame["error_type"].to_numpy()):
            raise ValueError(f"Baseline error types disagree with predictions: {name}")
        hashes = frame["text"].map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        if not hashes.equals(frame["text_sha256"]):
            raise ValueError(f"Baseline raw text hashes do not match: {name}")
        for column, expected in (("source_split", name), ("dataset_sha256", dataset_hash),
                                 ("baseline_fingerprint", fingerprint)):
            if not frame[column].eq(expected).all():
                raise ValueError(f"Baseline provenance mismatch in {name}.{column}")
        # The notebook's duplicate keys have already been checked on the fresh
        # path. Verify both exact raw hashes and cleaned text again on replay.
        for seen, values, description in (
            (seen_rows, set(row_ids), "row IDs"),
            (seen_raw, set(frame["text_sha256"]), "raw texts"),
            (seen_cleaned, set(frame["cleaned_text"]), "cleaned texts"),
        ):
            if seen & values:
                raise ValueError(f"Baseline splits overlap in {description}.")
            seen.update(values)


def _read_cache(cache_dir: Path, identity: dict[str, Any], fingerprint: str) -> BaselineResult:
    metadata_path = cache_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["identity"] != identity or metadata["baseline_fingerprint"] != fingerprint:
            raise ValueError("Baseline cache identity differs from the current dataset/configuration.")
        frames = {}
        for name in ("train", "validation", "test"):
            path = cache_dir / f"{name}.csv.gz"
            if _sha256_file(path) != metadata["table_sha256"][name]:
                raise ValueError(f"Baseline cache integrity check failed: {name}")
            frames[name] = pd.read_csv(path, keep_default_na=False)
            if len(frames[name]) != metadata["split_counts"][name]:
                raise ValueError(f"Baseline cache row count changed: {name}")
        _validate_frames(frames, fingerprint, identity["dataset_sha256"])
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("Baseline cache is incomplete or unreadable; inspect it before rebuilding.") from error
    return BaselineResult(frames, metadata, cache_dir, from_cache=True)


def build_baseline(
    dataset_path: str | Path,
    artifacts_dir: str | Path,
    notebook_path: str | Path | None = None,
    seed: int = SPLIT_SEED,
    force: bool = False,
) -> BaselineResult:
    """Fit/cache the full-data central pipeline with the notebook's exact split.

    The training table is provided for provenance only. Jev tuning must use
    validation errors, and final evaluation must use test errors.
    """
    if seed != SPLIT_SEED:
        raise ValueError("This study fixes the original notebook baseline seed at 42.")
    dataset_path = Path(dataset_path).expanduser().resolve(strict=True)
    notebook_path = Path(notebook_path or DEFAULT_NOTEBOOK).resolve(strict=True)
    ns = _load_notebook_definitions(notebook_path)
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_sha256": _sha256_file(dataset_path),
        "notebook_sha256": _sha256_file(notebook_path),
        "split_seed": seed,
        "test_size": ns["TEST_SIZE"],
        "validation_fraction_of_development": ns["VALIDATION_FRACTION_OF_DEVELOPMENT"],
        "w2v_config": ns["W2V_CONFIG"],
        "xgb_config": ns["XGB_CONFIG"],
        "xgb_threads": ns["XGB_THREADS_PER_CLIENT"],
        "primary_threshold": PRIMARY_THRESHOLD,
        "baseline": "Centralized",
        "versions": _versions(),
    }
    fingerprint = _canonical_hash(identity)
    cache_dir = Path(artifacts_dir).expanduser().resolve() / "baseline" / fingerprint
    if (cache_dir / "metadata.json").is_file() and not force:
        return _read_cache(cache_dir, identity, fingerprint)
    if cache_dir.exists() and any(cache_dir.iterdir()) and not force:
        raise ValueError("Incomplete baseline cache exists. Inspect it and use force=True to rebuild.")

    raw = pd.read_csv(dataset_path, usecols=["text", "class"])
    dataset, raw_audit = ns["validate_raw_data"](raw)
    del raw
    development, test = ns["train_test_split"](
        dataset, test_size=ns["TEST_SIZE"], random_state=seed, stratify=dataset["label"]
    )
    train, validation = ns["train_test_split"](
        development, test_size=ns["VALIDATION_FRACTION_OF_DEVELOPMENT"],
        random_state=seed, stratify=development["label"],
    )
    raw_partitions = {"train": train.copy(), "validation": validation.copy(), "test": test.copy()}
    for column in ("row_id", "raw_key"):
        ns["assert_disjoint"](list(raw_partitions.values()), column)
    partitions, cleaning_audit = ns["clean_partitions"](raw_partitions)
    for column in ("row_id", "raw_key", "cleaned_text"):
        ns["assert_disjoint"](list(partitions.values()), column)
    del raw_partitions, development, train, validation, test, dataset

    model = ns["LocalTextXGB"](seed=seed)
    documents = partitions["train"]["cleaned_text"].str.split().tolist()
    fit_info = model.fit(documents, partitions["train"]["label"].to_numpy(dtype=np.int64))
    del documents
    frames = {}
    coverage = {}
    for name, partition in partitions.items():
        batch = ns["SparseTokenBatch"](partition["cleaned_text"].str.split().tolist(), partition["row_id"])
        probabilities, batch_coverage, seconds = model.predict_batch(batch)
        frame = partition.loc[:, ["row_id", "text", "cleaned_text", "label"]].rename(columns={"label": "y_true"}).copy()
        frame["p_xgb"] = probabilities.astype(float)
        frame["y_xgb"] = (probabilities >= PRIMARY_THRESHOLD).astype(int)
        frame["error_type"] = np.where(
            (frame["y_true"] == 1) & (frame["y_xgb"] == 0), "false_negative",
            np.where((frame["y_true"] == 0) & (frame["y_xgb"] == 1), "false_positive", "correct"),
        )
        frame["text_sha256"] = frame["text"].map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
        frame["source_split"] = name
        frame["dataset_sha256"] = identity["dataset_sha256"]
        frame["baseline_fingerprint"] = fingerprint
        frames[name] = frame
        coverage[name] = {**batch_coverage, "prediction_seconds": seconds, "row_ids_digest": batch.id_digest}
        del batch
    _validate_frames(frames, fingerprint, identity["dataset_sha256"])

    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = {
        "identity": identity,
        "baseline_fingerprint": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_counts": {name: len(frame) for name, frame in frames.items()},
        "raw_quality_audit": {name: int(value) for name, value in raw_audit.items()},
        "cleaning_audit": cleaning_audit.reset_index().to_dict(orient="records"),
        "fit": fit_info,
        "coverage": coverage,
        "error_counts": {name: frame["error_type"].value_counts().to_dict() for name, frame in frames.items()},
        "training_table_scope": "In-sample provenance only; never eligible for Jev selection.",
        "table_sha256": {},
    }
    # Metadata is committed last. A partial write is never accepted as a cache.
    for name, frame in frames.items():
        destination = cache_dir / f"{name}.csv.gz"
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as temporary:
            path = Path(temporary.name)
        try:
            frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            path.chmod(0o600)
            path.replace(destination)
        finally:
            path.unlink(missing_ok=True)
        metadata["table_sha256"][name] = _sha256_file(destination)
    metadata_path = cache_dir / "metadata.json"
    temporary_metadata = cache_dir / "metadata.json.tmp"
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary_metadata.chmod(0o600)
    temporary_metadata.replace(metadata_path)
    return BaselineResult(frames, metadata, cache_dir)
