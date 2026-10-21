"""Validation-selected probability routing and a separately budgeted Jev run."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import sqlite3

import numpy as np
import pandas as pd

from .client import ProbabilityHybridJevClient, JevClientError, MissingCredential
from .hybrid_reporting import build_hybrid_report, write_hybrid_report
from .notebook_pipeline import _read_cache, build_baseline
from .probability_routing import select_probability_interval, select_routing_cohort
from .study import write_json


TARGET_SIZE = 5000
COHORT_COLUMNS = ["sample_id", "row_id", "text_sha256", "p_xgb", "source_split",
                  "dataset_sha256", "baseline_fingerprint"]


def _cohort_digest(frame):
    records = frame[COHORT_COLUMNS].sort_values("sample_id").to_dict("records")
    return hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_probability_hybrid(dataset_path, artifacts_dir="artifacts/jev_probability_hybrid",
                               results_dir="research/results/jev_probability_hybrid",
                               baseline_artifacts="artifacts/jev_study"):
    """Lock the validation-only interval and label-blind 5,000-text test sample."""
    artifacts, results = Path(artifacts_dir), Path(results_dir)
    baseline = build_baseline(dataset_path, baseline_artifacts)
    validation, test = baseline.frames["validation"], baseline.frames["test"]
    if set(validation.row_id) & set(test.row_id) or set(validation.text_sha256) & set(test.text_sha256):
        raise ValueError("Validation and test identities must be disjoint.")
    interval = select_probability_interval(validation, target_size=TARGET_SIZE, test_size=len(test))
    selection = select_routing_cohort(test, interval.interval, target_size=TARGET_SIZE, seed=42)
    cohort = selection.cohort
    digest = _cohort_digest(cohort)
    artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
    results.mkdir(parents=True, exist_ok=True)
    private_metadata = artifacts / "study.json"
    prepared_at = datetime.now(timezone.utc).isoformat()
    if private_metadata.exists():
        previous = json.loads(private_metadata.read_text())
        if previous["cohort_sha256"] != digest or previous["interval"] != interval.interval:
            raise ValueError("An existing probability study differs. Preserve its original cohort and ledger.")
        prepared_at = previous.get("cohort_locked_at_utc", prepared_at)
    destination = artifacts / "cohort.csv.gz"
    cohort.to_csv(destination, index=False, compression={"method": "gzip", "mtime": 0})
    destination.chmod(0o600)
    public = {
        "study_schema_version": 1, "study_kind": "probability_routed_hybrid",
        "dataset_sha256": baseline.metadata["identity"]["dataset_sha256"],
        "baseline_fingerprint": baseline.metadata["baseline_fingerprint"],
        "cohort_sha256": digest, "target_size": TARGET_SIZE,
        "cohort_locked_at_utc": prepared_at,
        "interval": interval.interval, "interval_selection": interval.metadata,
        "cohort_selection": selection.metadata,
        "test_size": len(test), "validation_size": len(validation),
        "base_model": "jev-latest", "score_threshold": 0.5,
        "new_paid_attempt_limit": TARGET_SIZE,
        "budget_scope": "Separately authorized 5,000-attempt probability study; prior 100-attempt study is unchanged.",
        "hybrid_rule": "Use Jev score for successfully queried selected texts; otherwise retain XGBoost score.",
        "training": "No weight fine-tuning; use the existing Jev base-model alias.",
    }
    private = {**public, "baseline_cache_dir": str(baseline.cache_dir.resolve()),
               "baseline_identity": baseline.metadata["identity"]}
    write_json(private_metadata, private)
    write_json(results / "study.json", public)
    interval.diagnostics.to_json(results / "validation_bins.json", orient="records", indent=2, double_precision=15)
    interval.candidates.to_json(results / "validation_interval_candidates.json", orient="records", indent=2, double_precision=15)
    selection.manifest.to_json(results / "routing_manifest.json", orient="records", indent=2, double_precision=15)
    return public


def load_probability_hybrid(artifacts_dir):
    artifacts = Path(artifacts_dir)
    study = json.loads((artifacts / "study.json").read_text())
    cohort = pd.read_csv(artifacts / "cohort.csv.gz", float_precision="round_trip", keep_default_na=False)
    if len(cohort) != TARGET_SIZE or _cohort_digest(cohort) != study["cohort_sha256"]:
        raise ValueError("The 5,000-text routing cohort failed its integrity check.")
    if cohort.sample_id.duplicated().any() or cohort.text_sha256.duplicated().any():
        raise ValueError("Routing identities must be distinct.")
    hashes = cohort.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    if not hashes.equals(cohort.text_sha256):
        raise ValueError("Routing text checksums changed; no paid call is permitted.")
    bounds = (study["interval"]["lower"], study["interval"]["upper"])
    # Construction validates every probability and split before any network I/O.
    return cohort, study, bounds


def _load_full_test(study):
    result = _read_cache(Path(study["baseline_cache_dir"]), study["baseline_identity"],
                         study["baseline_fingerprint"])
    return result.frames["test"]


def _verify_selected_against_test(cohort, full_test):
    if full_test.row_id.duplicated().any() or not set(cohort.row_id).issubset(set(full_test.row_id)):
        raise ValueError("Routing cohort identities do not match the full test cache.")
    aligned = full_test.set_index("row_id").loc[cohort.row_id]
    for column in ("text_sha256", "p_xgb", "y_true", "dataset_sha256", "baseline_fingerprint"):
        if not np.array_equal(cohort[column].to_numpy(), aligned[column].to_numpy()):
            raise ValueError(f"Routing cohort differs from the full test cache: {column}.")


def report_probability_hybrid(artifacts_dir="artifacts/jev_probability_hybrid",
                              results_dir="research/results/jev_probability_hybrid"):
    """Reconstruct reports from verified local evidence, without model calls."""
    artifacts, results = Path(artifacts_dir), Path(results_dir)
    cohort, study, bounds = load_probability_hybrid(artifacts)
    ledger = artifacts / "budget.sqlite3"
    if not ledger.exists():
        raise ValueError("The original probability-study ledger is required to reconstruct outcomes.")
    client = ProbabilityHybridJevClient(ledger, cohort=cohort, interval=bounds)
    records = client.ledger_records()
    budget = client.ledger_summary()
    with sqlite3.connect(ledger) as connection:
        started, finished = connection.execute(
            "SELECT MIN(started_at),MAX(finished_at) FROM attempts"
        ).fetchone()
    public_study = {key: value for key, value in study.items()
                    if key not in {"baseline_cache_dir", "baseline_identity"}}
    report = build_hybrid_report(_load_full_test(study), cohort, records,
                                 metadata={**public_study, "inference_budget": budget,
                                           "execution_window_utc": {"first_attempt": started, "last_finished": finished}})
    results.mkdir(parents=True, exist_ok=True)
    write_hybrid_report(report, results)
    write_json(results / "inference_budget.json", budget)
    # Whitelist predictions and provenance; never export provider payloads/texts.
    public_predictions = cohort[COHORT_COLUMNS + ["y_true"]].copy()
    lookup = {record["sample_id"]: record for record in records}
    public_predictions["jev_score"] = public_predictions.sample_id.map(
        lambda identity: lookup.get(identity, {}).get("score"))
    public_predictions["jev_status"] = public_predictions.sample_id.map(
        lambda identity: lookup.get(identity, {}).get("status", "not_attempted"))
    public_predictions.to_json(results / "predictions.json", orient="records", indent=2, double_precision=15)
    return report


def run_probability_hybrid(artifacts_dir="artifacts/jev_probability_hybrid",
                           results_dir="research/results/jev_probability_hybrid", *,
                           authorized_attempts, workers=4):
    """Execute at most 5,000 new paid attempts, with bounded concurrency.

    Existing successes, failures, and uncertain reservations are never resent.
    Stop scheduling after a provider error; a later deliberate resume may query
    still-unattempted rows while keeping failed cases on the XGBoost fallback.
    """
    if authorized_attempts != TARGET_SIZE or isinstance(authorized_attempts, bool):
        raise ValueError("Explicit authorization for the separate 5,000-attempt budget is required.")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Use between one and eight bounded request workers.")
    artifacts = Path(artifacts_dir)
    cohort, study, bounds = load_probability_hybrid(artifacts)
    ledger = artifacts / "budget.sqlite3"
    coordinator = ProbabilityHybridJevClient(ledger, cohort=cohort, interval=bounds)
    # Validate evidence needed for final metrics before any expensive request.
    _verify_selected_against_test(cohort, _load_full_test(study))
    recorded = {record["sample_id"] for record in coordinator.ledger_records()}
    work = iter(row for row in cohort.itertuples(index=False) if row.sample_id not in recorded)
    initial = coordinator.ledger_summary()
    if initial["attempts_spent"] < TARGET_SIZE and not os.environ.get("METIS_API_KEY", "").strip():
        raise MissingCredential("Set METIS_API_KEY securely; no paid attempt was made.")
    local = threading.local()

    def initialize():
        local.client = ProbabilityHybridJevClient(ledger, cohort=cohort, interval=bounds)

    def predict(row):
        try:
            local.client.predict(row.sample_id, row.text)
            return True
        except JevClientError:
            return False

    stopped = False
    completed = len(recorded)
    last_log = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
            pending = set()
            for _ in range(workers):
                row = next(work, None)
                if row is not None:
                    pending.add(pool.submit(predict, row))
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    if not future.result():
                        stopped = True
                if (done and completed % 100 == 0) or time.monotonic() - last_log >= 30:
                    summary = coordinator.ledger_summary()
                    write_json(artifacts / "progress.json", summary)
                    print(f"Probability hybrid: {summary['attempts_spent']}/5000 attempts; "
                          f"{summary['succeeded']} successful; {summary['failed']} failed.", flush=True)
                    last_log = time.monotonic()
                if not stopped:
                    for _ in done:
                        row = next(work, None)
                        if row is not None:
                            pending.add(pool.submit(predict, row))
    finally:
        report_probability_hybrid(artifacts, results_dir)
    summary = coordinator.ledger_summary()
    if stopped:
        raise RuntimeError("Paid dispatch stopped after a provider error. Preserve the ledger and diagnose before resuming.")
    return summary
