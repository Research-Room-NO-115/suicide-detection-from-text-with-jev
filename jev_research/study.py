"""Prepare private training data and publish only redacted study artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .notebook_pipeline import build_baseline
from .selection import select_error_cohort


TRAINING_STATUS = {
    "status": "blocked_not_run",
    "method_requested": "model_weight_fine_tuning",
    "provider_support": "unverified",
    "training_jobs_created": 0,
    "training_spend": 0,
    "missing_requirements": [
        "Official supported Jev training API and custom-model invocation specification",
        "Access to that training service",
        "An explicitly approved training-fee cap",
    ],
    "evidence": {
        "existing_interface": "https://api.metisai.ir/typesafe/v1/systemone",
        "authenticated_conventional_openapi_routes": "HTTP 404",
        "official_documentation": "https://docs.metisai.ir/",
        "official_documentation_access": "Proxy blocked during this study",
        "provider_discovery_paid_inference_attempts": 0,
    },
    "data_format": "Provider-neutral records; not a verified provider upload schema",
}

MANIFEST_COLUMNS = [
    "sample_id", "dataset_sha256", "baseline_fingerprint", "source_split", "row_id",
    "text_sha256", "y_true", "p_xgb", "y_xgb", "error_type", "jev_partition", "jev_role",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def cohort_digest(cohort: pd.DataFrame) -> str:
    records = cohort[MANIFEST_COLUMNS].sort_values("sample_id").to_dict(orient="records")
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _full_test_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    tn, fp, fn, tp = confusion_matrix(frame.y_true, frame.y_xgb, labels=[0, 1]).ravel()
    return {
        "n": len(frame), "threshold": 0.5,
        "accuracy": float(accuracy_score(frame.y_true, frame.y_xgb)),
        "positive_f1": float(f1_score(frame.y_true, frame.y_xgb)),
        "roc_auc": float(roc_auc_score(frame.y_true, frame.p_xgb)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def prepare_study(dataset_path: str | Path, artifacts_dir: str | Path,
                  results_dir: str | Path) -> dict[str, Any]:
    """Use the full original XGBoost split, then preselect at most 100 errors."""
    artifacts = Path(artifacts_dir)
    results = Path(results_dir)
    baseline = build_baseline(dataset_path, artifacts)
    selection = select_error_cohort(baseline.frames["validation"], baseline.frames["test"])
    cohort = selection.cohort
    digest = cohort_digest(cohort)
    artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata_path = artifacts / "study.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing["cohort_sha256"] != digest:
            raise ValueError("The saved study differs. Preserve its ledger; do not reset the budget.")
    private_cohort = artifacts / "cohort.csv.gz"
    cohort.to_csv(private_cohort, index=False, compression={"method": "gzip", "mtime": 0})
    private_cohort.chmod(0o600)

    training_counts = {}
    for role in ("train", "validation"):
        rows = cohort.loc[cohort.jev_role.eq(role)]
        destination = artifacts / f"{role}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for row in rows.itertuples(index=False):
                # Gold labels are training targets, never inference request metadata.
                stream.write(json.dumps({"sample_id": row.sample_id, "text": row.text,
                                         "label": int(row.y_true)}, ensure_ascii=False) + "\n")
        destination.chmod(0o600)
        training_counts[role] = len(rows)

    metadata = {
        "study_schema_version": 1, "cohort_sha256": digest,
        "baseline_fingerprint": baseline.metadata["baseline_fingerprint"],
        "dataset_sha256": baseline.metadata["identity"]["dataset_sha256"],
        "sampling": selection.metadata, "training_counts": training_counts,
        "evaluation_count": int(cohort.jev_role.eq("evaluation").sum()),
        "training_status": "blocked_not_run", "max_paid_inference_attempts": 100,
        "base_model": "jev-latest", "classification_threshold": 0.5,
        "full_xgboost_test_metrics": _full_test_metrics(baseline.frames["test"]),
    }
    write_json(metadata_path, metadata)
    # Publish hashes/provenance/aggregate scores only. Raw texts stay ignored.
    write_json(results / "study.json", metadata)
    write_json(results / "xgboost_baseline.json", baseline.metadata)
    write_json(results / "training_status.json", TRAINING_STATUS)
    cohort[MANIFEST_COLUMNS].to_json(results / "selection_manifest.json", orient="records", indent=2,
                                      double_precision=15)
    return metadata


def load_study(artifacts_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifacts = Path(artifacts_dir)
    metadata = json.loads((artifacts / "study.json").read_text())
    cohort = pd.read_csv(artifacts / "cohort.csv.gz", float_precision="round_trip")
    if cohort_digest(cohort) != metadata["cohort_sha256"]:
        raise ValueError("Study cohort integrity check failed; no provider requests were made.")
    if cohort.sample_id.duplicated().any() or cohort.text_sha256.duplicated().any():
        raise ValueError("Study cohort contains duplicate samples.")
    actual_hashes = cohort.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    if not actual_hashes.equals(cohort.text_sha256):
        raise ValueError("Study text checksum failed; no provider requests were made.")
    if not set(cohort.jev_role).issubset({"train", "validation", "evaluation"}):
        raise ValueError("Unknown study role.")
    for row in cohort.itertuples(index=False):
        if row.y_xgb != int(row.p_xgb >= 0.5) or row.y_xgb == row.y_true:
            raise ValueError("Every selected row must be an XGBoost error at threshold 0.50.")
        expected_source = "test" if row.jev_role == "evaluation" else "validation"
        expected_partition = "evaluation" if row.jev_role == "evaluation" else "tuning"
        if row.source_split != expected_source or row.jev_partition != expected_partition:
            raise ValueError("The evaluation and tuning roles must remain separate.")
    return cohort, metadata


def run_base_evaluation(artifacts_dir: str | Path, results_dir: str | Path) -> dict[str, Any]:
    """Run only the 40 prespecified held-out errors; cache/replay spends nothing."""
    from .client import BudgetedJevClient, JevClientError

    artifacts, results = Path(artifacts_dir), Path(results_dir)
    cohort, metadata = load_study(artifacts)
    client = BudgetedJevClient(artifacts / "budget.sqlite3", cohort=cohort)
    results.mkdir(parents=True, exist_ok=True)
    evaluation = cohort.loc[cohort.jev_role.eq("evaluation")].copy()
    failed = False
    for completed, row in enumerate(evaluation.itertuples(index=False), start=1):
        try:
            client.predict(row.sample_id, row.text)
        except JevClientError:
            failed = True
        if failed:
            break  # A diagnosis/state change is required before any further paid work.
        if completed % 5 == 0:
            print(f"Base Jev evaluation: {completed}/{len(evaluation)} responses available.", flush=True)
    summary = client.ledger_summary()
    _report_from_ledger(cohort, metadata, client, results)
    if failed:
        raise RuntimeError("Provider evaluation stopped after a failure; consult redacted status and ledger.")
    return summary


def _report_from_ledger(cohort: pd.DataFrame, metadata: dict[str, Any], client: Any,
                        results: Path) -> dict[str, Any]:
    results.mkdir(parents=True, exist_ok=True)
    outcomes = {record["sample_id"]: record for record in client.ledger_records()}
    evaluation = cohort.loc[cohort.jev_role.eq("evaluation")].copy()
    evaluation["base_score"] = evaluation.sample_id.map(
        lambda sample: outcomes.get(sample, {}).get("score", float("nan")))
    evaluation["base_status"] = evaluation.sample_id.map(
        lambda sample: outcomes.get(sample, {}).get("status", "not_attempted"))
    evaluation["tuned_score"] = float("nan")
    public = evaluation[MANIFEST_COLUMNS + ["base_score", "base_status"]]
    public.to_json(results / "base_predictions.json", orient="records", indent=2, double_precision=15)
    summary = client.ledger_summary()
    write_json(results / "inference_budget.json", summary)
    metadata = {**metadata, "inference_budget": summary}
    return generate_report(evaluation, metadata, results)


def generate_report(evaluation: pd.DataFrame, metadata: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    from .reporting import build_comparison_report, write_report
    report = build_comparison_report(evaluation, metadata=metadata)
    write_report(report, results_dir)
    return report


def report_saved_results(artifacts_dir: str | Path, results_dir: str | Path) -> dict[str, Any]:
    """Rebuild reports offline. This function never contacts Jev."""
    from .client import BudgetedJevClient
    ledger = Path(artifacts_dir) / "budget.sqlite3"
    if not ledger.is_file():
        raise ValueError("The private ledger is required to regenerate results. Published reports can be read offline.")
    cohort, metadata = load_study(artifacts_dir)
    client = BudgetedJevClient(ledger, cohort=cohort)
    return _report_from_ledger(cohort, metadata, client, Path(results_dir))
