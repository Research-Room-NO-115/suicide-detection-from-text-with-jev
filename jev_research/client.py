"""Cost-bounded Jev inference with an append-only SQLite attempt ledger.

The caller must keep one ledger for the entire study. Failed and interrupted
requests consume attempts. Nothing in this module starts training or retries a
request. A ledger is research evidence: do not delete it to resume a study.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import sqlite3
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ENDPOINT = "https://api.metisai.ir/typesafe/v1/systemone"
BASE_MODEL = "jev-latest"
HARD_MAX_ATTEMPTS = 100
HARD_MAX_SAMPLES = 100
HYBRID_MAX_ATTEMPTS = 5000
HYBRID_MAX_SAMPLES = 5000
QUESTION = {
    "type": "score",
    "instructions": (
        "On a scale of 0 to 1, how likely is it that this text indicates a risk "
        "of suicidal intent?"
    ),
    "criteria": ["0 - No risk indicated", "1 - High risk indicated"],
}


class JevClientError(RuntimeError):
    """A sanitized provider/client failure that contains no request payload."""


class BudgetExhausted(JevClientError):
    """The durable study-wide inference-attempt cap has been reached."""


class CohortMismatch(JevClientError):
    """The request or cohort does not match the authorized study manifest."""


class AttemptAlreadyRecorded(JevClientError):
    """A failed or uncertain request must not be replayed automatically."""


class MissingCredential(JevClientError):
    """The environment lacks the provider credential; no attempt was spent."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _binary(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CohortMismatch(f"{name} must be a binary integer label.")
    if value not in (0, 1):
        raise CohortMismatch(f"{name} must be a binary integer label.")
    return int(value)


def _cohort_records(cohort: Any) -> dict[str, dict[str, Any]]:
    if hasattr(cohort, "to_dict") and not isinstance(cohort, Mapping):
        cohort = cohort.to_dict("records")
    if not isinstance(cohort, Iterable) or isinstance(cohort, (str, bytes, Mapping)):
        raise CohortMismatch("Provide the selected cohort as records or a DataFrame.")
    records: dict[str, dict[str, Any]] = {}
    for row in cohort:
        if not isinstance(row, Mapping):
            raise CohortMismatch("Every cohort row must be a mapping.")
        required = {"sample_id", "text_sha256", "y_true", "y_xgb", "error_type"}
        if required - row.keys():
            raise CohortMismatch("Cohort rows lack required authorization fields.")
        sample_id = str(row["sample_id"])
        digest = row["text_sha256"]
        if not sample_id or len(sample_id) > 256:
            raise CohortMismatch("Invalid sample identifier.")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CohortMismatch("Invalid text checksum.")
        try:
            int(digest, 16)
        except ValueError:
            raise CohortMismatch("Invalid text checksum.") from None
        truth, predicted = _binary(row["y_true"], "y_true"), _binary(row["y_xgb"], "y_xgb")
        if truth == predicted:
            raise CohortMismatch("Jev may only receive XGBoost-misclassified examples.")
        expected_error = "false_negative" if truth else "false_positive"
        if row["error_type"] != expected_error:
            raise CohortMismatch("The error type disagrees with the binary labels.")
        if sample_id in records:
            raise CohortMismatch("Duplicate sample identifiers are not allowed.")
        records[sample_id] = {
            "sample_id": sample_id,
            "text_sha256": digest.lower(),
            "y_true": truth,
            "y_xgb": predicted,
            "error_type": expected_error,
            "jev_partition": str(row.get("jev_partition", "")),
            "source_split": str(row.get("source_split", "")),
        }
    if not records or len(records) > HARD_MAX_SAMPLES:
        raise CohortMismatch("Authorize between one and 100 unique XGBoost errors.")
    if len({row["text_sha256"] for row in records.values()}) != len(records):
        raise CohortMismatch("Duplicate text checksums are not allowed in the cohort.")
    return records


class BudgetedJevClient:
    """Infer authorized error samples with at most 100 durable paid attempts.

    ``cohort`` contains the metadata identifying the exact selected errors. Raw
    texts are supplied to ``predict`` and are not stored in the ledger. The
    API key is read only from ``METIS_API_KEY`` at call time. ``session`` is an
    injection point for offline tests; do not use an automatically retrying
    custom transport in production.
    """

    ATTEMPT_CAP = HARD_MAX_ATTEMPTS
    SAMPLE_CAP = HARD_MAX_SAMPLES

    def _authorize_cohort(self, cohort: Any) -> dict[str, dict[str, Any]]:
        return _cohort_records(cohort)

    def _cohort_identity(self) -> Any:
        # Preserve the original study's fingerprint and its existing ledger.
        return sorted(self.cohort.values(), key=lambda row: row["sample_id"])

    def __init__(
        self,
        ledger_path: str | Path,
        cohort: Any,
        *,
        allowed_models: Iterable[str] = (BASE_MODEL,),
        max_attempts: int = HARD_MAX_ATTEMPTS,
        timeout: tuple[float, float] = (10.0, 60.0),
        session: Any = None,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError(f"max_attempts must be an integer between one and {self.ATTEMPT_CAP}.")
        if not 1 <= max_attempts <= self.ATTEMPT_CAP:
            raise ValueError(f"max_attempts must be an integer between one and {self.ATTEMPT_CAP}.")
        if len(timeout) != 2 or any(not math.isfinite(v) or v <= 0 for v in timeout):
            raise ValueError("Use positive finite connection and read timeouts.")
        self.cohort = self._authorize_cohort(cohort)
        self.cohort_fingerprint = text_sha256(_canonical(self._cohort_identity()))
        self._question_json = _canonical(QUESTION)
        self.instruction_fingerprint = text_sha256(self._question_json)
        self.allowed_models = frozenset(allowed_models)
        if not self.allowed_models or any(not isinstance(m, str) or not m.strip() for m in self.allowed_models):
            raise ValueError("Explicitly authorize at least one nonempty model identifier.")
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.session = session if session is not None else requests.Session()
        if session is None:
            retries = Retry(total=0, connect=0, read=0, redirect=0, status=0)
            self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self._initialize()

    @property
    def question(self) -> dict[str, Any]:
        """Return a copy; mutation cannot desynchronize the payload/cache hash."""
        return json.loads(self._question_json)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.ledger_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute("""CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_key TEXT UNIQUE NOT NULL,
                    sample_id TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    model TEXT NOT NULL,
                    instruction_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','succeeded','failed')),
                    score REAL,
                    error_code TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                )""")
                saved = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
                if saved:
                    if saved.get("cohort_fingerprint") != self.cohort_fingerprint:
                        raise CohortMismatch("This ledger belongs to a different selected cohort.")
                    old_cap = int(saved["max_attempts"])
                    if self.max_attempts > old_cap:
                        raise CohortMismatch("An existing ledger's inference cap cannot be increased.")
                    used = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                    if self.max_attempts < used:
                        raise CohortMismatch("The requested cap is below attempts already spent.")
                    if self.max_attempts < old_cap:
                        connection.execute("UPDATE metadata SET value=? WHERE key='max_attempts'", (str(self.max_attempts),))
                else:
                    connection.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", [
                        ("schema_version", "1"),
                        ("cohort_fingerprint", self.cohort_fingerprint),
                        ("max_attempts", str(self.max_attempts)),
                        ("authorized_samples", str(len(self.cohort))),
                    ])
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        # Private local research metadata; no API key or text is written here.
        self.ledger_path.chmod(0o600)

    def _request_key(self, sample_id: str, digest: str, model: str) -> str:
        return text_sha256(_canonical({"sample_id": sample_id, "text_sha256": digest,
                                       "model": model, "question_sha256": self.instruction_fingerprint,
                                       "endpoint": ENDPOINT}))

    def _reserve(self, sample_id: str, digest: str, model: str) -> tuple[int | None, float | None]:
        request_key = self._request_key(sample_id, digest, model)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute("SELECT id,status,score FROM attempts WHERE request_key=?", (request_key,)).fetchone()
                if existing:
                    if existing["status"] == "succeeded":
                        connection.commit()
                        return None, self._validated_saved_score(existing["score"])
                    raise AttemptAlreadyRecorded(
                        "This request has a failed or uncertain recorded attempt; it will not be replayed automatically."
                    )
                limit = int(connection.execute("SELECT value FROM metadata WHERE key='max_attempts'").fetchone()[0])
                used = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                if used >= limit:
                    raise BudgetExhausted("The durable inference-attempt cap is exhausted.")
                count = connection.execute("SELECT COUNT(DISTINCT sample_id) FROM attempts").fetchone()[0]
                known = connection.execute("SELECT 1 FROM attempts WHERE sample_id=? LIMIT 1", (sample_id,)).fetchone()
                if not known and count >= self.SAMPLE_CAP:
                    raise BudgetExhausted(f"The {self.SAMPLE_CAP}-unique-sample cap is exhausted.")
                cursor = connection.execute("""INSERT INTO attempts
                    (request_key,sample_id,text_sha256,model,instruction_sha256,status,started_at)
                    VALUES (?,?,?,?,?,'pending',?)""", (request_key, sample_id, digest, model,
                    self.instruction_fingerprint, datetime.now(timezone.utc).isoformat()))
                attempt_id = cursor.lastrowid
                connection.commit()
                return int(attempt_id), None
            except BaseException:
                connection.rollback()
                raise

    def _finish(self, attempt_id: int, *, score: float | None = None, error_code: str | None = None) -> None:
        with self._connection() as connection:
            connection.execute("""UPDATE attempts SET status=?,score=?,error_code=?,finished_at=?
                WHERE id=? AND status='pending'""", (
                "succeeded" if error_code is None else "failed", score, error_code,
                datetime.now(timezone.utc).isoformat(), attempt_id,
            ))

    def predict(self, sample_id: str, text: str, *, model: str = BASE_MODEL) -> float:
        sample_id = str(sample_id)
        row = self.cohort.get(sample_id)
        if row is None or not isinstance(text, str):
            raise CohortMismatch("The sample is not in the authorized study cohort.")
        digest = text_sha256(text)
        if digest != row["text_sha256"]:
            raise CohortMismatch("The text does not match the authorized sample checksum.")
        if model not in self.allowed_models:
            raise CohortMismatch("The model is not authorized for this client.")
        # A cached success may be read without a credential or another attempt.
        request_key = self._request_key(sample_id, digest, model)
        with self._connection() as connection:
            cached = connection.execute("SELECT status,score FROM attempts WHERE request_key=?", (request_key,)).fetchone()
        if cached:
            if cached["status"] == "succeeded":
                return self._validated_saved_score(cached["score"])
            raise AttemptAlreadyRecorded("This failed or uncertain attempt cannot be replayed automatically.")
        api_key = os.environ.get("METIS_API_KEY", "").strip()
        if not api_key:
            raise MissingCredential("Set METIS_API_KEY securely in environment settings; no inference attempt was spent.")
        attempt_id, cached_score = self._reserve(sample_id, digest, model)
        if attempt_id is None:
            return float(cached_score)
        payload = {"state": text, "model": model, "questions": {"department": json.loads(self._question_json)}}
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        try:
            response = self.session.post(ENDPOINT, headers=headers, json=payload,
                                         timeout=self.timeout, allow_redirects=False)
        except Exception:
            self._finish(attempt_id, error_code="transport_error")
            raise JevClientError("Provider transport failure; the attempt remains counted and was not retried.") from None
        if not 200 <= response.status_code < 300:
            self._finish(attempt_id, error_code=f"http_{response.status_code}")
            raise JevClientError(f"Provider returned HTTP {response.status_code}; the attempt remains counted.")
        try:
            score = response.json()["answers"]["department"]["score"]
            if isinstance(score, bool) or not isinstance(score, numbers.Real):
                raise ValueError("Invalid score type")
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid score range")
        except (ValueError, TypeError, KeyError, OverflowError):
            self._finish(attempt_id, error_code="invalid_response")
            raise JevClientError("Provider returned an invalid risk score; the attempt remains counted.") from None
        self._finish(attempt_id, score=score)
        return score

    @staticmethod
    def _validated_saved_score(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise JevClientError("The saved ledger contains an invalid risk score; preserve it for diagnosis.")
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise JevClientError("The saved ledger contains an invalid risk score; preserve it for diagnosis.")
        return score

    def ledger_records(self, *, model: str = BASE_MODEL) -> list[dict[str, Any]]:
        """Return verified current-question outcomes without any provider I/O.

        These records, rather than an editable predictions JSON file, are the
        source for offline report regeneration. Texts and credentials are never
        returned. Pending records remain explicitly uncertain.
        """
        if model not in self.allowed_models:
            raise CohortMismatch("The model is not authorized for this client.")
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT request_key,sample_id,text_sha256,model,status,score,error_code
                FROM attempts WHERE model=? AND instruction_sha256=? ORDER BY id""",
                (model, self.instruction_fingerprint),
            ).fetchall()
        records, seen = [], set()
        for row in rows:
            sample_id = row["sample_id"]
            expected = self.cohort.get(sample_id)
            if expected is None or row["text_sha256"] != expected["text_sha256"]:
                raise CohortMismatch("A saved attempt does not match the authorized cohort.")
            if row["request_key"] != self._request_key(sample_id, row["text_sha256"], model):
                raise CohortMismatch("A saved request checksum does not match its authorization.")
            if sample_id in seen:
                raise CohortMismatch("The ledger contains duplicate current-model outcomes.")
            seen.add(sample_id)
            status = row["status"]
            if status not in {"pending", "succeeded", "failed"}:
                raise JevClientError("The saved ledger contains an invalid request status.")
            if status == "succeeded":
                score = self._validated_saved_score(row["score"])
                if row["error_code"] is not None:
                    raise JevClientError("The saved ledger has contradictory success/error fields.")
            else:
                score = None
                if row["score"] is not None:
                    raise JevClientError("The saved ledger has a score for an unresolved request.")
            error_code = row["error_code"]
            if error_code is not None:
                known = error_code in {"transport_error", "invalid_response"}
                http_code = (isinstance(error_code, str) and len(error_code) == 8
                             and error_code.startswith("http_") and error_code[5:].isdigit())
                if not (known or http_code):
                    raise JevClientError("The saved ledger contains an unrecognized error code.")
            records.append({"sample_id": sample_id, "status": status,
                            "score": score, "error_code": error_code})
        return records

    def ledger_summary(self) -> dict[str, Any]:
        with self._connection() as connection:
            counts = {row["status"]: row["n"] for row in connection.execute("SELECT status,COUNT(*) AS n FROM attempts GROUP BY status")}
            total = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            unique = connection.execute("SELECT COUNT(DISTINCT sample_id) FROM attempts").fetchone()[0]
            cap = int(connection.execute("SELECT value FROM metadata WHERE key='max_attempts'").fetchone()[0])
        return {"max_attempts": cap, "attempts_spent": total, "remaining_attempts": cap-total,
                "unique_samples_attempted": unique, "authorized_samples": len(self.cohort),
                "succeeded": counts.get("succeeded", 0), "failed": counts.get("failed", 0),
                "pending_uncertain": counts.get("pending", 0),
                "cohort_fingerprint": self.cohort_fingerprint}


class ProbabilityHybridJevClient(BudgetedJevClient):
    """Infer a preselected test-score interval with a separate 5,000-attempt cap.

    Unlike the original error-only study, this routing study accepts correctly
    and incorrectly classified texts. Authorization depends on the locked
    probability interval and sample/text checksums, never a gold label. Use a
    separate ledger for this separately authorized experiment. Intervals are
    ``[lower, upper)``; an upper bound of 1.0 includes the score 1.0.

    For parallel I/O, create one client per worker with an independent default
    requests Session, all sharing the same ledger and manifest. A client should
    be created once per worker, not once per sample. Concurrent reservations
    remain serialized by SQLite before any provider request.
    """

    ATTEMPT_CAP = HYBRID_MAX_ATTEMPTS
    SAMPLE_CAP = HYBRID_MAX_SAMPLES

    def __init__(
        self,
        ledger_path: str | Path,
        cohort: Any,
        *,
        interval: tuple[float, float],
        allowed_models: Iterable[str] = (BASE_MODEL,),
        max_attempts: int = HYBRID_MAX_ATTEMPTS,
        timeout: tuple[float, float] = (10.0, 60.0),
        session: Any = None,
    ) -> None:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise CohortMismatch("Provide the locked probability interval as (lower, upper).")
        if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
               for value in interval):
            raise CohortMismatch("Probability interval bounds must be finite numbers.")
        lower, upper = map(float, interval)
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper <= 1):
            raise CohortMismatch("Probability interval must satisfy 0 <= lower < upper <= 1.")
        self.interval = (lower, upper)
        super().__init__(ledger_path, cohort, allowed_models=allowed_models,
                         max_attempts=max_attempts, timeout=timeout, session=session)

    def _authorize_cohort(self, cohort: Any) -> dict[str, dict[str, Any]]:
        if hasattr(cohort, "to_dict") and not isinstance(cohort, Mapping):
            cohort = cohort.to_dict("records")
        if not isinstance(cohort, Iterable) or isinstance(cohort, (str, bytes, Mapping)):
            raise CohortMismatch("Provide the locked probability-routed cohort as records or a DataFrame.")
        records = {}
        lower, upper = self.interval
        for row in cohort:
            if not isinstance(row, Mapping):
                raise CohortMismatch("Every probability-routed cohort row must be a mapping.")
            if {"sample_id", "text_sha256", "p_xgb", "source_split"} - row.keys():
                raise CohortMismatch("Probability-routed rows lack required authorization fields.")
            sample_id, digest, probability = str(row["sample_id"]), row["text_sha256"], row["p_xgb"]
            if not sample_id or len(sample_id) > 256:
                raise CohortMismatch("Invalid sample identifier.")
            if not isinstance(digest, str) or len(digest) != 64:
                raise CohortMismatch("Invalid text checksum.")
            try:
                int(digest, 16)
            except ValueError:
                raise CohortMismatch("Invalid text checksum.") from None
            if row["source_split"] != "test":
                raise CohortMismatch("Probability-routed inference requires held-out test samples.")
            if isinstance(probability, bool) or not isinstance(probability, numbers.Real):
                raise CohortMismatch("XGBoost routing scores must be finite numbers.")
            probability = float(probability)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise CohortMismatch("XGBoost routing scores must be probabilities in [0,1].")
            inside = lower <= probability and (probability < upper or (upper == 1.0 and probability == 1.0))
            if not inside:
                raise CohortMismatch("The sample score lies outside the locked probability interval.")
            if sample_id in records:
                raise CohortMismatch("Duplicate sample identifiers are not allowed.")
            records[sample_id] = {"sample_id": sample_id, "text_sha256": digest.lower(),
                                  "p_xgb": probability, "source_split": "test"}
        if not records or len(records) > self.SAMPLE_CAP:
            raise CohortMismatch(f"Authorize between one and {self.SAMPLE_CAP} unique probability-routed samples.")
        if len({row["text_sha256"] for row in records.values()}) != len(records):
            raise CohortMismatch("Duplicate text checksums are not allowed in the cohort.")
        return records

    def _cohort_identity(self) -> Any:
        return {
            "study_type": "probability_routed_hybrid_v1",
            "interval": list(self.interval),
            "upper_inclusive": self.interval[1] == 1.0,
            "cohort": sorted(self.cohort.values(), key=lambda row: row["sample_id"]),
        }
