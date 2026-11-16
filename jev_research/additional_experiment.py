"""Optional, explicitly activated <=500-attempt study; never used by Run All.

The one ledger must be retained for the entire new study. Successful, failed and
uncertain requests share the same cap. No function changes the older ledgers.
The inherited production transport disables retries/redirects and validates the
original fixed score question. Monetary caps require an externally verified
per-request upper bound in a stated currency; unknown pricing is never invented.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

from .budgeted_experiments import (MAX_NEW_JEV_ATTEMPTS, RANDOM_SEED, digest,
    evaluate_policies, fit_policies, choose_threshold, policy_output, metrics,
    save_json, save_table, routing_candidates)
from .client import (BASE_MODEL, BudgetedJevClient, BudgetExhausted,
                     CohortMismatch, AttemptAlreadyRecorded, JevClientError,
                     text_sha256)

STRATA = ("near_threshold", "eligible_negative", "eligible_positive", "outside_control")
QUOTAS = {"development": dict(zip(STRATA, [100, 60, 20, 20])),
          "evaluation": dict(zip(STRATA, [150, 90, 30, 30]))}
MONEY_SCALE = 1_000_000


def monetary_units(value):
    try:
        if isinstance(value, bool):
            raise ValueError("Boolean monetary values are invalid.")
        scaled = Decimal(str(value)) * MONEY_SCALE
        if not scaled.is_finite() or scaled < 0 or scaled != scaled.to_integral_value() or scaled > 2**60:
            raise ValueError("Use nonnegative finite money with at most six decimals.")
        return int(scaled)
    except (InvalidOperation, TypeError):
        raise ValueError("Invalid monetary value.") from None


def score_strata(scores):
    p = np.asarray(scores, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Sampling requires finite baseline scores in [0,1].")
    return np.select([(p >= .4) & (p < .6), (p >= .25) & (p < .4),
                      (p >= .6) & (p < .8)], STRATA[:3], default=STRATA[3])


def prepare_additional_plan(frames, previous_manifests, *, seed=RANDOM_SEED):
    """Select 200 original-validation / 300 original-test rows without labels.

    Exclude every previously selected example, including prior error-study
    train/validation roles and failed requests. All rows of the old test split
    remain previously inspected in aggregate, even after these exclusions.
    No fallback to another stratum or source split is allowed for small pools.
    """
    fields = ["row_id", "text_sha256", "p_xgb", "source_split", "dataset_sha256", "baseline_fingerprint"]
    excluded = set()
    previous = []
    for name, manifest in previous_manifests.items():
        ids = set(manifest.text_sha256)
        excluded.update(ids)
        previous.append((name, ids))
    parts, populations = [], []
    seen_rows, seen_hashes, provenance = set(), set(), None
    for role, split in [("development", "validation"), ("evaluation", "test")]:
        source = frames[split]
        if not set(fields).issubset(source):
            raise ValueError("Full baseline scores and stable provenance are required; do not retrain implicitly.")
        # Discard labels/correctness before any sampling decision.
        pool = source[fields].copy()
        if pool.empty or pool.row_id.duplicated().any() or pool.text_sha256.duplicated().any():
            raise ValueError("Sampling frame must have unique row/text identities.")
        if not pool.source_split.eq(split).all():
            raise ValueError("Sampling frame split mismatch.")
        if seen_rows & set(pool.row_id) or seen_hashes & set(pool.text_sha256):
            raise ValueError("Original validation and test identities overlap.")
        seen_rows.update(pool.row_id)
        seen_hashes.update(pool.text_sha256)
        if pool.dataset_sha256.nunique() != 1 or pool.baseline_fingerprint.nunique() != 1:
            raise ValueError("Ambiguous baseline provenance.")
        identity = (pool.dataset_sha256.iloc[0], pool.baseline_fingerprint.iloc[0])
        if provenance is not None and identity != provenance:
            raise ValueError("Sampling frames use different baseline identities.")
        provenance = identity
        pool["stratum"] = score_strata(pool.p_xgb)
        pool = pool.loc[~pool.text_sha256.isin(excluded)].copy()
        pool["sample_id"] = [text_sha256(f"{d}:{int(r)}:{h}")
                             for d, r, h in zip(pool.dataset_sha256, pool.row_id, pool.text_sha256)]
        pool["selection_priority"] = pool.sample_id.map(lambda s: digest([seed, "additional-500", role, s]))
        for stratum in STRATA:
            candidates = pool.loc[pool.stratum.eq(stratum)].sort_values(["selection_priority", "sample_id"])
            quota, count = QUOTAS[role][stratum], len(candidates)
            if count < quota:
                raise ValueError(f"Insufficient {role}/{stratum}: {count} available; {quota} required.")
            selected = candidates.head(quota).copy()
            selected["study_partition"] = role
            selected["stratum_population_n"] = count
            selected["stratum_sample_n"] = quota
            selected["inclusion_probability"] = quota / count
            selected["prior_baseline_population_inspected"] = True
            for name, ids in previous:
                selected[f"overlap_{name}"] = selected.text_sha256.isin(ids)
            populations.append({"partition": role, "source_split": split, "stratum": stratum,
                                "population_after_exclusions_n": count, "sample_n": quota,
                                "inclusion_probability": quota / count})
            parts.append(selected)
    manifest = pd.concat(parts, ignore_index=True)
    assert len(manifest) == MAX_NEW_JEV_ATTEMPTS
    return manifest, {"status": "manifest_frozen_outcomes_pending", "seed": seed,
                      "manifest_sha256": manifest_hash(manifest), "strata": populations,
                      "excluded_previous_unique_text_n": len(excluded),
                      "interpretation": "exploratory: old validation/test populations were already inspected; exclusions do not establish independent replication",
                      "estimation_scope": "stratum-specific results; weighted estimates may target only the sampling frame AFTER exclusions, not the full dataset"}


def manifest_hash(manifest):
    # Include all selection/design fields; exclude text/labels even if provided.
    columns = [c for c in manifest.columns if c not in {"text", "cleaned_text", "y_true", "y_xgb", "error_type"}]
    return digest(manifest[sorted(columns)].sort_values("sample_id").to_dict("records"))


class AdditionalJevClient(BudgetedJevClient):
    """One durable, fixed-question, <=500-attempt ledger for both phases."""

    ATTEMPT_CAP = MAX_NEW_JEV_ATTEMPTS
    SAMPLE_CAP = MAX_NEW_JEV_ATTEMPTS

    def __init__(self, ledger_path, cohort, *, run_new_inference=False,
                 authorized_manifest_sha256=None, max_attempts=MAX_NEW_JEV_ATTEMPTS,
                 monetary_cap=None, request_cost_upper_bound=None, currency=None,
                 pricing_evidence=None, session=None):
        self.manifest = pd.DataFrame(cohort).copy()
        self.manifest_sha256 = manifest_hash(self.manifest)
        self.enabled = run_new_inference is True
        self.authorization = authorized_manifest_sha256
        if monetary_cap is None:
            if request_cost_upper_bound is not None or currency is not None:
                raise ValueError("Provide cap, verified request upper bound, and currency together.")
            self.money = None
        else:
            if request_cost_upper_bound is None or not currency or not pricing_evidence:
                raise ValueError("Monetary cap requires verified per-request upper-bound pricing and currency; inference refused.")
            cap, bound = monetary_units(monetary_cap), monetary_units(request_cost_upper_bound)
            if bound <= 0:
                raise ValueError("A positive verified request upper bound is required.")
            self.money = {"cap_units": cap, "per_request_upper_units": bound,
                          "currency": str(currency), "pricing_evidence": str(pricing_evidence)}
        super().__init__(ledger_path, self.manifest, allowed_models=(BASE_MODEL,),
                         max_attempts=max_attempts, session=session)

    def _authorize_cohort(self, cohort):
        required = {"sample_id", "text_sha256", "p_xgb", "source_split", "study_partition"}
        if not required.issubset(cohort) or not 1 <= len(cohort) <= self.SAMPLE_CAP:
            raise CohortMismatch("Authorize at most 500 frozen new-study samples.")
        if cohort.sample_id.duplicated().any() or cohort.text_sha256.duplicated().any():
            raise CohortMismatch("Duplicate new-study samples or texts.")
        rows = {}
        for row in cohort.to_dict("records"):
            role = row["study_partition"]
            if role not in ("development", "evaluation") or row["source_split"] != {"development": "validation", "evaluation": "test"}[role]:
                raise CohortMismatch("New development uses validation; evaluation uses test.")
            p = row["p_xgb"]
            if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(p) or not 0 <= p <= 1:
                raise CohortMismatch("Invalid XGBoost sampling score.")
            h, s = row["text_sha256"], row["sample_id"]
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise CohortMismatch("Invalid text checksum.")
            if not isinstance(s, str) or not s or len(s) > 256:
                raise CohortMismatch("Invalid sample identifier.")
            rows[s] = {k: row[k] for k in sorted(required)}
        return rows

    def _cohort_identity(self):
        return {"study": "budgeted_additional_500_v1", "manifest_sha256": self.manifest_sha256,
                "money": self.money, "question": self._fixed_question(), "model": BASE_MODEL,
                "cohort": sorted(self.cohort.values(), key=lambda r: r["sample_id"])}

    @staticmethod
    def _fixed_question():
        from .client import QUESTION
        return QUESTION

    def _initialize(self):
        super()._initialize()
        with self._connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS spending (
                attempt_id INTEGER PRIMARY KEY REFERENCES attempts(id),
                reserved_upper_units INTEGER, elapsed_seconds REAL)""")

    def predict(self, sample_id, text, *, model=BASE_MODEL):
        if model != BASE_MODEL:
            raise CohortMismatch("Prompt/model experiments must not evade the shared new-study budget.")
        if sample_id not in self.cohort or not isinstance(text, str) or text_sha256(text) != self.cohort[sample_id]["text_sha256"]:
            raise CohortMismatch("Text/identity does not match the frozen additional manifest.")
        key = self._request_key(sample_id, text_sha256(text), model)
        with self._connection() as db:
            cached = db.execute("SELECT status FROM attempts WHERE request_key=?", (key,)).fetchone()
        # Cache reads are free even without activation/credentials. Failed or
        # interrupted rows are rejected by the inherited client, never resent.
        if not cached and (not self.enabled or self.authorization != self.manifest_sha256):
            raise JevClientError("New inference disabled or frozen-manifest authorization absent; zero dispatches.")
        if not cached and self.cohort[sample_id]['study_partition'] == 'evaluation':
            with self._connection() as db:
                sealed = db.execute("SELECT value FROM metadata WHERE key='evaluation_decisions_sha256'").fetchone()
            if not sealed:
                raise CohortMismatch("Evaluation dispatch requires sealed development-selected policy decisions.")
        return super().predict(sample_id, text, model=model)

    def _reserve(self, sample_id, text_digest, model):
        key = self._request_key(sample_id, text_digest, model)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute("SELECT status,score FROM attempts WHERE request_key=?", (key,)).fetchone()
                if existing:
                    if existing["status"] != "succeeded":
                        raise AttemptAlreadyRecorded("Failed/uncertain request already counted; no replay.")
                    db.commit()
                    return None, self._validated_saved_score(existing["score"])
                limit = int(db.execute("SELECT value FROM metadata WHERE key='max_attempts'").fetchone()[0])
                if db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] >= limit:
                    raise BudgetExhausted("The shared new-study attempt cap is exhausted.")
                units = None
                if self.money is not None:
                    units = self.money["per_request_upper_units"]
                    used = db.execute("SELECT COALESCE(SUM(reserved_upper_units),0) FROM spending").fetchone()[0]
                    if used + units > self.money["cap_units"]:
                        raise BudgetExhausted("The conservative monetary reservation would exceed the cap.")
                # Attempt and maximum charge reservation commit atomically,
                # before the inherited transport can dispatch any request.
                cur = db.execute("""INSERT INTO attempts
                    (request_key,sample_id,text_sha256,model,instruction_sha256,status,started_at)
                    VALUES (?,?,?,?,?,'pending',?)""",
                    (key, sample_id, text_digest, model, self.instruction_fingerprint,
                     datetime.now(timezone.utc).isoformat()))
                attempt_id = int(cur.lastrowid)
                db.execute("INSERT INTO spending(attempt_id,reserved_upper_units) VALUES (?,?)", (attempt_id, units))
                db.commit()
                return attempt_id, None
            except BaseException:
                db.rollback()
                raise

    def _finish(self, attempt_id, *, score=None, error_code=None):
        super()._finish(attempt_id, score=score, error_code=error_code)
        with self._connection() as db:
            times = db.execute("SELECT started_at,finished_at FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            elapsed = (datetime.fromisoformat(times["finished_at"]) - datetime.fromisoformat(times["started_at"])).total_seconds()
            db.execute("UPDATE spending SET elapsed_seconds=? WHERE attempt_id=?", (max(0., elapsed), attempt_id))

    def ledger_summary(self):
        summary = super().ledger_summary()
        with self._connection() as db:
            units, timed, seconds = db.execute("SELECT SUM(reserved_upper_units),COUNT(elapsed_seconds),SUM(elapsed_seconds) FROM spending").fetchone()
            attempts = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            spending = db.execute("SELECT COUNT(*) FROM spending").fetchone()[0]
        if attempts != spending:
            raise JevClientError("Request/spending accounting mismatch; preserve the ledger for diagnosis.")
        summary.update({"monetary_policy": self.money,
                        "reserved_upper_cost": units / MONEY_SCALE if units is not None else None,
                        "measured_currency_cost": None, "pricing_status": "unknown" if self.money is None else "externally supplied verified upper bound; not measured spending",
                        "timed_attempts": timed, "mean_completed_elapsed_seconds": seconds / timed if timed else None})
        return summary


def attach_outcomes(manifest, frames, records):
    """Attach true labels only for analysis; raw text never goes into exports."""
    source = pd.concat([frames["validation"], frames["test"]], ignore_index=True)
    if source.text_sha256.duplicated().any():
        raise ValueError("Source frames must remain text-disjoint.")
    source = source.set_index("text_sha256")
    result = manifest.copy()
    labels = source.loc[result.text_sha256, "y_true"].to_numpy()
    result["y_true"] = labels
    lookup = {r["sample_id"]: r for r in records}
    result["jev_score"] = result.sample_id.map(lambda s: lookup.get(s, {}).get("score"))
    result["jev_status"] = result.sample_id.map(lambda s: lookup.get(s, {}).get("status", "not_attempted"))
    return result


def freeze_additional_policies(manifest, frames, client, output_path):
    development = manifest.loc[manifest.study_partition.eq("development")]
    table = attach_outcomes(development, frames, client.ledger_records())
    if table.jev_status.eq("not_attempted").any():
        raise ValueError("Complete/resolve development attempts before freezing evaluation decisions.")
    with client._connection() as db:
        if db.execute("SELECT value FROM metadata WHERE key='evaluation_decisions_sha256'").fetchone():
            raise ValueError("Evaluation already started; policy decisions are sealed.")
    specs, thresholds, selection = fit_policies(table)
    routing_specs = routing_candidates()
    routing = evaluate_policies(table, routing_specs, dict.fromkeys(routing_specs, .5))
    routing_choices = {}
    for ratio in [1, 5, 10]:
        ranked = routing.assign(loss=ratio*routing.fn+routing.fp).sort_values(
            ['loss', 'prospective_requests', 'policy'], kind='stable')
        name = ranked.iloc[0].policy
        routing_choices[str(ratio)] = {'policy': name, 'specification': routing_specs[name]}
    frozen = {"manifest_sha256": manifest_hash(manifest), "development_ids_sha256": digest(sorted(table.sample_id)),
              "specifications": specs, "thresholds": thresholds, "selection": selection,
              "routing_cost_choices": routing_choices,
              "matched_recall_choices": {f"{name}:{target}": choose_threshold(table, spec, target)
                                         for name, spec in specs.items() for target in [.9, .95]}}
    save_json(output_path, frozen)
    return frozen


def collect_additional_phase(manifest, frames, *, phase, ledger_path,
                             run_new_inference=False, authorized_manifest_sha256=None,
                             frozen_decisions=None, max_attempts=MAX_NEW_JEV_ATTEMPTS,
                             monetary_cap=None, request_cost_upper_bound=None,
                             currency=None, pricing_evidence=None, session=None):
    if run_new_inference is not True:
        return {"status": "disabled", "new_paid_attempts": 0}
    if phase not in ("development", "evaluation"):
        raise ValueError("Run development then freeze decisions, then evaluation.")
    if authorized_manifest_sha256 != manifest_hash(manifest):
        raise ValueError("Explicit frozen-manifest authorization is required.")
    # Validate every selected source text and score before reserving any attempt.
    sources = pd.concat([frames["validation"], frames["test"]], ignore_index=True)
    if sources.text_sha256.duplicated().any():
        raise ValueError("Source texts overlap.")
    sources = sources.set_index("text_sha256")
    for row in manifest.itertuples(index=False):
        s = sources.loc[row.text_sha256]
        for field in ["row_id", "source_split", "dataset_sha256", "baseline_fingerprint"]:
            if s[field] != getattr(row, field):
                raise CohortMismatch("Additional source identity/provenance changed.")
        if not math.isclose(float(s.p_xgb), row.p_xgb, rel_tol=0, abs_tol=1e-14) or text_sha256(s.text) != row.text_sha256:
            raise CohortMismatch("Additional source score or raw text changed.")
    client = AdditionalJevClient(ledger_path, manifest, run_new_inference=True,
        authorized_manifest_sha256=authorized_manifest_sha256, max_attempts=max_attempts,
        monetary_cap=monetary_cap, request_cost_upper_bound=request_cost_upper_bound,
        currency=currency, pricing_evidence=pricing_evidence, session=session)
    recorded = {r["sample_id"] for r in client.ledger_records()}
    if phase == "evaluation":
        dev_ids = set(manifest.loc[manifest.study_partition.eq("development"), "sample_id"])
        if not dev_ids.issubset(recorded) or not frozen_decisions:
            raise ValueError("Evaluation requires completed development and frozen policy decisions.")
        if frozen_decisions.get("manifest_sha256") != client.manifest_sha256 or frozen_decisions.get("development_ids_sha256") != digest(sorted(dev_ids)):
            raise ValueError("Frozen decisions do not match the new-study development manifest.")
        decision_hash = digest(frozen_decisions)
        with client._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT value FROM metadata WHERE key='evaluation_decisions_sha256'").fetchone()
            if old and old["value"] != decision_hash:
                db.rollback()
                raise ValueError("Evaluation decisions changed after dispatch began.")
            db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES ('evaluation_decisions_sha256',?)", (decision_hash,))
            db.commit()
    before = client.ledger_summary()["attempts_spent"]
    for row in manifest.loc[manifest.study_partition.eq(phase)].itertuples(index=False):
        if row.sample_id in recorded:
            continue  # No resending successes, failures, or uncertain interruptions.
        client.predict(row.sample_id, sources.loc[row.text_sha256, "text"])
        # Provider failure propagates immediately: deliberate diagnosis/resume required.
    return {**client.ledger_summary(), "new_paid_attempts_this_run": client.ledger_summary()["attempts_spent"] - before}


def analyze_additional(manifest, frames, client, frozen_decisions, results):
    with client._connection() as db:
        sealed = db.execute("SELECT value FROM metadata WHERE key='evaluation_decisions_sha256'").fetchone()
    if not sealed or sealed["value"] != digest(frozen_decisions):
        raise ValueError("Evaluation decisions were not sealed before dispatch.")
    table = attach_outcomes(manifest, frames, client.ledger_records())
    evaluation = table.loc[table.study_partition.eq("evaluation")]
    if evaluation.jev_status.eq("not_attempted").any():
        raise ValueError("Additional evaluation is incomplete; do not report a completed study.")
    result_rows = []
    specs, thresholds = frozen_decisions["specifications"], frozen_decisions["thresholds"]
    for stratum, rows in evaluation.groupby("stratum"):
        result_rows.append(evaluate_policies(rows, specs, thresholds).assign(stratum=stratum))
    result = pd.concat(result_rows, ignore_index=True)
    Path(results).mkdir(parents=True, exist_ok=True)
    save_table(results, "additional_stratum_evaluation", result)
    pooled = evaluate_policies(evaluation, specs, thresholds)
    pooled["scope"] = "unweighted oversampled 300-case subset only; NOT full-population performance"
    save_table(results, "additional_oversampled_subset", pooled)
    matched_rows = []
    for name, spec in specs.items():
        if spec['kind'] == 'xgb' and name != 'XGB@0.50':
            continue
        for target in [.9, .95]:
            choice = frozen_decisions['matched_recall_choices'][f'{name}:{target}']
            if not choice['achievable']:
                matched_rows.append({'policy': name, 'target': target, 'achievable': False})
                continue
            s, pred, needs = policy_output(evaluation, spec, choice['threshold'])
            display_name = 'XGB threshold (target)' if name == 'XGB@0.50' else name
            matched_rows.append({'policy': display_name, **choice,
                                 **metrics(evaluation.y_true, pred, None if spec['kind'] in ('or', 'and') else s),
                                 'prospective_requests': int(needs.sum()),
                                 'scope': 'oversampled subset; NOT full population'})
    save_table(results, 'additional_matched_recall', pd.DataFrame(matched_rows))
    routing_rows = []
    for ratio, choice in frozen_decisions['routing_cost_choices'].items():
        spec = choice['specification']
        s, pred, needs = policy_output(evaluation, spec)
        routing_rows.append({'fn_to_fp_cost_ratio': int(ratio), 'policy': choice['policy'],
                             **metrics(evaluation.y_true, pred, s),
                             'prospective_requests': int(needs.sum()),
                             'scope': 'oversampled subset; NOT full population'})
    save_table(results, 'additional_selected_routing', pd.DataFrame(routing_rows))
    save_table(results, "additional_predictions", table.drop(columns=["text", "cleaned_text"], errors="ignore"))
    return result


def write_pending_plan(directory):
    """A concrete design, not a fabricated manifest when full scores are absent."""
    plan = {"status": "pending_full_baseline_artifacts_and_explicit_activation",
            "run_new_jev_inference": False, "max_new_jev_attempts": MAX_NEW_JEV_ATTEMPTS,
            "new_paid_attempts": 0, "seed": RANDOM_SEED, "quotas": QUOTAS,
            "strata": {"near_threshold": "0.40 <= XGB < 0.60", "eligible_negative": "0.25 <= XGB < 0.40",
                       "eligible_positive": "0.60 <= XGB < 0.80", "outside_control": "XGB < 0.25 or XGB >= 0.80"},
            "required_artifacts": ["original validation/test per-row XGBoost scores with row/text/dataset/model hashes",
                                   "matching original texts and binary labels for analysis"],
            "manifest_status": "not generated: full sampling frame absent; IDs, population counts, and inclusion probabilities are not invented",
            "generator": "jev_research.additional_experiment.prepare_additional_plan",
            "ledger": "artifacts/jev_probability_hybrid_budgeted_experiments/budget.sqlite3",
            "execution": "collect_additional_phase(development) -> freeze_additional_policies -> collect_additional_phase(evaluation) -> analyze_additional",
            "authorization": "Boolean activation AND exact frozen-manifest digest required. Allowance alone never activates dispatch.",
            "money": "Unknown; no currency amounts invented. If a monetary cap is supplied, also require verified upper-bound price per attempt and currency, or refuse inference. Reservations for failures/interruptions are retained.",
            "sampling": "Hash-ranked within mutually exclusive score strata, no labels/correctness. Exclude all prior probability/error-study selected texts. Fail if quotas unavailable; no automatic expansion.",
            "overlap": "Record selected-manifest overlap and original baseline-population exposure. Remaining old test texts are still exploratory.",
            "analysis": "Stratum-specific metrics and explicitly oversampled-subset pooled metrics. No full-population estimate from unweighted samples.",
            "budget_scope": "One persistent ledger for all development/evaluation calls, failures, and interruptions. No alternate prompts/models or retry loops. Do not delete or replace ledger to reset the cap."}
    save_json(Path(directory) / "additional_experiment_plan.json", plan)
    return plan
