"""Offline, exploratory comparisons on the historical probability-study cache.

This module has no inference path. Public prediction exports are evidence, not a
replacement for the absent private ledger. Policies are fitted on a label-blind
hash development partition and frozen before evaluating its disjoint complement.
Neither partition is an independent replication of the already inspected study.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .client import BASE_MODEL, ENDPOINT, QUESTION

RANDOM_SEED = 42
MAX_NEW_JEV_ATTEMPTS = 500
RUN_NEW_JEV_INFERENCE = False
STUDY_NAME = "jev_probability_hybrid_budgeted_experiments"
INTERVALS = {"original": (0.25, 0.80), "narrow": (0.40, 0.60),
             "below": (0.25, 0.50), "above": (0.50, 0.80)}
BOOTSTRAP_REPLICATES = 4000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_table(directory, name, frame):
    # JSON tables remain tracked under the repository's global *.csv ignore rule.
    frame.to_json(Path(directory) / (name + ".json"), orient="records", indent=2,
                  double_precision=15)


def validate_cache(frame):
    required = {"sample_id", "row_id", "text_sha256", "p_xgb", "y_true",
                "jev_score", "jev_status", "source_split", "dataset_sha256",
                "baseline_fingerprint"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("Incomplete prediction evidence.")
    for col in ("sample_id", "row_id", "text_sha256"):
        if frame[col].isna().any() or frame[col].duplicated().any():
            raise ValueError(f"Duplicate/missing cache identity: {col}")
    if not frame.y_true.isin([0, 1]).all() or not frame.source_split.eq("test").all():
        raise ValueError("Require binary positive=1 labels from the original test split.")
    x = frame.p_xgb.to_numpy(float)
    if not np.isfinite(x).all() or not ((0.25 <= x) & (x < 0.80)).all():
        raise ValueError("Cache scores must lie within the historical routing interval.")
    if not frame.jev_status.isin(["succeeded", "failed", "pending", "not_attempted"]).all():
        raise ValueError("Unknown request status.")
    ok = frame.jev_status.eq("succeeded")
    j = frame.loc[ok, "jev_score"].to_numpy(float)
    if not np.isfinite(j).all() or not ((0 <= j) & (j <= 1)).all():
        raise ValueError("Invalid succeeded JEV score.")
    if frame.loc[~ok, "jev_score"].notna().any():
        raise ValueError("An unavailable response must not carry a score.")
    for col in ("dataset_sha256", "baseline_fingerprint"):
        if frame[col].nunique() != 1:
            raise ValueError("Ambiguous source provenance.")
    expected = [hashlib.sha256(f"{d}:{int(r)}:{h}".encode()).hexdigest()
                for d, r, h in zip(frame.dataset_sha256, frame.row_id, frame.text_sha256)]
    if expected != frame.sample_id.tolist():
        raise ValueError("Sample IDs disagree with row/dataset/text provenance.")


def split_cache(frame, development_n=2000, seed=RANDOM_SEED):
    if not 0 < development_n < len(frame):
        raise ValueError("Both development and evaluation must be nonempty.")
    result = frame.copy()
    result["partition_hash"] = result.sample_id.map(
        lambda s: hashlib.sha256(f"{seed}:budgeted-cache-split:{s}".encode()).hexdigest())
    result = result.sort_values(["partition_hash", "sample_id"], kind="stable").reset_index(drop=True)
    result["partition"] = np.where(np.arange(len(result)) < development_n,
                                   "development", "evaluation")
    a, b = (result.loc[result.partition.eq(p)].copy() for p in ["development", "evaluation"])
    for col in ("sample_id", "row_id", "text_sha256"):
        if set(a[col]) & set(b[col]):
            raise ValueError("Development and evaluation overlap.")
    return result, a, b


def counts_metrics(tn, fp, fn, tp):
    n, positive, negative = tn + fp + fn + tp, tp + fn, tn + fp
    ratio = lambda a, b: float(a / b) if b else 0.0
    recall, specificity = ratio(tp, positive), ratio(tn, negative)
    f1, f1_negative = ratio(2 * tp, 2 * tp + fp + fn), ratio(2 * tn, 2 * tn + fp + fn)
    return {"n": int(n), "positive_n": int(positive), "negative_n": int(negative),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": ratio(tp + tn, n), "balanced_accuracy": (recall + specificity) / 2,
            "precision": ratio(tp, tp + fp), "recall": recall, "specificity": specificity,
            "f1": f1, "f2": ratio(5 * tp, 5 * tp + 4 * fn + fp),
            "macro_f1": (f1 + f1_negative) / 2,
            "misses_per_1000": ratio(1000 * fn, n), "false_alarms_per_1000": ratio(1000 * fp, n)}


def metrics(y, predicted, scores=None):
    y, predicted = np.asarray(y, int), np.asarray(predicted, bool)
    if not len(y) or len(y) != len(predicted) or not np.isin(y, [0, 1]).all():
        raise ValueError("Require aligned nonempty binary arrays.")
    result = counts_metrics(np.sum((y == 0) & ~predicted), np.sum((y == 0) & predicted),
                            np.sum((y == 1) & ~predicted), np.sum((y == 1) & predicted))
    result.update({k: None for k in ("roc_auc", "average_precision", "brier", "log_loss")})
    if scores is not None:
        scores = np.asarray(scores, float)
        if scores.shape != y.shape or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("Invalid aligned scores.")
        if len(np.unique(y)) == 2:
            result["roc_auc"] = float(roc_auc_score(y, scores))
            result["average_precision"] = float(average_precision_score(y, scores))
        result["brier"] = float(brier_score_loss(y, scores))
        result["log_loss"] = float(log_loss(y, scores, labels=[0, 1]))
    return result


def transitions(y, base, other):
    y, base, other = np.asarray(y, int), np.asarray(base, bool), np.asarray(other, bool)
    b, o = base == y, other == y
    corrected, introduced = int(np.sum(~b & o)), int(np.sum(b & ~o))
    fn_recovered = int(np.sum((y == 1) & ~base & other))
    fn_new = int(np.sum((y == 1) & base & ~other))
    fp_corrected = int(np.sum((y == 0) & base & ~other))
    fp_new = int(np.sum((y == 0) & ~base & other))
    assert corrected - introduced == int(o.sum() - b.sum())
    assert corrected == fn_recovered + fp_corrected and introduced == fn_new + fp_new
    exact = lambda a, b: float(binomtest(min(a, b), a + b, 0.5).pvalue) if a + b else 1.0
    return {"corrected": corrected, "introduced": introduced, "net_correct": corrected - introduced,
            "fn_recovered": fn_recovered, "fn_introduced": fn_new,
            "fp_corrected": fp_corrected, "fp_introduced": fp_new,
            "net_fn_reduction": fn_recovered - fn_new, "net_fp_reduction": fp_corrected - fp_new,
            "accuracy_change_pp": 100 * (corrected - introduced) / len(y),
            "accuracy_relative_improvement_pct": 100 * (corrected - introduced) / b.sum(),
            "mcnemar_accuracy_p": exact(corrected, introduced),
            "mcnemar_sensitivity_p": exact(fn_recovered, fn_new)}


def policy_output(frame, spec, threshold=0.5):
    """Return scores, decisions and prospective requests, without inference.

    Unrouted cases keep XGB@0.50. An unavailable required JEV result also keeps
    XGB@0.50, including for matched-recall experiments. OR/AND short-circuit on
    the XGB score at the policy threshold. Score diagnostics retain raw XGB on
    missing results and are explicitly distinct from this hard fallback rule.
    """
    x = frame.p_xgb.to_numpy(float)
    ok = frame.jev_status.eq("succeeded").to_numpy()
    j = frame.jev_score.fillna(frame.p_xgb).to_numpy(float)
    kind = spec["kind"]
    lo, hi = spec.get("interval", [0, 1.000001])
    route = (x >= lo) & (x < hi)
    needs = np.zeros(len(frame), bool)
    if kind == "xgb":
        s = x.copy()
    elif kind in ("xgb_sigmoid", "jev_sigmoid", "logistic"):
        if kind == "xgb_sigmoid":
            features = logit(np.clip(x, 1e-6, 1 - 1e-6))[:, None]
        elif kind == "jev_sigmoid":
            features = logit(np.clip(j, 1e-6, 1 - 1e-6))[:, None]
        else:
            features = np.column_stack([logit(np.clip(x, 1e-6, 1 - 1e-6)),
                                        logit(np.clip(j, 1e-6, 1 - 1e-6))])
        s = expit(features @ np.asarray(spec["coefficients"]) + spec["intercept"])
        if kind != "xgb_sigmoid":
            needs = route.copy()
    elif kind == "replacement":
        s, needs = j.copy(), route.copy()
    elif kind == "or":
        s, needs = np.maximum(x, j), route & (x < threshold)
    elif kind == "and":
        s, needs = np.minimum(x, j), route & (x >= threshold)
    elif kind == "weighted":
        alpha = spec["jev_weight"]
        if not 0 < alpha < 1:
            raise ValueError("Use a genuine two-score weighted average.")
        s, needs = (1 - alpha) * x + alpha * j, route.copy()
    else:
        raise ValueError(f"Unknown policy kind: {kind}")
    if kind not in ("xgb", "xgb_sigmoid"):
        s = np.where(route & ok, s, x)
    predicted = s >= threshold
    if kind not in ("xgb", "xgb_sigmoid"):
        predicted = np.where(~route | (needs & ~ok), x >= 0.5, predicted)
    return s, predicted, needs


def choose_threshold(development, spec, target=None, objective="f2"):
    # Candidate cutoffs are development score values only, with exact >= ties.
    # Count threshold-dependent short circuits/fallbacks for OR and AND too.
    s, _, _ = policy_output(development, spec)
    thresholds = np.unique(np.r_[0.0, s, 0.5, np.nextafter(1.0, 2.0)])
    y = development.y_true.to_numpy()
    rows = []
    for t in thresholds:
        _, pred, needs = policy_output(development, spec, float(t))
        tp, fp = int(np.sum((y == 1) & pred)), int(np.sum((y == 0) & pred))
        fn, tn = int(np.sum(y == 1)) - tp, int(np.sum(y == 0)) - fp
        row = counts_metrics(tn, fp, fn, tp)
        rows.append({"threshold": float(t), "prospective_requests": int(needs.sum()), **row})
    if target is None:
        # A strong sensitivity-aware baseline: maximize development F2.
        winner = min(rows, key=lambda r: (-r[objective], -r["precision"], abs(r["threshold"] - .5), r["threshold"]))
    else:
        feasible = [r for r in rows if r["recall"] >= target]
        if not feasible:
            return {"threshold": None, "target": target, "achievable": False,
                    "max_development_recall": max(r["recall"] for r in rows)}
        winner = min(feasible, key=lambda r: (r["fp"], -r["precision"], -r["threshold"]))
    return {"threshold": winner["threshold"], "target": target, "achievable": True,
            "development_recall": winner["recall"], "development_precision": winner["precision"],
            "development_fp": winner["fp"], "development_f2": winner["f2"],
            "development_f1": winner["f1"], "objective": objective if target is None else "minimize FP subject to recall"}


def fit_policies(development):
    if set(development.y_true.unique()) != {0, 1}:
        raise ValueError("Development requires both classes.")
    specs = {"XGB@0.50": {"kind": "xgb"}, "JEV replacement": {"kind": "replacement"},
             "OR": {"kind": "or"}, "AND": {"kind": "and"},
             "Mean 50:50": {"kind": "weighted", "jev_weight": .5}}
    weight_trials = []
    for w in [.25, .5, .75]:
        spec = {"kind": "weighted", "jev_weight": w}
        s, _, _ = policy_output(development, spec)
        weight_trials.append({"jev_weight": w, "development_log_loss": float(log_loss(development.y_true, s))})
    weight = min(weight_trials, key=lambda r: (r["development_log_loss"], abs(r["jev_weight"] - .5)))["jev_weight"]
    specs["Weighted (dev)"] = {"kind": "weighted", "jev_weight": weight}
    for name, kind in [("XGB sigmoid", "xgb_sigmoid"), ("JEV sigmoid", "jev_sigmoid"),
                       ("Logistic fusion", "logistic")]:
        train = development if kind == "xgb_sigmoid" else development.loc[development.jev_status.eq("succeeded")]
        x, j = train.p_xgb.to_numpy(), train.jev_score.to_numpy()
        lx, lj = logit(np.clip(x, 1e-6, 1-1e-6)), logit(np.clip(j, 1e-6, 1-1e-6))
        features = lx[:, None] if kind == "xgb_sigmoid" else lj[:, None] if kind == "jev_sigmoid" else np.column_stack([lx, lj])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        model.fit(features, train.y_true)
        specs[name] = {"kind": kind, "coefficients": model.coef_[0].tolist(),
                       "intercept": float(model.intercept_[0]), "C": 1.0,
                       "development_training_n": len(train)}
    thresholds = {"XGB@0.50": .5}
    for name in specs:
        thresholds.setdefault(name, .5)
    selected = choose_threshold(development, specs["XGB@0.50"])
    specs["XGB threshold (dev F2)"] = {"kind": "xgb"}
    thresholds["XGB threshold (dev F2)"] = selected["threshold"]
    selected_f1 = choose_threshold(development, {"kind": "xgb"}, objective="f1")
    specs["XGB threshold (dev F1)"] = {"kind": "xgb"}
    thresholds["XGB threshold (dev F1)"] = selected_f1["threshold"]
    return specs, thresholds, {"weight_trials": weight_trials, "xgb_f2_threshold": selected,
                              "xgb_f1_threshold": selected_f1}


def evaluate_policies(frame, specs, thresholds):
    rows = []
    base = frame.p_xgb.to_numpy() >= .5
    for name, spec in specs.items():
        s, pred, needs = policy_output(frame, spec, thresholds[name])
        unavailable = ~frame.jev_status.eq("succeeded").to_numpy()
        rows.append({"policy": name, "threshold": thresholds[name],
                     **metrics(frame.y_true, pred, None if spec["kind"] in ("or", "and") else s),
                     **transitions(frame.y_true, base, pred),
                     "prospective_requests": int(needs.sum()), "routing_fraction": float(needs.mean()),
                     "required_unavailable_fallbacks": int(np.sum(needs & unavailable)),
                     "new_paid_attempts": 0})
    return pd.DataFrame(rows)


def matched_recall(development, evaluation, specs, choices=None):
    rows, frozen = [], {}
    for target in [.90, .95]:
        for name, spec in specs.items():
            if spec["kind"] == "xgb" and name != "XGB@0.50":
                continue
            choice = choices[f"{name}:{target}"] if choices is not None else choose_threshold(development, spec, target)
            frozen[f"{name}:{target}"] = choice
            if not choice["achievable"]:
                rows.append({"policy": name, "target": target, "achievable": False})
                continue
            s, pred, needs = policy_output(evaluation, spec, choice["threshold"])
            display_name = "XGB threshold (target)" if name == "XGB@0.50" else name
            rows.append({"policy": display_name, **choice,
                         **metrics(evaluation.y_true, pred, None if spec["kind"] in ("or", "and") else s),
                         "prospective_requests": int(needs.sum()),
                         "required_unavailable_fallbacks": int(np.sum(needs & ~evaluation.jev_status.eq("succeeded")))})
    return pd.DataFrame(rows), frozen


def matched_uncertainty(development, evaluation, specs, frozen):
    rows = []
    for target in [.90, .95]:
        reference = frozen[f"XGB@0.50:{target}"]
        if not reference["achievable"]:
            continue
        _, base, _ = policy_output(evaluation, {"kind": "xgb"}, reference["threshold"])
        for name, spec in specs.items():
            if spec["kind"] in ("xgb", "xgb_sigmoid"):
                continue
            choice = frozen[f"{name}:{target}"]
            if not choice["achievable"]:
                continue
            _, pred, _ = policy_output(evaluation, spec, choice["threshold"])
            ci = paired_bootstrap(evaluation.y_true, base, pred)
            row = {"policy": name, "development_target_recall": target,
                   "reference": "XGB threshold selected at same development recall target",
                   **transitions(evaluation.y_true, base, pred)}
            for metric, item in ci.items():
                row[f"delta_{metric}"] = item["difference"]
                row[f"delta_{metric}_ci_low"], row[f"delta_{metric}_ci_high"] = item["ci95"]
            rows.append(row)
    result = pd.DataFrame(rows)
    for metric in ["accuracy", "sensitivity"]:
        result[f"mcnemar_{metric}_holm_p"] = holm(result[f"mcnemar_{metric}_p"].tolist())
    return result


def routing_candidates():
    specs = {"No JEV": {"kind": "xgb"}}
    for interval, bounds in INTERVALS.items():
        for kind in ["replacement", "or", "weighted"]:
            specs[f"{interval}/{kind}"] = {"kind": kind, "interval": list(bounds),
                                           **({"jev_weight": .5} if kind == "weighted" else {})}
    return specs


def select_routing(development):
    specs = routing_candidates()
    dev = evaluate_policies(development, specs, dict.fromkeys(specs, .5))
    # Selection happens before evaluation; no threshold/bin search on evaluation.
    frozen = {}
    for ratio in [1, 5, 10]:
        choices = dev.assign(hypothetical_loss=ratio * dev.fn + dev.fp)
        best = choices.sort_values(["hypothetical_loss", "prospective_requests", "policy"], kind="stable").iloc[0]
        frozen[str(ratio)] = {"policy": best.policy, "development_loss": int(best.hypothetical_loss),
                              "specification": specs[best.policy]}
    return dev, frozen


def routing_analysis(development, evaluation, selection=None):
    specs = routing_candidates()
    dev, frozen = select_routing(development) if selection is None else selection
    ev = evaluate_policies(evaluation, specs, dict.fromkeys(specs, .5))
    for frame in [dev, ev]:
        frame["baseline_error_rate"] = (frame.fn + frame.fp + frame.net_correct) / frame.n
        frame["net_correct_per_request"] = frame.net_correct.div(frame.prospective_requests.replace(0, np.nan))
        for ratio in [1, 5, 10]:
            frame[f"loss_fn_cost_{ratio}"] = (ratio * frame.fn + frame.fp) / frame.n
    selected = []
    for ratio, choice in frozen.items():
        selected.append({"fn_to_fp_cost_ratio": int(ratio), **ev.loc[ev.policy.eq(choice["policy"])].iloc[0].to_dict(),
                         "development_selected_loss": choice["development_loss"]})
    return dev, ev, pd.DataFrame(selected), frozen


def descriptive_bins(frame):
    rows = []
    for partition, source in frame.groupby("partition", sort=True):
        for i in range(5, 16):
            lo, hi = i / 20, (i + 1) / 20
            bin_frame = source.loc[(source.p_xgb >= lo) & (source.p_xgb < hi)]
            if bin_frame.empty:
                continue
            x = bin_frame.p_xgb.to_numpy() >= .5
            _, pred, needs = policy_output(bin_frame, {"kind": "replacement"})
            transition = transitions(bin_frame.y_true, x, pred)
            rows.append({"partition": partition, "lower": lo, "upper": hi, "n": len(bin_frame),
                         "baseline_errors": int(np.sum(x != bin_frame.y_true)),
                         "baseline_error_rate": float(np.mean(x != bin_frame.y_true)),
                         **transition, "successful_responses": int(bin_frame.jev_status.eq("succeeded").sum()),
                         "net_correct_per_request": transition["net_correct"] / int(needs.sum())})
    return pd.DataFrame(rows)


def paired_bootstrap(y, base, other, replicates=BOOTSTRAP_REPLICATES, seed=RANDOM_SEED):
    """Exact empirical paired row bootstrap via the eight joint outcome counts.

    Multinomial resampling is equivalent to sampling row indices with replacement
    for these confusion-based metrics. It preserves within-row dependence.
    """
    y, base, other = np.asarray(y, int), np.asarray(base, int), np.asarray(other, int)
    observed = np.bincount(4 * y + 2 * base + other, minlength=8)
    draws = np.random.default_rng(seed).multinomial(len(y), observed / len(y), size=replicates)
    truth = np.arange(8) // 4
    b = (np.arange(8) // 2) % 2
    o = np.arange(8) % 2

    def stats(pred):
        tn = draws[:, (truth == 0) & (pred == 0)].sum(axis=1)
        fp = draws[:, (truth == 0) & (pred == 1)].sum(axis=1)
        fn = draws[:, (truth == 1) & (pred == 0)].sum(axis=1)
        tp = draws[:, (truth == 1) & (pred == 1)].sum(axis=1)
        ratio = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        return {"accuracy": (tp + tn) / len(y), "recall": ratio(tp, tp + fn),
                "specificity": ratio(tn, tn + fp), "precision": ratio(tp, tp + fp),
                "f2": ratio(5*tp, 5*tp+4*fn+fp)}
    bs, os = stats(b), stats(o)
    point_b, point_o = metrics(y, base), metrics(y, other)
    return {metric: {"difference": point_o[metric] - point_b[metric],
                     "ci95": np.quantile(os[metric] - bs[metric], [.025, .975]).tolist()}
            for metric in bs}


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, previous = np.empty(len(order)), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def uncertainty(evaluation, specs, thresholds):
    rows = []
    references = ["XGB@0.50", "XGB threshold (dev F1)", "XGB threshold (dev F2)"]
    for reference in references:
        _, base, _ = policy_output(evaluation, specs[reference], thresholds[reference])
        for name, spec in specs.items():
            if name in references:
                continue
            _, pred, _ = policy_output(evaluation, spec, thresholds[name])
            paired = transitions(evaluation.y_true, base, pred)
            ci = paired_bootstrap(evaluation.y_true, base, pred)
            row = {"reference": reference, "policy": name, **paired}
            for metric, item in ci.items():
                row[f"delta_{metric}"] = item["difference"]
                row[f"delta_{metric}_ci_low"], row[f"delta_{metric}_ci_high"] = item["ci95"]
            rows.append(row)
    result = pd.DataFrame(rows)
    for suffix in ["accuracy", "sensitivity"]:
        result[f"mcnemar_{suffix}_holm_p"] = holm(result[f"mcnemar_{suffix}_p"].tolist())
    return result


def reliability_table(evaluation, specs):
    rows = []
    for name in ["XGB@0.50", "JEV replacement", "Mean 50:50", "XGB sigmoid", "JEV sigmoid", "Logistic fusion"]:
        s, _, _ = policy_output(evaluation, specs[name])
        bins = np.minimum((s * 10).astype(int), 9)
        for i in range(10):
            select = bins == i
            if np.any(select):
                rows.append({"policy": name, "lower": i/10, "upper": (i+1)/10,
                             "n": int(select.sum()), "mean_score": float(s[select].mean()),
                             "observed_positive_fraction": float(evaluation.y_true.to_numpy()[select].mean())})
    return pd.DataFrame(rows)


def score_loss_uncertainty(evaluation, specs, replicates=2000):
    """Paired percentile bootstrap of additive proper-score loss differences."""
    y = evaluation.y_true.to_numpy()
    pairs = [("JEV replacement", "XGB@0.50"), ("JEV sigmoid", "JEV replacement"),
             ("XGB sigmoid", "XGB@0.50"), ("Logistic fusion", "XGB@0.50"),
             ("Logistic fusion", "JEV sigmoid")]
    differences, descriptors = [], []
    for policy, reference in pairs:
        a = policy_output(evaluation, specs[policy])[0]
        b = policy_output(evaluation, specs[reference])[0]
        for loss in ["brier", "log_loss"]:
            def individual(s):
                if loss == "brier":
                    return (s - y) ** 2
                s = np.clip(s, np.finfo(float).eps, 1-np.finfo(float).eps)
                return -(y*np.log(s) + (1-y)*np.log1p(-s))
            differences.append(individual(a)-individual(b))
            descriptors.append({"policy": policy, "reference": reference, "loss": loss})
    delta = np.asarray(differences).T
    rng = np.random.default_rng(RANDOM_SEED)
    means = []
    for start in range(0, replicates, 50):
        indexes = rng.integers(0, len(y), size=(min(50, replicates-start), len(y)))
        means.append(delta[indexes].mean(axis=1))
    means = np.concatenate(means)
    rows = []
    for i, descriptor in enumerate(descriptors):
        low, high = np.quantile(means[:, i], [.025, .975])
        rows.append({**descriptor, "difference": float(delta[:, i].mean()),
                     "ci_low": float(low), "ci_high": float(high), "replicates": replicates})
    return pd.DataFrame(rows)


def boundary_ranking(frame):
    # The uncovered 41,342 scores cannot be reconstructed. Restrict pairwise
    # boundary diagnostics to actual cached rows just inside/outside .40/.60.
    rows = []
    x, y = frame.p_xgb.to_numpy(), frame.y_true.to_numpy()
    ok = frame.jev_status.eq("succeeded").to_numpy()
    j = frame.jev_score.fillna(frame.p_xgb).to_numpy()
    for boundary in [.25, .4, .5, .6, .8]:
        left = (x >= boundary - .05) & (x < boundary)
        right = (x >= boundary) & (x < boundary + .05)
        n_pairs = int(left.sum() * right.sum())
        for name in ["baseline", "replacement below boundary", "replacement above boundary"]:
            score = x.copy()
            if name != "baseline":
                change = left if name == "replacement below boundary" else right
                score[change & ok] = j[change & ok]
            a, b = score[left], score[right]
            if n_pairs:
                # Searchsorted avoids allocating all pair combinations.
                sorted_b = np.sort(b)
                strict_inversions = int(np.searchsorted(sorted_b, a, side="left").sum())
                tie_count = int((np.searchsorted(sorted_b, a, side="right") - np.searchsorted(sorted_b, a, side="left")).sum())
                inversion = strict_inversions / n_pairs
            else:
                inversion, tie_count = None, 0
            local = left | right
            rows.append({"boundary": boundary, "diagnostic": name, "left_n": int(left.sum()),
                         "right_n": int(right.sum()), "pairs": n_pairs, "inversion_fraction": inversion,
                         "tied_pairs": tie_count, "local_auc": float(roc_auc_score(y[local], score[local]))
                         if local.any() and len(np.unique(y[local])) == 2 else None,
                         "scope": "cached rows only; missing outer-boundary pairs are not estimable"})
    return pd.DataFrame(rows)


def full_population_from_aggregates(frame, original_report):
    """Add unchanged out-of-cache confusion counts; never fabricate full scores."""
    baseline = original_report["full_test_partial_policy"]["xgboost"]
    base_selected = metrics(frame.y_true, frame.p_xgb >= .5, frame.p_xgb)
    outside = {k: baseline["confusion"][k] - base_selected[k] for k in ["tn", "fp", "fn", "tp"]}
    if min(outside.values()) < 0 or sum(outside.values()) != baseline["n"] - len(frame):
        raise ValueError("Published full-test aggregates are inconsistent with cached rows.")
    specs = {"XGB@0.50": {"kind": "xgb"}, "JEV replacement": {"kind": "replacement"},
             "OR": {"kind": "or"}, "AND": {"kind": "and"},
             "Mean 50:50": {"kind": "weighted", "jev_weight": .5}}
    rows = []
    for name, spec in specs.items():
        s, pred, needs = policy_output(frame, spec)
        selected = metrics(frame.y_true, pred, s)
        m = counts_metrics(**{k: outside[k] + selected[k] for k in outside})
        # Proper-score losses are additive; AUC/AP require missing rankings.
        m["brier"] = (baseline["n"] * baseline["brier_score"] + len(frame)*(selected["brier"]-base_selected["brier"])) / baseline["n"]
        m["log_loss"] = (baseline["n"] * baseline["log_loss"] + len(frame)*(selected["log_loss"]-base_selected["log_loss"])) / baseline["n"]
        m["roc_auc"], m["average_precision"] = None, None
        if spec["kind"] in ("or", "and"):
            m["brier"], m["log_loss"] = None, None
        tr = transitions(frame.y_true, frame.p_xgb.to_numpy() >= .5, pred)
        tr["accuracy_change_pp"] = 100 * tr["net_correct"] / baseline["n"]
        tr["accuracy_relative_improvement_pct"] = 100 * tr["net_correct"] / baseline["correct"]
        rows.append({"policy": name, **m, **tr, "prospective_requests": int(needs.sum()),
                     "routing_fraction": int(needs.sum()) / baseline["n"],
                     "evidence": "derived from public full-test baseline counts/losses plus exact selected-row changes",
                     "scope": "fixed historical 5,000-row manifest; every other test row unchanged"})
    return pd.DataFrame(rows)


def error_sample_manifest(frame):
    x, j = frame.p_xgb.to_numpy() >= .5, policy_output(frame, {"kind": "replacement"})[1]
    y = frame.y_true.to_numpy()
    groups = {"recovered_false_negative": (y == 1) & ~x & j,
              "introduced_false_negative": (y == 1) & x & ~j,
              "corrected_false_positive": (y == 0) & x & ~j,
              "introduced_false_positive": (y == 0) & ~x & j}
    parts = []
    for group, mask in groups.items():
        examples = frame.loc[mask, ["sample_id", "row_id", "text_sha256", "p_xgb", "jev_score", "y_true"]].copy()
        examples["review_hash"] = examples.sample_id.map(lambda s: digest([RANDOM_SEED, "error-review", s]))
        examples = examples.sort_values("review_hash").head(8)
        examples["transition"] = group
        examples["annotation_status"] = "pending: original text absent; no semantic annotation inferred from scores"
        parts.append(examples)
    return pd.concat(parts, ignore_index=True)


def audit_sources(root, frame, manifest, report):
    source = root / "research/results/jev_probability_hybrid"
    prior = pd.read_json(root / "research/results/jev_error_study/base_predictions.json", precise_float=True)
    joined = frame.merge(prior, on="text_sha256", suffixes=("_probability", "_error"))
    records = []
    for name in ["predictions.json", "routing_manifest.json", "study.json", "hybrid_report.json",
                 "execution_audit.json", "inference_budget.json", "validation_bins.json"]:
        p = source / name
        records.append({"path": str(p.relative_to(root)), "available": p.is_file(),
                        "sha256": file_digest(p), "evidence": "committed public export"})
    for relative in ["data/Suicide_Detection.csv", "artifacts/jev_probability_hybrid/cohort.csv.gz",
                     "artifacts/jev_probability_hybrid/budget.sqlite3", "artifacts/jev_study/baseline"]:
        records.append({"path": relative, "available": (root / relative).exists(),
                        "sha256": None, "evidence": "private artifact; availability checked in this checkout"})
    for relative in ["assets/notebooks/jev_probability_hybrid_5000.ipynb",
                     "assets/notebooks/federated_xgboost_distribution_iid_non_iid_executed.ipynb",
                     "jev_research/client.py", "jev_research/notebook_pipeline.py",
                     "research/results/jev_error_study/base_predictions.json",
                     "research/results/jev_error_study/selection_manifest.json"]:
        records.append({"path": relative, "available": True, "sha256": file_digest(root / relative),
                        "evidence": "committed source / earlier export"})
    audit = {"sources": records, "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dataset_sha256": frame.dataset_sha256.iloc[0],
        "baseline_fingerprint": frame.baseline_fingerprint.iloc[0],
        "selected_n": len(frame), "successes": int(frame.jev_status.eq("succeeded").sum()),
        "failures": int(frame.jev_status.eq("failed").sum()),
        "prior_error_study_overlap_n": len(joined),
        "prior_overlap_different_scores_n": int((joined.jev_score - joined.base_score).abs().gt(1e-12).sum()),
        "earlier_additional_compatible_cache_n": len(prior) - len(joined),
        "earlier_cache_handling": "Do not overwrite historical responses. Extra earlier error-selected cases are excluded from common-population comparisons; exclude them from new sampling.",
        "request_model": BASE_MODEL, "immutable_model_revision": None,
        "endpoint": ENDPOINT, "question": QUESTION, "question_sha256": digest(QUESTION),
        "score_semantics": "model-returned 0–1 suicidal-intent risk rating, not verified calibrated probability or token confidence",
        "parser": "answers.department.score; finite numeric non-boolean in [0,1]",
        "truncation": "client sends original text unchanged; no client-side truncation. Provider context handling unavailable.",
        "label_map": {"non-suicide": 0, "suicide": 1},
        "preprocessing_audit": "Source inspection: NFKC/HTML/lowercase, remove URLs/emails/mentions; expand negations, preserve words/numbers. Raw normalized duplicate/conflicting labels handled before stratified 64/16/20 split. Cleaned collisions removed with train > validation > test priority.",
        "feature_learning_audit": "Source inspection: Word2Vec vocabulary/vectors and XGBoost fit training rows only. No retraining performed. Runtime split/text checks unavailable without private tables.",
        "stronger_separation": "No author/time fields in supplied manifests; source loader retains text/class only. Author/temporal disjointness cannot be assessed; no author bootstrap is possible.",
        "dataset_recovery": "Version-14 kagglehub download attempted: api.kaggle.com proxy returned 403 Forbidden. Git tree and workspace contain no original CSV or full per-row baseline scores.",
        "historical_execution_window_utc": report["metadata"]["execution_window_utc"],
        "historical_attempts": report["metadata"]["inference_budget"]["attempts_spent"],
        "historical_phase_wall_seconds": sum((datetime.fromisoformat(p["last_finished_utc"]) - datetime.fromisoformat(p["first_started_utc"])).total_seconds()
                                               for p in json.loads((source / "execution_audit.json").read_text())["phases"]),
        "cost_and_latency": "No pricing, currency, usage, or per-request latency retained in public exports. Phase wall duration includes concurrent requests and is not mean latency.",
        "new_paid_attempts": 0}
    expected = manifest.set_index("sample_id").loc[frame.sample_id]
    for field in ["row_id", "text_sha256", "y_true", "source_split", "dataset_sha256", "baseline_fingerprint"]:
        if not np.array_equal(expected[field], frame[field]):
            raise ValueError(f"Prediction/manifest mismatch: {field}")
    np.testing.assert_allclose(expected.p_xgb, frame.p_xgb, rtol=0, atol=1e-14)
    if len(manifest) != len(frame):
        raise ValueError("Manifest population mismatch.")
    return audit


def reproduce(frame, old):
    specs = {"XGB@0.50": {"kind": "xgb"}, "JEV replacement": {"kind": "replacement"},
             "OR": {"kind": "or"}, "AND": {"kind": "and"},
             "Mean 50:50": {"kind": "weighted", "jev_weight": .5}}
    result = evaluate_policies(frame, specs, dict.fromkeys(specs, .5))
    aliases = {"precision": "precision_positive", "recall": "recall_positive", "f1": "f1_positive",
               "macro_f1": "f1_macro", "brier": "brier_score"}
    for name, key in [("XGB@0.50", "xgboost"), ("JEV replacement", "hybrid")]:
        actual = result.loc[result.policy.eq(name)].iloc[0]
        saved = old["selected_subset"][key]
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "roc_auc", "average_precision", "brier", "log_loss"]:
            np.testing.assert_allclose(actual[metric], saved[aliases.get(metric, metric)], rtol=1e-11, atol=1e-12)
        for col in ["tn", "fp", "fn", "tp"]:
            assert actual[col] == saved["confusion"][col]
    replacement = result.loc[result.policy.eq("JEV replacement")].iloc[0]
    for metric, old_metric in [("corrected", "corrected_xgboost_errors"), ("introduced", "introduced_errors"),
                              ("fn_recovered", "corrected_false_negatives"), ("fn_introduced", "introduced_false_negatives"),
                              ("fp_corrected", "corrected_false_positives"), ("fp_introduced", "introduced_false_positives")]:
        assert replacement[metric] == old["selected_subset"]["paired"][old_metric]
    return result


def run_cached_study(root, results=None):
    root = Path(root).resolve()
    output = Path(results) if results else root / "research/results" / STUDY_NAME
    output.mkdir(parents=True, exist_ok=True)
    source = root / "research/results/jev_probability_hybrid"
    frame = pd.read_json(source / "predictions.json", precise_float=True)
    manifest = pd.read_json(source / "routing_manifest.json", precise_float=True)
    original = json.loads((source / "hybrid_report.json").read_text())
    validate_cache(frame)
    audit = audit_sources(root, frame, manifest, original)
    save_json(output / "artifact_audit.json", audit)
    all_rows, development, evaluation = split_cache(frame)
    save_table(output, "cache_split_manifest", all_rows[["sample_id", "row_id", "text_sha256", "partition_hash", "partition"]])
    specs, thresholds, selection = fit_policies(development)
    # Freeze before any evaluation metric, reliability diagram or bin analysis.
    matched_choices = {f"{name}:{target}": choose_threshold(development, spec, target)
                       for target in [.9, .95] for name, spec in specs.items()
                       if not (spec["kind"] == "xgb" and name != "XGB@0.50")}
    routing_dev, routing_frozen = select_routing(development)
    frozen = {"seed": RANDOM_SEED, "development_n": len(development), "evaluation_n": len(evaluation),
              "development_ids_sha256": digest(sorted(development.sample_id)),
              "evaluation_ids_sha256": digest(sorted(evaluation.sample_id)),
              "specifications": specs, "thresholds": thresholds, "selection": selection,
              "matched_recall_choices": matched_choices, "routing_cost_choices": routing_frozen,
              "interpretation": "Within-study hash split of previously inspected test texts; exploratory, not independent confirmation."}
    save_json(output / "frozen_policies.json", frozen)
    routing_dev, routing_eval, routing_selected, _ = routing_analysis(
        development, evaluation, selection=(routing_dev, routing_frozen))
    tables = {"historical_selected_comparison": reproduce(frame, original),
              "full_population_derived": full_population_from_aggregates(frame, original),
              "development_comparison": evaluate_policies(development, specs, thresholds),
              "evaluation_comparison": evaluate_policies(evaluation, specs, thresholds),
              "matched_recall_evaluation": matched_recall(development, evaluation, specs, matched_choices)[0],
              "matched_recall_uncertainty": matched_uncertainty(development, evaluation, specs, matched_choices),
              "routing_development": routing_dev, "routing_evaluation": routing_eval,
              "routing_selected_evaluation": routing_selected,
              "score_bins_descriptive": descriptive_bins(all_rows),
              "paired_uncertainty": uncertainty(evaluation, specs, thresholds),
              "reliability": reliability_table(evaluation, specs),
              "score_loss_uncertainty": score_loss_uncertainty(evaluation, specs),
              "boundary_ranking": boundary_ranking(evaluation),
              "error_review_manifest": error_sample_manifest(frame)}
    predictions = all_rows[["sample_id", "row_id", "text_sha256", "partition", "y_true", "p_xgb", "jev_score", "jev_status"]].copy()
    for name, spec in specs.items():
        s, pred, needs = policy_output(all_rows, spec, thresholds[name])
        predictions[name + " / score"], predictions[name + " / positive"] = s, pred.astype(int)
        predictions[name + " / requires_jev"] = needs
    tables["policy_predictions"] = predictions
    for name, table in tables.items():
        save_table(output, name, table)
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "nbclient", "xgboost-cpu", "gensim"]:
        versions[package] = importlib.metadata.version(package)
    save_json(output / "reproducibility.json", {
        "source_revision": audit["source_revision"], "seed": RANDOM_SEED, "dependencies": versions,
        "analysis_executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code_sha256": file_digest(Path(__file__)),
        "additional_inference_code_sha256": file_digest(Path(__file__).with_name("additional_experiment.py")),
        "source_fingerprints": audit["sources"], "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "paired individual text; author IDs unavailable",
        "new_paid_attempts": 0, "max_new_jev_attempts": MAX_NEW_JEV_ATTEMPTS,
        "run_new_jev_inference": RUN_NEW_JEV_INFERENCE,
        "scores": "raw risk ratings and bounded fused scores; no population calibration claim",
        "frozen_policy_sha256": digest(frozen)})
    return {"audit": audit, "frozen": frozen, "tables": tables,
            "frame": all_rows, "development": development, "evaluation": evaluation, "results": output}
