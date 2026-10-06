from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from .clustering import cluster_observations
from .dependency import normalize_domain, source_category, token_jaccard
from .normalization import (
    is_no_answer,
    normalize_answer,
    qacc_exact_match,
    qacc_token_f1,
    safe_surface_equivalent,
)
from .qacc import evaluation_references, qacc_conflict_label
from .schemas import EvidenceObservation, PreparedExample


UQ_ROUTING_PROTOCOL_VERSION = "1.1"
UNKNOWN = "unknown"

UQ_FEATURE_NAMES = (
    "log_context_count",
    "log_extracted_cluster_count",
    "evidence_semantic_entropy",
    "evidence_vote_uncertainty",
    "numeric_answer_conflict",
    "model_disagreement",
    "luna_abstained",
    "mini_abstained",
    "luna_log_support",
    "luna_log_independent_domains",
    "luna_mean_entailment",
    "luna_mean_directness",
    "luna_mean_answer_type_match",
    "luna_mean_temporal_fit",
    "luna_exact_mention_rate",
    "luna_unsupported_token_rate",
    "mini_minus_luna_log_support",
    "mini_minus_luna_entailment",
    "log_domain_count",
    "duplicate_source_rate",
    "official_domain_share",
    "reference_domain_share",
    "social_domain_share",
    "context_year_dispersion",
    "question_has_temporal_cue",
    "luna_log_answer_length",
)

# Stage one must be computable before the secondary model is called.  Features
# derived from Mini are deliberately excluded from the pre-routing risk model.
PRE_ROUTING_FEATURE_NAMES = tuple(
    name for name in UQ_FEATURE_NAMES
    if name not in {
        "model_disagreement",
        "mini_abstained",
        "mini_minus_luna_log_support",
        "mini_minus_luna_entailment",
    }
)
def _pre_routing_indices(mode: str) -> np.ndarray:
    excluded: set[str]
    if mode == "full":
        excluded = set()
    elif mode == "no-source-time":
        excluded = {
            "luna_log_independent_domains",
            "luna_mean_temporal_fit",
            "log_domain_count",
            "duplicate_source_rate",
            "official_domain_share",
            "reference_domain_share",
            "social_domain_share",
            "context_year_dispersion",
            "question_has_temporal_cue",
        }
    elif mode == "answer-only":
        keep = {
            "log_context_count",
            "luna_abstained",
            "luna_exact_mention_rate",
            "luna_unsupported_token_rate",
            "luna_log_answer_length",
        }
        excluded = set(PRE_ROUTING_FEATURE_NAMES) - keep
    else:
        raise ValueError("pre_feature_mode must be full, no-source-time, or answer-only")
    names = [name for name in PRE_ROUTING_FEATURE_NAMES if name not in excluded]
    return np.asarray([UQ_FEATURE_NAMES.index(name) for name in names], dtype=int)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
_TEMPORAL_CUE = re.compile(
    r"\b(?:when|year|date|month|day|current|currently|latest|recent|first|last|founded|born|died)\b",
    re.IGNORECASE,
)


def _hash_fraction(value: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _answer(record: Mapping | None) -> str:
    value = str((record or {}).get("answer", UNKNOWN)).strip()
    return value or UNKNOWN


def _support_summary(
    example: PreparedExample,
    answer: str,
    observations: Sequence[EvidenceObservation],
) -> dict[str, float]:
    matched = [
        item for item in observations
        if safe_surface_equivalent(answer, item.candidate_answer)
    ]
    support_indices = {item.context_index for item in matched}
    context_by_index = {item.index: item for item in example.contexts}
    domains = {
        normalize_domain(context_by_index[index].source)
        for index in support_indices
        if index in context_by_index
    }
    normalized = normalize_answer(answer)
    tokens = normalized.split()
    context_keys = [normalize_answer(item.text) for item in example.contexts]
    context_tokens = set(" ".join(context_keys).split())

    def mean(name: str, default: float = 0.5) -> float:
        values = [float(getattr(item, name)) for item in matched]
        return float(np.mean(values)) if values else default

    return {
        "log_support": math.log1p(len(support_indices)),
        "log_domains": math.log1p(len(domains)),
        "entailment": mean("entailment"),
        "directness": mean("directness"),
        "answer_type": mean("answer_type_match"),
        "temporal_fit": mean("temporal_fit"),
        "mention_rate": sum(bool(normalized and normalized in key) for key in context_keys)
        / max(len(context_keys), 1),
        "unsupported_rate": (
            sum(token not in context_tokens for token in tokens) / len(tokens)
            if tokens else 1.0
        ),
        "log_length": math.log1p(len(tokens)),
    }


def build_uq_features(
    example: PreparedExample,
    luna_record: Mapping | None,
    mini_record: Mapping | None,
    observations: Sequence[EvidenceObservation] = (),
) -> tuple[np.ndarray, dict]:
    luna = _answer(luna_record)
    mini = _answer(mini_record)
    clusters = cluster_observations(observations)
    counts = np.asarray(
        [len({item.context_index for item in cluster.observations}) for cluster in clusters],
        dtype=float,
    )
    if len(counts) > 1 and float(counts.sum()) > 0:
        probabilities = counts / counts.sum()
        semantic_entropy = -float(
            np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
        ) / math.log(len(probabilities))
        ordered = np.sort(probabilities)[::-1]
        vote_uncertainty = 1.0 - float(ordered[0] - ordered[1])
    else:
        semantic_entropy = 0.0
        vote_uncertainty = 0.0
    cluster_answers = [normalize_answer(cluster.answer) for cluster in clusters]
    numeric_conflict = any(
        bool(_NUMBER.search(left) and _NUMBER.search(right))
        and set(_NUMBER.findall(left)) != set(_NUMBER.findall(right))
        for index, left in enumerate(cluster_answers)
        for right in cluster_answers[index + 1 :]
    )

    luna_support = _support_summary(example, luna, observations)
    mini_support = _support_summary(example, mini, observations)
    domains = [normalize_domain(item.source) for item in example.contexts]
    unique_domains = set(domains)
    categories = Counter(source_category(domain) for domain in unique_domains)
    category_total = max(len(unique_domains), 1)
    duplicate_pairs = 0
    total_pairs = 0
    for index, left in enumerate(example.contexts):
        for right in example.contexts[index + 1 :]:
            total_pairs += 1
            if (
                normalize_domain(left.source) == normalize_domain(right.source)
                or token_jaccard(left.text, right.text) >= 0.88
            ):
                duplicate_pairs += 1
    years = [
        int(value)
        for context in example.contexts
        for value in _YEAR.findall(context.text)
    ]
    year_dispersion = (
        min(1.0, math.log1p(max(years) - min(years)) / math.log1p(100.0))
        if len(years) >= 2 else 0.0
    )
    disagreement = not safe_surface_equivalent(luna, mini)

    values = np.asarray([
        math.log1p(len(example.contexts)),
        math.log1p(len(clusters)),
        semantic_entropy,
        vote_uncertainty,
        float(numeric_conflict),
        float(disagreement),
        float(is_no_answer(luna)),
        float(is_no_answer(mini)),
        luna_support["log_support"],
        luna_support["log_domains"],
        luna_support["entailment"],
        luna_support["directness"],
        luna_support["answer_type"],
        luna_support["temporal_fit"],
        luna_support["mention_rate"],
        luna_support["unsupported_rate"],
        mini_support["log_support"] - luna_support["log_support"],
        mini_support["entailment"] - luna_support["entailment"],
        math.log1p(len(unique_domains)),
        duplicate_pairs / total_pairs if total_pairs else 0.0,
        categories["official"] / category_total,
        categories["reference"] / category_total,
        categories["social"] / category_total,
        year_dispersion,
        float(bool(_TEMPORAL_CUE.search(example.question))),
        luna_support["log_length"],
    ], dtype=float)
    return values, {
        "luna_answer": luna,
        "mini_answer": mini,
        "model_disagreement": disagreement,
        "evidence_semantic_entropy": semantic_entropy,
        "evidence_vote_uncertainty": vote_uncertainty,
        "extracted_cluster_count": len(clusters),
        "year_count": len(years),
        "independent_domain_count": len(unique_domains),
    }


@dataclass
class _RidgeLogit:
    prior_scale: float = 0.75

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "_RidgeLogit":
        matrix = np.asarray(matrix, dtype=float)
        labels = np.asarray(labels, dtype=float)
        self.mean_ = matrix.mean(axis=0)
        raw_scale = matrix.std(axis=0)
        self.scale_ = np.where(raw_scale > 1e-8, raw_scale, 1.0)
        transformed = (matrix - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(transformed)), transformed])
        precision = 1.0 / (self.prior_scale**2)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            scores = design @ beta
            loss = float(np.sum(np.logaddexp(0.0, scores) - labels * scores))
            loss += 0.5 * precision * float(beta[1:] @ beta[1:])
            gradient = design.T @ (expit(scores) - labels)
            gradient[1:] += precision * beta[1:]
            return loss, gradient

        base_rate = float(np.clip(labels.mean(), 1e-4, 1.0 - 1e-4))
        initial = np.zeros(design.shape[1], dtype=float)
        initial[0] = math.log(base_rate / (1.0 - base_rate))
        result = minimize(objective, initial, jac=True, method="L-BFGS-B")
        self.coefficients_ = np.asarray(result.x, dtype=float)
        self.temperature_ = 1.0
        self.converged_ = bool(result.success)
        return self

    def logits(self, matrix: np.ndarray) -> np.ndarray:
        transformed = (np.asarray(matrix, dtype=float) - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(transformed)), transformed])
        return design @ self.coefficients_

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return expit(self.logits(matrix) / self.temperature_)

    def calibrate_temperature(self, matrix: np.ndarray, labels: np.ndarray) -> float:
        if len(matrix) < 2:
            return self.temperature_
        logits = self.logits(matrix)
        labels = np.asarray(labels, dtype=float)

        def nll(log_temperature: float) -> float:
            scores = logits / math.exp(log_temperature)
            return float(np.mean(np.logaddexp(0.0, scores) - labels * scores))

        result = minimize_scalar(
            nll, bounds=(math.log(0.25), math.log(4.0)), method="bounded"
        )
        self.temperature_ = math.exp(float(result.x))
        return self.temperature_


def _is_correct(answer: str, references: Sequence[str]) -> bool:
    return not is_no_answer(answer) and any(
        qacc_exact_match(answer, value) for value in references
    )


def _answer_metrics(rows: Sequence[Mapping], answer_key: str) -> dict:
    exact_values: list[float] = []
    f1_values: list[float] = []
    covered: list[float] = []
    for row in rows:
        answer = str(row[answer_key])
        references = list(row["references"])
        exact = float(_is_correct(answer, references))
        exact_values.append(exact)
        if is_no_answer(answer):
            f1_values.append(0.0)
        else:
            f1_values.append(
                max((qacc_token_f1(answer, item) for item in references), default=0.0)
            )
            covered.append(exact)
    return {
        "examples": len(rows),
        "answer_exact": float(np.mean(exact_values)) if rows else 0.0,
        "answer_token_f1": float(np.mean(f1_values)) if rows else 0.0,
        "coverage": len(covered) / len(rows) if rows else 0.0,
        "selective_exact": float(np.mean(covered)) if covered else 0.0,
    }


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels.astype(bool)].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def _risk_quality(
    rows: Sequence[Mapping],
    *,
    correct_key: str = "luna_correct",
    score_key: str = "luna_error_probability",
) -> dict:
    labels = np.asarray([not row[correct_key] for row in rows], dtype=float)
    scores = np.asarray([row[score_key] for row in rows], dtype=float)
    order = np.argsort(scores, kind="stable")
    ordered_errors = labels[order]
    cumulative_risk = np.cumsum(ordered_errors) / np.arange(1, len(rows) + 1)
    bins = []
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        observed = float(labels[mask].mean())
        weight = float(mask.mean())
        calibration_error += weight * abs(confidence - observed)
        bins.append({
            "lower": float(lower), "upper": float(upper), "examples": int(mask.sum()),
            "mean_predicted_error": confidence, "observed_error": observed,
        })
    coverage = {}
    for target in (0.5, 0.7, 0.8, 0.9, 1.0):
        count = max(1, math.ceil(target * len(rows)))
        accepted = ordered_errors[:count]
        coverage[f"{target:.1f}"] = {
            "accepted": count,
            "actual_coverage": count / len(rows),
            "selective_exact": 1.0 - float(accepted.mean()),
            "risk": float(accepted.mean()),
        }
    return {
        "error_prevalence": float(labels.mean()),
        "auroc_error_detection": _binary_auc(labels, scores),
        "average_precision_error_detection": _average_precision(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2)),
        "ece_10_bins": calibration_error,
        "aurc": float(np.mean(cumulative_risk)),
        "selective_by_coverage": coverage,
        "calibration_bins": bins,
    }


def _paired_exact_p(helped: int, hurt: int) -> float:
    discordant = helped + hurt
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(helped, hurt) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _bootstrap_delta(
    rows: Sequence[Mapping],
    candidate_key: str,
    *,
    seed: int,
    baseline_correct_key: str = "luna_correct",
    samples: int = 5000,
) -> list[float]:
    differences = np.asarray([
        float(_is_correct(str(row[candidate_key]), row["references"]))
        - float(row[baseline_correct_key])
        for row in rows
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(samples, len(rows)))
    means = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def run_uncertainty_routing_cross_validation(
    examples: Sequence[PreparedExample],
    luna_predictions: Mapping[str, Mapping],
    mini_predictions: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]] | None = None,
    *,
    folds: int = 5,
    calibration_fraction: float = 0.25,
    prior_scale: float = 0.75,
    intervention_quantile: float = 0.75,
    delta_margin: float = 0.10,
    selective_coverage: float = 0.80,
    pre_feature_mode: str = "full",
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5)")
    if not 0.0 < intervention_quantile < 1.0:
        raise ValueError("intervention_quantile must be in (0, 1)")
    if not 0.0 < selective_coverage <= 1.0:
        raise ValueError("selective_coverage must be in (0, 1]")
    pre_routing_indices = _pre_routing_indices(pre_feature_mode)

    extraction_cache = extractions or {}
    prepared = []
    order = {}
    for index, example in enumerate(examples):
        features, metadata = build_uq_features(
            example,
            luna_predictions.get(example.question_id),
            mini_predictions.get(example.question_id),
            extraction_cache.get(example.question_id, ()),
        )
        references = evaluation_references(example)
        luna_correct = _is_correct(metadata["luna_answer"], references)
        mini_correct = _is_correct(metadata["mini_answer"], references)
        prepared.append({
            "example": example,
            "features": features,
            "metadata": metadata,
            "references": references,
            "luna_error": float(not luna_correct),
            "mini_error": float(not mini_correct),
        })
        order[example.question_id] = index

    assignments = {
        item["example"].question_id:
        int(_hash_fraction(item["example"].question_id, f"uq-fold-{seed}") * folds)
        for item in prepared
    }
    rows = []
    fold_reports = []
    for fold in range(folds):
        held_out = [item for item in prepared if assignments[item["example"].question_id] == fold]
        remaining = [item for item in prepared if assignments[item["example"].question_id] != fold]
        calibration = [
            item for item in remaining
            if _hash_fraction(item["example"].question_id, f"uq-cal-{seed}-{fold}")
            < calibration_fraction
        ]
        calibration_ids = {item["example"].question_id for item in calibration}
        fitting = [
            item for item in remaining
            if item["example"].question_id not in calibration_ids
        ]
        if len(fitting) < 20 or len(calibration) < 5:
            raise ValueError(f"Fold {fold} has too few fit/calibration examples")
        fit_matrix = np.vstack([item["features"] for item in fitting])
        calibration_matrix = np.vstack([item["features"] for item in calibration])
        held_matrix = np.vstack([item["features"] for item in held_out])
        pre_fit_matrix = fit_matrix[:, pre_routing_indices]
        pre_calibration_matrix = calibration_matrix[:, pre_routing_indices]
        pre_held_matrix = held_matrix[:, pre_routing_indices]
        pre_luna_model = _RidgeLogit(prior_scale).fit(
            pre_fit_matrix, np.asarray([item["luna_error"] for item in fitting])
        )
        full_luna_model = _RidgeLogit(prior_scale).fit(
            fit_matrix, np.asarray([item["luna_error"] for item in fitting])
        )
        mini_model = _RidgeLogit(prior_scale).fit(
            fit_matrix, np.asarray([item["mini_error"] for item in fitting])
        )
        pre_luna_temperature = pre_luna_model.calibrate_temperature(
            pre_calibration_matrix,
            np.asarray([item["luna_error"] for item in calibration]),
        )
        full_luna_temperature = full_luna_model.calibrate_temperature(
            calibration_matrix,
            np.asarray([item["luna_error"] for item in calibration]),
        )
        mini_temperature = mini_model.calibrate_temperature(
            calibration_matrix,
            np.asarray([item["mini_error"] for item in calibration]),
        )
        calibration_luna_risk = pre_luna_model.predict(pre_calibration_matrix)
        intervention_threshold = float(np.quantile(
            calibration_luna_risk, intervention_quantile, method="higher"
        ))
        selective_threshold = float(np.quantile(
            calibration_luna_risk, selective_coverage, method="higher"
        ))
        held_luna_risk = pre_luna_model.predict(pre_held_matrix)
        held_full_luna_risk = full_luna_model.predict(held_matrix)
        held_mini_risk = mini_model.predict(held_matrix)
        fold_reports.append({
            "fold": fold,
            "fit_examples": len(fitting),
            "calibration_examples": len(calibration),
            "held_out_examples": len(held_out),
            "pre_luna_temperature": pre_luna_temperature,
            "full_luna_temperature": full_luna_temperature,
            "mini_temperature": mini_temperature,
            "intervention_risk_threshold": intervention_threshold,
            "selective_acceptance_threshold": selective_threshold,
        })

        for position, item in enumerate(held_out):
            example = item["example"]
            metadata = item["metadata"]
            luna = metadata["luna_answer"]
            mini = metadata["mini_answer"]
            luna_risk = float(held_luna_risk[position])
            mini_risk = float(held_mini_risk[position])
            full_luna_risk = float(held_full_luna_risk[position])
            expected_gain = full_luna_risk - mini_risk
            eligible = bool(metadata["model_disagreement"] and not is_no_answer(mini))
            high_risk = luna_risk >= intervention_threshold
            risk_only_switch = bool(eligible and high_risk)
            conservative_switch = bool(risk_only_switch and expected_gain >= delta_margin)
            accepted = luna_risk <= selective_threshold
            rows.append({
                "question_id": example.question_id,
                "fold": fold,
                "references": item["references"],
                "gold_conflict": qacc_conflict_label(example),
                "luna_answer": luna,
                "mini_answer": mini,
                "luna_correct": not bool(item["luna_error"]),
                "mini_correct": not bool(item["mini_error"]),
                "luna_error_probability": luna_risk,
                "post_secondary_luna_error_probability": full_luna_risk,
                "mini_error_probability": mini_risk,
                "expected_mini_gain": expected_gain,
                "model_disagreement": metadata["model_disagreement"],
                "evidence_semantic_entropy": metadata["evidence_semantic_entropy"],
                "evidence_vote_uncertainty": metadata["evidence_vote_uncertainty"],
                "extracted_cluster_count": metadata["extracted_cluster_count"],
                "independent_domain_count": metadata["independent_domain_count"],
                "intervention_risk_threshold": intervention_threshold,
                "selective_acceptance_threshold": selective_threshold,
                "secondary_triggered": high_risk,
                "risk_only_answer": mini if risk_only_switch else luna,
                "routed_answer": mini if conservative_switch else luna,
                "selective_answer": luna if accepted else UNKNOWN,
                "routing_decision": "switch_to_mini" if conservative_switch else "keep_luna",
                "risk_only_decision": "switch_to_mini" if risk_only_switch else "keep_luna",
                "selective_decision": "accept" if accepted else "abstain",
            })

    rows.sort(key=lambda item: order[item["question_id"]])

    def routing_report(answer_key: str) -> dict:
        changed = [row for row in rows if not safe_surface_equivalent(
            str(row[answer_key]), str(row["luna_answer"])
        )]
        helped = sum(
            _is_correct(str(row[answer_key]), row["references"]) and not row["luna_correct"]
            for row in changed
        )
        hurt = sum(
            not _is_correct(str(row[answer_key]), row["references"]) and row["luna_correct"]
            for row in changed
        )
        base_em = _answer_metrics(rows, "luna_answer")["answer_exact"]
        metrics = _answer_metrics(rows, answer_key)
        return {
            **metrics,
            "switches": len(changed),
            "helped": helped,
            "hurt": hurt,
            "net_exact_changes": helped - hurt,
            "em_delta_vs_luna": metrics["answer_exact"] - base_em,
            "mcnemar_exact_p_two_sided": _paired_exact_p(helped, hurt),
            "bootstrap_95pct_em_delta": _bootstrap_delta(
                rows, answer_key, seed=seed + (17 if answer_key == "routed_answer" else 29)
            ),
        }

    conflict = [row for row in rows if row["gold_conflict"]]
    nonconflict = [row for row in rows if not row["gold_conflict"]]
    summary = {
        "protocol": "labelled_dev_out_of_fold_uq_pilot_not_frozen_test",
        "protocol_version": UQ_ROUTING_PROTOCOL_VERSION,
        "examples": len(rows),
        "configuration": {
            "folds": folds,
            "calibration_fraction": calibration_fraction,
            "prior_scale": prior_scale,
            "intervention_quantile": intervention_quantile,
            "delta_margin": delta_margin,
            "selective_coverage": selective_coverage,
            "pre_feature_mode": pre_feature_mode,
            "pre_feature_count": len(pre_routing_indices),
            "seed": seed,
        },
        "risk_quality": _risk_quality(rows),
        "usage": {
            "secondary_calls": sum(row["secondary_triggered"] for row in rows),
            "secondary_call_rate": sum(row["secondary_triggered"] for row in rows) / len(rows),
            "mean_llm_calls_per_question": 1.0 + sum(
                row["secondary_triggered"] for row in rows
            ) / len(rows),
        },
        "overall": {
            "luna_direct_rag": _answer_metrics(rows, "luna_answer"),
            "mini_direct_rag": _answer_metrics(rows, "mini_answer"),
            "risk_only_route": routing_report("risk_only_answer"),
            "conservative_uq_route": routing_report("routed_answer"),
            "selective_luna": _answer_metrics(rows, "selective_answer"),
        },
        "conflict_only": {
            key: _answer_metrics(conflict, answer_key)
            for key, answer_key in (
                ("luna_direct_rag", "luna_answer"),
                ("mini_direct_rag", "mini_answer"),
                ("conservative_uq_route", "routed_answer"),
            )
        },
        "nonconflict_only": {
            key: _answer_metrics(nonconflict, answer_key)
            for key, answer_key in (
                ("luna_direct_rag", "luna_answer"),
                ("mini_direct_rag", "mini_answer"),
                ("conservative_uq_route", "routed_answer"),
            )
        },
        "folds": fold_reports,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _serialize_logit(model: _RidgeLogit) -> dict:
    return {
        "prior_scale": model.prior_scale,
        "feature_mean": model.mean_.tolist(),
        "feature_scale": model.scale_.tolist(),
        "coefficients": model.coefficients_.tolist(),
        "temperature": model.temperature_,
        "converged": model.converged_,
    }


def _deserialize_logit(payload: Mapping) -> _RidgeLogit:
    model = _RidgeLogit(float(payload["prior_scale"]))
    model.mean_ = np.asarray(payload["feature_mean"], dtype=float)
    model.scale_ = np.asarray(payload["feature_scale"], dtype=float)
    model.coefficients_ = np.asarray(payload["coefficients"], dtype=float)
    model.temperature_ = float(payload["temperature"])
    model.converged_ = bool(payload.get("converged", True))
    return model


def run_uncertainty_routing_train_dev(
    train_examples: Sequence[PreparedExample],
    dev_examples: Sequence[PreparedExample],
    train_primary_predictions: Mapping[str, Mapping],
    train_secondary_predictions: Mapping[str, Mapping],
    dev_primary_predictions: Mapping[str, Mapping],
    dev_secondary_predictions: Mapping[str, Mapping],
    train_extractions: Mapping[str, Sequence[EvidenceObservation]] | None = None,
    dev_extractions: Mapping[str, Sequence[EvidenceObservation]] | None = None,
    *,
    primary_name: str,
    secondary_name: str,
    prior_scale: float = 0.75,
    intervention_quantile: float = 0.75,
    delta_margin: float = 0.10,
    selective_coverage: float = 0.80,
    pre_feature_mode: str = "answer-only",
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    """Fit only on train and use dev for calibration and method selection.

    The function is deliberately symmetric: the first prediction cache is the
    frozen primary model and the second is called only by the simulated
    high-risk route.  Dev metrics are tuning diagnostics, never test results.
    """
    if not train_examples or not dev_examples:
        raise ValueError("train and dev examples must both be non-empty")
    if any(item.split == "test" for item in (*train_examples, *dev_examples)):
        raise ValueError("train-dev routing must not receive test inputs")
    if not 0.0 < intervention_quantile < 1.0:
        raise ValueError("intervention_quantile must be in (0, 1)")
    if not 0.0 < selective_coverage <= 1.0:
        raise ValueError("selective_coverage must be in (0, 1]")
    pre_indices = _pre_routing_indices(pre_feature_mode)

    def prepare(
        examples: Sequence[PreparedExample],
        primary_predictions: Mapping[str, Mapping],
        secondary_predictions: Mapping[str, Mapping],
        extractions: Mapping[str, Sequence[EvidenceObservation]],
    ) -> list[dict]:
        prepared = []
        for example in examples:
            # build_uq_features is position-based despite its historical names:
            # the first answer is primary and the second is secondary.
            features, metadata = build_uq_features(
                example,
                primary_predictions.get(example.question_id),
                secondary_predictions.get(example.question_id),
                extractions.get(example.question_id, ()),
            )
            references = evaluation_references(example)
            primary_answer = metadata["luna_answer"]
            secondary_answer = metadata["mini_answer"]
            prepared.append({
                "example": example,
                "features": features,
                "metadata": metadata,
                "references": references,
                "primary_answer": primary_answer,
                "secondary_answer": secondary_answer,
                "primary_error": float(not _is_correct(primary_answer, references)),
                "secondary_error": float(not _is_correct(secondary_answer, references)),
            })
        return prepared

    train = prepare(
        train_examples,
        train_primary_predictions,
        train_secondary_predictions,
        train_extractions or {},
    )
    dev = prepare(
        dev_examples,
        dev_primary_predictions,
        dev_secondary_predictions,
        dev_extractions or {},
    )
    train_matrix = np.vstack([item["features"] for item in train])
    dev_matrix = np.vstack([item["features"] for item in dev])
    train_primary_error = np.asarray([item["primary_error"] for item in train])
    train_secondary_error = np.asarray([item["secondary_error"] for item in train])
    dev_primary_error = np.asarray([item["primary_error"] for item in dev])
    dev_secondary_error = np.asarray([item["secondary_error"] for item in dev])

    pre_primary = _RidgeLogit(prior_scale).fit(
        train_matrix[:, pre_indices], train_primary_error
    )
    full_primary = _RidgeLogit(prior_scale).fit(train_matrix, train_primary_error)
    full_secondary = _RidgeLogit(prior_scale).fit(train_matrix, train_secondary_error)
    pre_primary.calibrate_temperature(dev_matrix[:, pre_indices], dev_primary_error)
    full_primary.calibrate_temperature(dev_matrix, dev_primary_error)
    full_secondary.calibrate_temperature(dev_matrix, dev_secondary_error)

    pre_risk = pre_primary.predict(dev_matrix[:, pre_indices])
    post_primary_risk = full_primary.predict(dev_matrix)
    secondary_risk = full_secondary.predict(dev_matrix)
    intervention_threshold = float(np.quantile(
        pre_risk, intervention_quantile, method="higher"
    ))
    selective_threshold = float(np.quantile(
        pre_risk, selective_coverage, method="higher"
    ))

    rows = []
    for index, item in enumerate(dev):
        metadata = item["metadata"]
        primary_answer = item["primary_answer"]
        secondary_answer = item["secondary_answer"]
        expected_gain = float(post_primary_risk[index] - secondary_risk[index])
        triggered = bool(pre_risk[index] >= intervention_threshold)
        disagrees = bool(metadata["model_disagreement"])
        eligible = bool(triggered and disagrees and not is_no_answer(secondary_answer))
        risk_only_switch = eligible
        conservative_switch = bool(eligible and expected_gain >= delta_margin)
        accepted = bool(pre_risk[index] <= selective_threshold)
        rows.append({
            "question_id": item["example"].question_id,
            "references": item["references"],
            "gold_conflict": qacc_conflict_label(item["example"]),
            "primary_name": primary_name,
            "secondary_name": secondary_name,
            "primary_answer": primary_answer,
            "secondary_answer": secondary_answer,
            "primary_correct": not bool(item["primary_error"]),
            "secondary_correct": not bool(item["secondary_error"]),
            "primary_error_probability": float(pre_risk[index]),
            "post_secondary_primary_error_probability": float(post_primary_risk[index]),
            "secondary_error_probability": float(secondary_risk[index]),
            "expected_secondary_gain": expected_gain,
            "model_disagreement": disagrees,
            "evidence_semantic_entropy": metadata["evidence_semantic_entropy"],
            "extracted_cluster_count": metadata["extracted_cluster_count"],
            "independent_domain_count": metadata["independent_domain_count"],
            "secondary_triggered": triggered,
            "risk_only_answer": secondary_answer if risk_only_switch else primary_answer,
            "routed_answer": secondary_answer if conservative_switch else primary_answer,
            "selective_answer": primary_answer if accepted else UNKNOWN,
            "routing_decision": "switch_to_secondary" if conservative_switch else "keep_primary",
            "risk_only_decision": "switch_to_secondary" if risk_only_switch else "keep_primary",
            "selective_decision": "accept" if accepted else "abstain",
        })

    def routing_report(answer_key: str) -> dict:
        changed = [row for row in rows if not safe_surface_equivalent(
            str(row[answer_key]), str(row["primary_answer"])
        )]
        helped = sum(
            _is_correct(str(row[answer_key]), row["references"])
            and not row["primary_correct"]
            for row in changed
        )
        hurt = sum(
            not _is_correct(str(row[answer_key]), row["references"])
            and row["primary_correct"]
            for row in changed
        )
        baseline = _answer_metrics(rows, "primary_answer")["answer_exact"]
        metrics = _answer_metrics(rows, answer_key)
        return {
            **metrics,
            "switches": len(changed),
            "helped": helped,
            "hurt": hurt,
            "net_exact_changes": helped - hurt,
            "em_delta_vs_primary": metrics["answer_exact"] - baseline,
            "mcnemar_exact_p_two_sided": _paired_exact_p(helped, hurt),
            "bootstrap_95pct_em_delta": _bootstrap_delta(
                rows,
                answer_key,
                seed=seed + (17 if answer_key == "routed_answer" else 29),
                baseline_correct_key="primary_correct",
            ),
        }

    conflict = [row for row in rows if row["gold_conflict"]]
    nonconflict = [row for row in rows if not row["gold_conflict"]]
    secondary_calls = sum(row["secondary_triggered"] for row in rows)
    summary = {
        "protocol": "train_fit_dev_calibration_and_selection_not_test",
        "protocol_version": UQ_ROUTING_PROTOCOL_VERSION,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "configuration": {
            "primary_name": primary_name,
            "secondary_name": secondary_name,
            "prior_scale": prior_scale,
            "intervention_quantile": intervention_quantile,
            "delta_margin": delta_margin,
            "selective_coverage": selective_coverage,
            "pre_feature_mode": pre_feature_mode,
            "pre_feature_count": len(pre_indices),
            "seed": seed,
        },
        "thresholds": {
            "intervention_risk": intervention_threshold,
            "selective_acceptance": selective_threshold,
        },
        "risk_quality": _risk_quality(
            rows,
            correct_key="primary_correct",
            score_key="primary_error_probability",
        ),
        "usage": {
            "secondary_calls": secondary_calls,
            "secondary_call_rate": secondary_calls / len(rows),
            "mean_llm_calls_per_question": 1.0 + secondary_calls / len(rows),
        },
        "overall": {
            "primary_direct_rag": _answer_metrics(rows, "primary_answer"),
            "secondary_direct_rag": _answer_metrics(rows, "secondary_answer"),
            "risk_only_route": routing_report("risk_only_answer"),
            "conservative_uq_route": routing_report("routed_answer"),
            "selective_primary": _answer_metrics(rows, "selective_answer"),
        },
        "conflict_only": {
            key: _answer_metrics(conflict, answer_key)
            for key, answer_key in (
                ("primary_direct_rag", "primary_answer"),
                ("secondary_direct_rag", "secondary_answer"),
                ("conservative_uq_route", "routed_answer"),
            )
        },
        "nonconflict_only": {
            key: _answer_metrics(nonconflict, answer_key)
            for key, answer_key in (
                ("primary_direct_rag", "primary_answer"),
                ("secondary_direct_rag", "secondary_answer"),
                ("conservative_uq_route", "routed_answer"),
            )
        },
    }
    model_artifact = {
        "protocol_version": UQ_ROUTING_PROTOCOL_VERSION,
        "configuration": summary["configuration"],
        "thresholds": summary["thresholds"],
        "feature_names": list(UQ_FEATURE_NAMES),
        "pre_feature_names": [UQ_FEATURE_NAMES[index] for index in pre_indices],
        "models": {
            "pre_primary_error": _serialize_logit(pre_primary),
            "post_secondary_primary_error": _serialize_logit(full_primary),
            "secondary_error": _serialize_logit(full_secondary),
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_path.with_suffix(".model.json").write_text(
            json.dumps(model_artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def write_frozen_uncertainty_routing_predictions(
    examples: Sequence[PreparedExample],
    primary_predictions: Mapping[str, Mapping],
    secondary_predictions: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]],
    model_artifact: Mapping,
    output_path: Path,
) -> dict:
    """Apply a frozen train/dev model without reading any references."""
    if any(example.split != "test" for example in examples):
        raise ValueError("Frozen UQ prediction expects test inputs only")
    if tuple(model_artifact.get("feature_names", ())) != UQ_FEATURE_NAMES:
        raise ValueError("Frozen model feature names do not match this implementation")
    pre_feature_names = tuple(model_artifact["pre_feature_names"])
    pre_indices = np.asarray(
        [UQ_FEATURE_NAMES.index(name) for name in pre_feature_names], dtype=int
    )
    models = model_artifact["models"]
    pre_primary = _deserialize_logit(models["pre_primary_error"])
    post_primary = _deserialize_logit(models["post_secondary_primary_error"])
    secondary = _deserialize_logit(models["secondary_error"])
    thresholds = model_artifact["thresholds"]
    intervention_threshold = float(thresholds["intervention_risk"])
    selective_threshold = float(thresholds["selective_acceptance"])
    delta_margin = float(model_artifact["configuration"]["delta_margin"])
    primary_name = str(model_artifact["configuration"]["primary_name"])
    secondary_name = str(model_artifact["configuration"]["secondary_name"])

    missing_primary = [
        item.question_id for item in examples if item.question_id not in primary_predictions
    ]
    missing_secondary = [
        item.question_id for item in examples if item.question_id not in secondary_predictions
    ]
    if missing_primary or missing_secondary:
        raise ValueError(
            "Incomplete frozen answer caches: "
            f"primary_missing={len(missing_primary)}, "
            f"secondary_missing={len(missing_secondary)}"
        )

    prepared = []
    for example in examples:
        features, metadata = build_uq_features(
            example,
            primary_predictions[example.question_id],
            secondary_predictions[example.question_id],
            extractions.get(example.question_id, ()),
        )
        prepared.append((example, features, metadata))
    matrix = np.vstack([item[1] for item in prepared])
    pre_risk = pre_primary.predict(matrix[:, pre_indices])
    post_primary_risk = post_primary.predict(matrix)
    secondary_risk = secondary.predict(matrix)

    rows = []
    for index, (example, _, metadata) in enumerate(prepared):
        primary_answer = metadata["luna_answer"]
        secondary_answer = metadata["mini_answer"]
        expected_gain = float(post_primary_risk[index] - secondary_risk[index])
        triggered = bool(pre_risk[index] >= intervention_threshold)
        disagrees = bool(metadata["model_disagreement"])
        eligible = bool(triggered and disagrees and not is_no_answer(secondary_answer))
        risk_only_switch = eligible
        conservative_switch = bool(eligible and expected_gain >= delta_margin)
        accepted = bool(pre_risk[index] <= selective_threshold)
        rows.append({
            "question_id": example.question_id,
            "primary_name": primary_name,
            "secondary_name": secondary_name,
            "primary_answer": primary_answer,
            "secondary_answer": secondary_answer,
            "primary_error_probability": float(pre_risk[index]),
            "post_secondary_primary_error_probability": float(post_primary_risk[index]),
            "secondary_error_probability": float(secondary_risk[index]),
            "expected_secondary_gain": expected_gain,
            "model_disagreement": disagrees,
            "secondary_triggered": triggered,
            "risk_only_answer": secondary_answer if risk_only_switch else primary_answer,
            "answer": secondary_answer if conservative_switch else primary_answer,
            "selective_answer": primary_answer if accepted else UNKNOWN,
            "routing_decision": "switch_to_secondary" if conservative_switch else "keep_primary",
            "risk_only_decision": "switch_to_secondary" if risk_only_switch else "keep_primary",
            "selective_decision": "accept" if accepted else "abstain",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    secondary_calls = sum(row["secondary_triggered"] for row in rows)
    switches = sum(row["routing_decision"] == "switch_to_secondary" for row in rows)
    return {
        "examples": len(rows),
        "primary_name": primary_name,
        "secondary_name": secondary_name,
        "secondary_calls": secondary_calls,
        "secondary_call_rate": secondary_calls / len(rows),
        "mean_llm_calls_per_question": 1.0 + secondary_calls / len(rows),
        "switches": switches,
        "output": str(output_path),
    }


def evaluate_frozen_uncertainty_routing_predictions(
    examples: Sequence[PreparedExample],
    predictions: Mapping[str, Mapping],
    evaluation_metadata: Mapping[str, Mapping],
    *,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    """Score an already-frozen prediction cache against separately loaded refs."""
    rows = []
    for example in examples:
        if example.question_id not in predictions:
            raise ValueError(f"Missing frozen prediction for {example.question_id}")
        if example.question_id not in evaluation_metadata:
            raise ValueError(f"Missing evaluation metadata for {example.question_id}")
        prediction = dict(predictions[example.question_id])
        metadata = evaluation_metadata[example.question_id]
        references = list(metadata["references"])
        prediction.update({
            "references": references,
            "gold_conflict": bool(metadata.get("has_conflict", False)),
            "primary_correct": _is_correct(str(prediction["primary_answer"]), references),
            "secondary_correct": _is_correct(str(prediction["secondary_answer"]), references),
        })
        rows.append(prediction)

    def routing_report(answer_key: str) -> dict:
        changed = [row for row in rows if not safe_surface_equivalent(
            str(row[answer_key]), str(row["primary_answer"])
        )]
        helped = sum(
            _is_correct(str(row[answer_key]), row["references"])
            and not row["primary_correct"]
            for row in changed
        )
        hurt = sum(
            not _is_correct(str(row[answer_key]), row["references"])
            and row["primary_correct"]
            for row in changed
        )
        baseline = _answer_metrics(rows, "primary_answer")["answer_exact"]
        metrics = _answer_metrics(rows, answer_key)
        return {
            **metrics,
            "switches": len(changed),
            "helped": helped,
            "hurt": hurt,
            "net_exact_changes": helped - hurt,
            "em_delta_vs_primary": metrics["answer_exact"] - baseline,
            "mcnemar_exact_p_two_sided": _paired_exact_p(helped, hurt),
            "bootstrap_95pct_em_delta": _bootstrap_delta(
                rows,
                answer_key,
                seed=seed + (17 if answer_key == "answer" else 29),
                baseline_correct_key="primary_correct",
            ),
        }

    conflict = [row for row in rows if row["gold_conflict"]]
    nonconflict = [row for row in rows if not row["gold_conflict"]]
    secondary_calls = sum(row["secondary_triggered"] for row in rows)
    summary = {
        "protocol": "frozen_official_test_evaluation",
        "protocol_version": UQ_ROUTING_PROTOCOL_VERSION,
        "examples": len(rows),
        "risk_quality": _risk_quality(
            rows,
            correct_key="primary_correct",
            score_key="primary_error_probability",
        ),
        "usage": {
            "secondary_calls": secondary_calls,
            "secondary_call_rate": secondary_calls / len(rows),
            "mean_llm_calls_per_question": 1.0 + secondary_calls / len(rows),
        },
        "overall": {
            "primary_direct_rag": _answer_metrics(rows, "primary_answer"),
            "secondary_direct_rag": _answer_metrics(rows, "secondary_answer"),
            "risk_only_route": routing_report("risk_only_answer"),
            "conservative_uq_route": routing_report("answer"),
            "selective_primary": _answer_metrics(rows, "selective_answer"),
        },
        "conflict_only": {
            key: _answer_metrics(conflict, answer_key)
            for key, answer_key in (
                ("primary_direct_rag", "primary_answer"),
                ("secondary_direct_rag", "secondary_answer"),
                ("conservative_uq_route", "answer"),
            )
        },
        "nonconflict_only": {
            key: _answer_metrics(nonconflict, answer_key)
            for key, answer_key in (
                ("primary_direct_rag", "primary_answer"),
                ("secondary_direct_rag", "secondary_answer"),
                ("conservative_uq_route", "answer"),
            )
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


__all__ = [
    "PRE_ROUTING_FEATURE_NAMES",
    "UQ_FEATURE_NAMES",
    "UQ_ROUTING_PROTOCOL_VERSION",
    "build_uq_features",
    "run_uncertainty_routing_cross_validation",
    "run_uncertainty_routing_train_dev",
    "write_frozen_uncertainty_routing_predictions",
    "evaluate_frozen_uncertainty_routing_predictions",
]
