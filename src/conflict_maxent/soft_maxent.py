"""Small utilities shared by the three passage-weight conditions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .clustering import cluster_observations
from .source_utils import normalize_domain
from .schemas import EvidenceObservation, PreparedExample


@dataclass(frozen=True)
class ProjectionResult:
    """Result and diagnostics for one soft information projection."""

    distribution: np.ndarray
    lambdas: np.ndarray
    converged: bool
    iterations: int
    max_violation: float
    mean_violation: float
    total_violation: float
    violated_constraints: int


def _logsumexp(values: np.ndarray) -> float:
    """Stable log of a sum of exponentials."""

    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def soft_information_projection(
    prior: Sequence[float],
    feature_matrix: np.ndarray,
    targets: Sequence[float],
    *,
    signs: Sequence[float] | None = None,
    slack_penalty: float = 5.0,
    max_iter: int = 200,
    tolerance: float = 1e-6,
) -> ProjectionResult:
    """Minimize KL(q || prior) plus linear penalties for unmet constraints.

    A positive sign imposes ``E_q[g] >= target``. A negative sign imposes
    ``E_q[g] <= target``.
    The positive slack penalty gives the dual variables an upper bound, so the
    optimization remains defined even when all requested targets cannot be met.
    """

    prior_array = np.asarray(prior, dtype=float)
    matrix = np.asarray(feature_matrix, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    sign_array = (
        np.ones(len(target_array), dtype=float)
        if signs is None
        else np.asarray(signs, dtype=float)
    )
    if prior_array.ndim != 1 or not len(prior_array):
        raise ValueError("prior must be a non-empty vector")
    if matrix.shape != (len(prior_array), len(target_array)):
        raise ValueError("feature matrix shape does not match prior and targets")
    if sign_array.shape != target_array.shape or not np.all(
        np.isin(sign_array, (-1.0, 1.0))
    ):
        raise ValueError("signs must contain one +1 or -1 per target")
    if not np.all(np.isfinite(prior_array)) or np.any(prior_array <= 0):
        raise ValueError("prior must be finite and strictly positive")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target_array)):
        raise ValueError("features and targets must be finite")
    if slack_penalty <= 0 or max_iter <= 0 or tolerance <= 0:
        raise ValueError("optimizer settings must be positive")

    prior_array = prior_array / prior_array.sum()
    if matrix.shape[1] == 0:
        return ProjectionResult(
            prior_array.copy(), np.zeros(0), True, 0, 0.0, 0.0, 0.0, 0
        )

    log_prior = np.log(prior_array)
    signed_matrix = matrix * sign_array[None, :]
    signed_targets = target_array * sign_array

    def dual_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
        logits = log_prior + signed_matrix @ values
        distribution = np.exp(logits - _logsumexp(logits))
        value = _logsumexp(logits) - float(values @ signed_targets)
        gradient = signed_matrix.T @ distribution - signed_targets
        return value, gradient

    # The slack penalty is the upper bound of each dual variable.
    optimized = minimize(
        dual_and_gradient,
        np.zeros(matrix.shape[1]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, slack_penalty)] * matrix.shape[1],
        options={"maxiter": max_iter, "ftol": tolerance * 0.01, "gtol": tolerance},
    )
    lambdas = np.asarray(optimized.x, dtype=float)
    converged = bool(optimized.success)
    iteration = int(optimized.nit)

    logits = log_prior + signed_matrix @ lambdas
    distribution = np.exp(logits - _logsumexp(logits))
    violations = np.maximum(
        0.0, -sign_array * (matrix.T @ distribution - target_array)
    )
    practical_tolerance = max(1e-5, tolerance)
    return ProjectionResult(
        distribution=distribution,
        lambdas=lambdas,
        converged=converged,
        iterations=iteration,
        max_violation=float(violations.max(initial=0.0)),
        mean_violation=float(violations.mean()) if len(violations) else 0.0,
        total_violation=float(violations.sum()),
        violated_constraints=int((violations > practical_tolerance).sum()),
    )


def passage_features(
    example: PreparedExample,
    observations: Sequence[EvidenceObservation],
) -> list[dict[str, float | str | int | None]]:
    """Create only the fields needed by independent-source voting."""

    observation_by_context = {item.context_index: item for item in observations}
    context_by_index = {item.index: item for item in example.contexts}
    clusters = cluster_observations(observations)
    support_by_context: dict[int, float] = {}
    raw_support: list[int] = []
    cluster_domains: list[set[str]] = []
    for cluster in clusters:
        domains = {
            normalize_domain(context_by_index[item.context_index].source)
            for item in cluster.observations
            if item.context_index in context_by_index
        }
        cluster_domains.append(domains)
        raw_support.append(len(domains))
    maximum_support = max(raw_support, default=1)
    for cluster, domains in zip(clusters, cluster_domains):
        value = len(domains) / maximum_support
        for item in cluster.observations:
            support_by_context[item.context_index] = value

    result: list[dict[str, float | str | int | None]] = []
    for context in example.contexts:
        observation = observation_by_context.get(context.index)
        domain = normalize_domain(context.source)
        result.append(
            {
                "context_index": context.index,
                "source": context.source,
                "domain": domain,
                "candidate_answer": (
                    observation.candidate_answer if observation is not None else None
                ),
                "independent_answer_support": support_by_context.get(context.index, 0.0),
            }
        )
    return result


def _support_prior(features: Sequence[Mapping[str, object]], uniform_mix: float) -> np.ndarray:
    """Turn independent answer support into a strictly positive prior."""

    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be between zero and one")
    count = len(features)
    if not count:
        raise ValueError("cannot weight an example with no contexts")
    scores = np.asarray(
        [max(0.0, float(item["independent_answer_support"])) for item in features]
    )
    uniform = np.full(count, 1.0 / count)
    relative = scores / scores.sum() if scores.sum() > 0 else uniform
    prior = (1.0 - uniform_mix) * relative + uniform_mix * uniform
    return prior / prior.sum()


def support_voting_weights(
    example: PreparedExample,
    observations: Sequence[EvidenceObservation],
    *,
    uniform_mix: float = 0.1,
) -> tuple[np.ndarray, list[dict[str, float | str | int | None]]]:
    """Return the independent-source voting baseline and its features."""

    features = passage_features(example, observations)
    return _support_prior(features, uniform_mix), features


def read_weight_cache(path: Path) -> dict[str, dict]:
    """Read weight records and reject duplicate question ids."""

    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = str(record["question_id"])
            if question_id in records:
                raise ValueError(f"Duplicate question_id in {path}: {question_id}")
            records[question_id] = record
    return records


__all__ = [
    "ProjectionResult",
    "passage_features",
    "read_weight_cache",
    "soft_information_projection",
    "support_voting_weights",
]
