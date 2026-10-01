from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .bayes import BayesianConditionalLogit
from .clustering import cluster_observations
from .dependency import normalize_domain, source_category, token_jaccard
from .map_gca import candidates_from_generation
from .normalization import answers_match, exact_match, is_no_answer, target_matches_candidate, token_f1
from .schemas import ChoiceSet, EvidenceObservation, PreparedExample


TWO_STAGE_PROTOCOL_VERSION = "3.0"

CONFLICT_FEATURE_NAMES = (
    "log_candidate_count",
    "supported_alternative_count",
    "explicit_contradiction_count",
    "evidence_insufficient",
    "confidence_low",
    "confidence_medium",
    "support_entropy",
    "top_support_share",
    "support_domain_count",
    "same_domain_support_rate",
    "mean_pairwise_answer_distance",
    "numeric_answer_disagreement",
    "temporal_answer_disagreement",
    "temporal_evidence_disagreement",
    "extracted_candidate_count",
    "extracted_support_entropy",
    "extracted_top_support_share",
    "extracted_pairwise_answer_distance",
    "extracted_numeric_disagreement",
    "extracted_temporal_disagreement",
    "mean_extraction_temporal_fit",
    "extraction_temporal_fit_range",
    "source_category_diversity",
    "external_authority_range",
)

RANK_FEATURE_NAMES = (
    "log_support",
    "support_share",
    "log_independent_domain_support",
    "independent_support_share",
    "domain_diversity",
    "same_domain_redundancy",
    "mean_directness",
    "max_entailment",
    "mean_entailment",
    "mean_answer_type_match",
    "mean_llm_temporal_fit",
    "question_time_match",
    "temporal_specificity",
    "temporal_conflict_rate",
    "official_source_share",
    "reference_source_share",
    "news_source_share",
    "social_source_share",
    "external_authority_mean",
    "external_metadata_coverage",
    "initial_answer_indicator",
)

_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?", re.I)
_YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
_MONTH = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.I,
)
_CURRENT_WORDS = re.compile(r"\b(?:current|currently|latest|today|now|present)\b", re.I)


def qacc_conflict_label(example: PreparedExample) -> bool:
    """QACC conflict target available on train/dev only.

    The prepared protocol retains the annotated answer list in supervision.
    Two or more distinct annotated answers correspond to the release's
    secondAnswerExist conflict signal. This function is never used at inference.
    """
    annotations = (example.supervision or {}).get("candidates", [])
    answers = {
        re.sub(r"\s+", " ", str(item.get("answer", "")).strip().lower())
        for item in annotations
        if str(item.get("answer", "")).strip()
    }
    return len(answers) >= 2


def _numbers(text: str) -> set[str]:
    return {value.replace(",", "").lower() for value in _NUMBER.findall(text or "")}


def _years(text: str) -> set[int]:
    return {int(value) for value in _YEAR.findall(text or "")}


def _has_temporal_expression(text: str) -> bool:
    return bool(_YEAR.search(text or "") or _MONTH.search(text or ""))


def _entropy(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    total = float(array.sum())
    if total <= 0 or len(array) <= 1:
        return 0.0
    probabilities = array / total
    raw = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return raw / math.log(len(array))


def load_domain_metadata(path: Path | None) -> dict[str, dict]:
    """Load a frozen, user-auditable domain metadata snapshot.

    Supported values per domain are category, authority in [0, 1], and
    independence_group. Missing domains receive neutral authority and retain
    the deterministic built-in category heuristic.
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Domain metadata must be a JSON object keyed by domain")
    result: dict[str, dict] = {}
    for raw_domain, raw_item in payload.items():
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"Domain metadata for {raw_domain!r} must be an object")
        domain = normalize_domain(str(raw_domain))
        authority = float(raw_item.get("authority", 0.5))
        if not 0.0 <= authority <= 1.0:
            raise ValueError(f"authority for {domain} must be in [0, 1]")
        category = str(raw_item.get("category", source_category(domain)))
        if category not in {"official", "reference", "news", "social", "other"}:
            raise ValueError(f"Unknown source category {category!r} for {domain}")
        result[domain] = {
            "authority": authority,
            "category": category,
            "independence_group": str(raw_item.get("independence_group", domain)),
        }
    return result


def _domain_attributes(source: str | None, metadata: Mapping[str, Mapping]) -> tuple[str, float, str, bool]:
    domain = normalize_domain(source)
    item = metadata.get(domain)
    if item is None:
        return source_category(domain), 0.5, domain, False
    return (
        str(item.get("category", source_category(domain))),
        float(item.get("authority", 0.5)),
        str(item.get("independence_group", domain)),
        True,
    )


def conflict_features(
    generation: Mapping,
    example: PreparedExample,
    observations: Sequence[EvidenceObservation] = (),
    domain_metadata: Mapping[str, Mapping] | None = None,
) -> np.ndarray:
    candidates = candidates_from_generation(generation)
    support_counts = [len({int(item["context_index"]) for item in candidate["evidence"]}) for candidate in candidates]
    total_support = sum(support_counts)
    top_share = max(support_counts, default=0) / total_support if total_support else 0.0
    cited_indices = {
        int(item["context_index"])
        for candidate in candidates
        for item in candidate["evidence"]
    }
    contexts = {context.index: context for context in example.contexts}
    domains = [normalize_domain(contexts[index].source) for index in cited_indices if index in contexts]
    same_domain_rate = (
        1.0 - len(set(domains)) / len(domains) if domains else 0.0
    )

    distances = []
    numeric_disagreements = []
    temporal_disagreements = []
    evidence_year_sets: list[set[int]] = []
    for candidate in candidates:
        text = " ".join(
            str(item.get("quote", "")) for item in candidate.get("evidence", [])
        )
        evidence_year_sets.append(_years(text))
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            distances.append(1.0 - token_jaccard(left["answer"], right["answer"]))
            left_numbers, right_numbers = _numbers(left["answer"]), _numbers(right["answer"])
            numeric_disagreements.append(float(bool(left_numbers and right_numbers and left_numbers != right_numbers)))
            left_years, right_years = _years(left["answer"]), _years(right["answer"])
            temporal_disagreements.append(float(bool(left_years and right_years and left_years != right_years)))
    evidence_temporal_pairs = []
    for left_index, left_years in enumerate(evidence_year_sets):
        for right_years in evidence_year_sets[left_index + 1 :]:
            evidence_temporal_pairs.append(float(bool(left_years and right_years and left_years != right_years)))

    extracted = cluster_observations(observations)
    extracted_counts = [
        len({item.context_index for item in candidate.observations})
        for candidate in extracted
    ]
    extracted_total = sum(extracted_counts)
    extracted_distances = []
    extracted_numeric = []
    extracted_temporal = []
    for left_index, left in enumerate(extracted):
        for right in extracted[left_index + 1 :]:
            extracted_distances.append(1.0 - token_jaccard(left.answer, right.answer))
            left_numbers, right_numbers = _numbers(left.answer), _numbers(right.answer)
            extracted_numeric.append(float(bool(left_numbers and right_numbers and left_numbers != right_numbers)))
            left_years, right_years = _years(left.answer), _years(right.answer)
            extracted_temporal.append(float(bool(left_years and right_years and left_years != right_years)))
    extraction_temporal_fit = [item.temporal_fit for item in observations]
    metadata = domain_metadata or {}
    extracted_context_indices = {item.context_index for item in observations}
    extracted_contexts = [
        context for context in example.contexts if context.index in extracted_context_indices
    ]
    extracted_attributes = [
        _domain_attributes(context.source, metadata) for context in extracted_contexts
    ]
    extracted_categories = {item[0] for item in extracted_attributes}
    extracted_authorities = [item[1] for item in extracted_attributes]

    confidence = str(generation.get("confidence", "low"))
    values = (
        math.log1p(len(candidates)),
        float(max(0, len(candidates) - 1)),
        float(len(generation.get("contradictions", []))),
        float(not generation.get("evidence_sufficient", False)),
        float(confidence == "low"),
        float(confidence == "medium"),
        _entropy(support_counts),
        top_share,
        float(len(set(domains))),
        same_domain_rate,
        float(np.mean(distances)) if distances else 0.0,
        float(np.mean(numeric_disagreements)) if numeric_disagreements else 0.0,
        float(np.mean(temporal_disagreements)) if temporal_disagreements else 0.0,
        float(np.mean(evidence_temporal_pairs)) if evidence_temporal_pairs else 0.0,
        float(len(extracted)),
        _entropy(extracted_counts),
        max(extracted_counts, default=0) / extracted_total if extracted_total else 0.0,
        float(np.mean(extracted_distances)) if extracted_distances else 0.0,
        float(np.mean(extracted_numeric)) if extracted_numeric else 0.0,
        float(np.mean(extracted_temporal)) if extracted_temporal else 0.0,
        float(np.mean(extraction_temporal_fit)) if extraction_temporal_fit else 0.5,
        (
            float(max(extraction_temporal_fit) - min(extraction_temporal_fit))
            if extraction_temporal_fit else 0.0
        ),
        len(extracted_categories) / 5.0,
        (
            float(max(extracted_authorities) - min(extracted_authorities))
            if extracted_authorities else 0.0
        ),
    )
    return np.asarray(values, dtype=float)


@dataclass
class BinaryFitSummary:
    examples: int
    positives: int
    converged: bool
    objective: float


class BayesianBinaryLogit:
    """Gaussian-prior binary logistic regression with a Laplace posterior."""

    def __init__(self, feature_names: Sequence[str], *, prior_scale: float = 1.0) -> None:
        self.feature_names = tuple(feature_names)
        self.prior_scale = float(prior_scale)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.threshold = 0.5

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> BinaryFitSummary:
        matrix = np.asarray(matrix, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("Conflict feature matrix has the wrong shape")
        if len(matrix) != len(labels) or len(matrix) == 0:
            raise ValueError("Conflict features and labels must be non-empty and aligned")
        if len(set(labels.tolist())) < 2:
            raise ValueError("Conflict fitting requires both classes")
        self.feature_mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.feature_scale = np.where(scale > 1e-8, scale, 1.0)
        transformed = (matrix - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(transformed)), transformed])
        precision = 1.0 / (self.prior_scale**2)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            scores = design @ beta
            probabilities = expit(scores)
            penalty = 0.5 * precision * float(beta[1:] @ beta[1:])
            loss = float(np.sum(np.logaddexp(0.0, scores) - labels * scores)) + penalty
            gradient = design.T @ (probabilities - labels)
            gradient[1:] += precision * beta[1:]
            return loss, gradient

        result = minimize(
            lambda beta: objective(beta),
            np.zeros(design.shape[1], dtype=float),
            jac=True,
            method="L-BFGS-B",
        )
        self.coefficients = np.asarray(result.x, dtype=float)
        probabilities = expit(design @ self.coefficients)
        weights = probabilities * (1.0 - probabilities)
        penalty = np.diag([1e-8, *([precision] * (design.shape[1] - 1))])
        hessian = design.T @ (design * weights[:, None]) + penalty
        self.covariance = np.linalg.pinv(hessian)
        return BinaryFitSummary(
            examples=len(labels),
            positives=int(labels.sum()),
            converged=bool(result.success),
            objective=float(result.fun),
        )

    def predict_probability(self, matrix: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None or self.coefficients is None:
            raise RuntimeError("Conflict detector is not fitted")
        matrix = np.asarray(matrix, dtype=float)
        transformed = (matrix - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(transformed)), transformed])
        return expit(design @ self.coefficients)

    def to_dict(self) -> dict:
        if self.feature_mean is None or self.feature_scale is None or self.coefficients is None or self.covariance is None:
            raise RuntimeError("Conflict detector is not fitted")
        return {
            "feature_names": list(self.feature_names),
            "prior_scale": self.prior_scale,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "covariance": self.covariance.tolist(),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "BayesianBinaryLogit":
        model = cls(payload["feature_names"], prior_scale=float(payload["prior_scale"]))
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=float)
        model.feature_scale = np.asarray(payload["feature_scale"], dtype=float)
        model.coefficients = np.asarray(payload["coefficients"], dtype=float)
        model.covariance = np.asarray(payload["covariance"], dtype=float)
        model.threshold = float(payload["threshold"])
        return model


def _stratified_folds(labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % folds].append(int(index))
    return [np.asarray(sorted(bucket), dtype=int) for bucket in buckets if bucket]


def _select_f1_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    candidates = sorted({0.0, 0.5, 1.0, *probabilities.tolist()})
    rows = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum((~predictions) & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((f1, precision, recall, -abs(threshold - 0.5), threshold, tp, fp, fn))
    best = max(rows)
    return float(best[4]), {
        "precision": best[1], "recall": best[2], "f1": best[0],
        "tp": best[5], "fp": best[6], "fn": best[7],
    }


def fit_conflict_detector(
    examples: Sequence[PreparedExample],
    generations: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]] | None = None,
    domain_metadata: Mapping[str, Mapping] | None = None,
    *,
    prior_scale: float = 1.0,
    folds: int = 5,
    seed: int = 42,
) -> tuple[BayesianBinaryLogit, dict]:
    usable = [example for example in examples if example.question_id in generations]
    matrix = np.vstack([
        conflict_features(
            generations[item.question_id].get("generation", generations[item.question_id]),
            item,
            (extractions or {}).get(item.question_id, []),
            domain_metadata,
        )
        for item in usable
    ])
    labels = np.asarray([qacc_conflict_label(item) for item in usable], dtype=int)
    fold_indices = _stratified_folds(labels, max(2, min(folds, int(labels.sum()))), seed)
    oof = np.zeros(len(labels), dtype=float)
    all_indices = np.arange(len(labels))
    for held_out in fold_indices:
        train_indices = np.setdiff1d(all_indices, held_out)
        fold_model = BayesianBinaryLogit(CONFLICT_FEATURE_NAMES, prior_scale=prior_scale)
        fold_model.fit(matrix[train_indices], labels[train_indices])
        oof[held_out] = fold_model.predict_probability(matrix[held_out])
    threshold, oof_metrics = _select_f1_threshold(oof, labels)
    model = BayesianBinaryLogit(CONFLICT_FEATURE_NAMES, prior_scale=prior_scale)
    fit = model.fit(matrix, labels)
    model.threshold = threshold
    return model, {
        "fit": asdict(fit),
        "oof_folds": len(fold_indices),
        "oof_threshold": threshold,
        "oof_metrics": oof_metrics,
        "missing_generations": len(examples) - len(usable),
    }


def _observation_lookup(observations: Sequence[EvidenceObservation]) -> dict[int, list[EvidenceObservation]]:
    result: dict[int, list[EvidenceObservation]] = {}
    for observation in observations:
        result.setdefault(observation.context_index, []).append(observation)
    return result


def _matching_observation(
    answer: str,
    context_index: int,
    observations: Mapping[int, Sequence[EvidenceObservation]],
) -> EvidenceObservation | None:
    matches = [
        item for item in observations.get(context_index, [])
        if answers_match(answer, item.candidate_answer)
        or target_matches_candidate(answer, item.candidate_answer)
        or target_matches_candidate(item.candidate_answer, answer)
    ]
    return max(matches, key=lambda item: item.entailment, default=None)


def rank_feature_matrix(
    generation: Mapping,
    example: PreparedExample,
    observations: Sequence[EvidenceObservation] = (),
    domain_metadata: Mapping[str, Mapping] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    candidates = candidates_from_generation(generation)
    answers = tuple(str(item["answer"]) for item in candidates)
    if not candidates:
        return answers, np.empty((0, len(RANK_FEATURE_NAMES)), dtype=float)
    metadata = domain_metadata or {}
    contexts = {context.index: context for context in example.contexts}
    by_context = _observation_lookup(observations)
    support_counts = [len({int(item["context_index"]) for item in candidate["evidence"]}) for candidate in candidates]
    total_support = sum(support_counts) or 1
    independent_sets = []
    for candidate in candidates:
        groups = set()
        for citation in candidate["evidence"]:
            context = contexts.get(int(citation["context_index"]))
            if context is None:
                continue
            _, _, independence_group, _ = _domain_attributes(context.source, metadata)
            groups.add(independence_group)
        independent_sets.append(groups)
    total_independent = sum(len(groups) for groups in independent_sets) or 1
    question_years = _years(example.question)
    current_question = bool(_CURRENT_WORDS.search(example.question))
    all_context_years = {year for context in example.contexts for year in _years(context.text)}
    latest_context_year = max(all_context_years, default=None)
    candidate_years = []
    for candidate in candidates:
        combined = " ".join(
            f"{citation.get('quote', '')} {contexts.get(int(citation['context_index'])).text if int(citation['context_index']) in contexts else ''}"
            for citation in candidate["evidence"]
        )
        candidate_years.append(_years(combined))

    rows = []
    initial_answer = str(generation.get("answer", ""))
    for index, candidate in enumerate(candidates):
        citations = candidate["evidence"]
        context_indices = list(dict.fromkeys(int(item["context_index"]) for item in citations))
        candidate_contexts = [contexts[value] for value in context_indices if value in contexts]
        domains = [normalize_domain(context.source) for context in candidate_contexts]
        attributes = [_domain_attributes(context.source, metadata) for context in candidate_contexts]
        categories = Counter(value[0] for value in attributes)
        category_total = len(attributes) or 1
        authority = [value[1] for value in attributes]
        metadata_coverage = [value[3] for value in attributes]
        independent_count = len(independent_sets[index])
        matched = [
            match
            for context_index in context_indices
            if (match := _matching_observation(candidate["answer"], context_index, by_context)) is not None
        ]
        directness = [item.directness for item in matched]
        entailment = [item.entailment for item in matched]
        answer_type = [item.answer_type_match for item in matched]
        temporal_fit = [item.temporal_fit for item in matched]
        years = candidate_years[index]
        if question_years:
            time_match = len(question_years & years) / len(question_years)
        elif current_question and latest_context_year is not None and years:
            time_match = float(max(years) == latest_context_year)
        else:
            time_match = 0.5
        competing_years = set().union(*(value for pos, value in enumerate(candidate_years) if pos != index))
        temporal_conflict = float(bool(years and competing_years and years != competing_years))
        raw_count = support_counts[index]
        rows.append([
            math.log1p(raw_count),
            raw_count / total_support,
            math.log1p(independent_count),
            independent_count / total_independent,
            len(set(domains)) / raw_count if raw_count else 0.0,
            raw_count / independent_count if independent_count else 0.0,
            float(np.mean(directness)) if directness else 0.5,
            max(entailment, default=0.5),
            float(np.mean(entailment)) if entailment else 0.5,
            float(np.mean(answer_type)) if answer_type else 0.5,
            float(np.mean(temporal_fit)) if temporal_fit else 0.5,
            time_match,
            float(any(_has_temporal_expression(context.text) for context in candidate_contexts)),
            temporal_conflict,
            categories["official"] / category_total,
            categories["reference"] / category_total,
            categories["news"] / category_total,
            categories["social"] / category_total,
            float(np.mean(authority)) if authority else 0.5,
            float(np.mean(metadata_coverage)) if metadata_coverage else 0.0,
            float(answers_match(candidate["answer"], initial_answer)),
        ])
    return answers, np.asarray(rows, dtype=float)


def build_rank_choice_sets(
    examples: Sequence[PreparedExample],
    generations: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]] | None = None,
    domain_metadata: Mapping[str, Mapping] | None = None,
    *,
    require_label: bool,
) -> tuple[list[ChoiceSet], dict]:
    result = []
    no_generation = no_candidates = single_candidate = label_unmapped = 0
    for example in examples:
        record = generations.get(example.question_id)
        if record is None:
            no_generation += 1
            continue
        generation = record.get("generation", record)
        answers, features = rank_feature_matrix(
            generation,
            example,
            (extractions or {}).get(example.question_id, []),
            domain_metadata,
        )
        if not answers:
            no_candidates += 1
            continue
        correct_index = None
        target = str((example.supervision or {}).get("correct_answer", ""))
        if target:
            matches = [index for index, answer in enumerate(answers) if target_matches_candidate(target, answer)]
            if matches:
                correct_index = matches[0]
        if require_label and correct_index is None:
            label_unmapped += 1
            continue
        if len(answers) < 2:
            single_candidate += 1
            if require_label:
                continue
        result.append(ChoiceSet(example.question_id, answers, features, correct_index=correct_index))
    return result, {
        "examples": len(examples), "built": len(result), "no_generation": no_generation,
        "no_candidates": no_candidates, "single_candidate": single_candidate,
        "label_unmapped": label_unmapped,
    }


def _ranker_payload(model: BayesianConditionalLogit) -> dict:
    coefficients, covariance = model._require_fit()
    assert model.feature_mean is not None and model.feature_scale is not None
    return {
        "feature_names": list(model.feature_names),
        "prior_scale": model.prior_scale,
        "posterior_samples": model.posterior_samples,
        "label_smoothing": model.label_smoothing,
        "seed": model.seed,
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "map_coefficients": coefficients.tolist(),
        "covariance": covariance.tolist(),
        "temperature": model.temperature,
    }


def _ranker_from_payload(payload: Mapping) -> BayesianConditionalLogit:
    model = BayesianConditionalLogit(
        payload["feature_names"], prior_scale=float(payload["prior_scale"]),
        posterior_samples=int(payload["posterior_samples"]),
        label_smoothing=float(payload["label_smoothing"]), seed=int(payload["seed"]),
    )
    model.feature_mean = np.asarray(payload["feature_mean"], dtype=float)
    model.feature_scale = np.asarray(payload["feature_scale"], dtype=float)
    model.map_coefficients = np.asarray(payload["map_coefficients"], dtype=float)
    model.covariance = np.asarray(payload["covariance"], dtype=float)
    model.temperature = float(payload.get("temperature", 1.0))
    return model


def select_ranker_acceptance_oof(
    choice_sets: Sequence[ChoiceSet],
    generations: Mapping[str, Mapping],
    *,
    prior_scale: float,
    posterior_samples: int,
    label_smoothing: float,
    folds: int,
    seed: int,
) -> dict:
    """Select conservative answer-switch thresholds without touching dev.

    The objective is end-to-end accuracy relative to retaining the joint
    generator's primary answer. Ties prefer fewer interventions.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(choice_sets))
    rng.shuffle(indices)
    fold_indices = [value for value in np.array_split(indices, min(folds, len(indices))) if len(value)]
    rows = []
    all_indices = np.arange(len(choice_sets))
    for held_out in fold_indices:
        train_indices = np.setdiff1d(all_indices, held_out)
        if len(train_indices) == 0:
            continue
        model = BayesianConditionalLogit(
            RANK_FEATURE_NAMES,
            prior_scale=prior_scale,
            posterior_samples=posterior_samples,
            label_smoothing=label_smoothing,
            seed=seed,
        )
        model.fit([choice_sets[int(index)] for index in train_indices])
        for index in held_out:
            item = choice_sets[int(index)]
            values = model.posterior_probabilities(item)
            order = np.argsort(-values)
            top = int(order[0])
            second = float(values[int(order[1])])
            generation_record = generations[item.question_id]
            generation = generation_record.get("generation", generation_record)
            initial = str(generation.get("answer", ""))
            initial_indices = [
                position for position, answer in enumerate(item.answers)
                if answers_match(answer, initial)
            ]
            initial_correct = bool(
                initial_indices and item.correct_index is not None
                and initial_indices[0] == item.correct_index
            )
            rows.append({
                "maximum": float(values[top]),
                "margin": float(values[top] - second),
                "top_correct": bool(item.correct_index is not None and top == item.correct_index),
                "initial_correct": initial_correct,
                "changes_answer": not initial_indices or top != initial_indices[0],
            })
    probability_grid = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    margin_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]
    candidates = []
    for probability in probability_grid:
        for margin in margin_grid:
            correct = 0
            switches = helped = hurt = 0
            for row in rows:
                accept = row["maximum"] >= probability and row["margin"] >= margin
                use_ranker = accept and row["changes_answer"]
                outcome = row["top_correct"] if use_ranker else row["initial_correct"]
                correct += int(outcome)
                switches += int(use_ranker)
                helped += int(use_ranker and row["top_correct"] and not row["initial_correct"])
                hurt += int(use_ranker and not row["top_correct"] and row["initial_correct"])
            accuracy = correct / len(rows) if rows else 0.0
            candidates.append((accuracy, -switches, probability, margin, helped, hurt))
    best = max(candidates)
    baseline = float(np.mean([row["initial_correct"] for row in rows])) if rows else 0.0
    return {
        "examples": len(rows),
        "folds": len(fold_indices),
        "baseline_accuracy": baseline,
        "selected_accuracy": best[0],
        "accept_probability": best[2],
        "accept_margin": best[3],
        "switches": -best[1],
        "helped": best[4],
        "hurt": best[5],
    }


class TwoStageConflictModel:
    def __init__(
        self,
        detector: BayesianBinaryLogit,
        ranker: BayesianConditionalLogit,
        *,
        accept_probability: float = 0.60,
        accept_margin: float = 0.10,
        domain_metadata: Mapping[str, Mapping] | None = None,
    ) -> None:
        self.detector = detector
        self.ranker = ranker
        self.accept_probability = float(accept_probability)
        self.accept_margin = float(accept_margin)
        self.domain_metadata = dict(domain_metadata or {})

    def save(self, path: Path) -> None:
        payload = {
            "model_type": "two_stage_conflict_then_rank",
            "protocol_version": TWO_STAGE_PROTOCOL_VERSION,
            "detector": self.detector.to_dict(),
            "ranker": _ranker_payload(self.ranker),
            "accept_probability": self.accept_probability,
            "accept_margin": self.accept_margin,
            "domain_metadata": self.domain_metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TwoStageConflictModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            BayesianBinaryLogit.from_dict(payload["detector"]),
            _ranker_from_payload(payload["ranker"]),
            accept_probability=float(payload["accept_probability"]),
            accept_margin=float(payload["accept_margin"]),
            domain_metadata=payload.get("domain_metadata", {}),
        )

    def resolve(
        self,
        example: PreparedExample,
        generation_record: Mapping,
        observations: Sequence[EvidenceObservation] = (),
        *,
        mode: str = "conservative",
    ) -> dict:
        if mode not in {"map", "conservative"}:
            raise ValueError("mode must be 'map' or 'conservative'")
        generation = generation_record.get("generation", generation_record)
        initial = str(generation.get("answer", "unknown"))
        detector_matrix = conflict_features(
            generation, example, observations, self.domain_metadata
        )[None, :]
        conflict_probability = float(self.detector.predict_probability(detector_matrix)[0])
        predicted_conflict = conflict_probability >= self.detector.threshold
        answers, features = rank_feature_matrix(
            generation, example, observations, self.domain_metadata
        )
        probabilities: list[float] = []
        selected_index = None
        accepted = False
        final = initial
        decision = "direct_initial"
        if predicted_conflict and len(answers) >= 2:
            choice_set = ChoiceSet(example.question_id, answers, features)
            values = self.ranker.posterior_probabilities(choice_set)
            probabilities = [float(value) for value in values]
            order = np.argsort(-values)
            selected_index = int(order[0])
            maximum = float(values[selected_index])
            second = float(values[int(order[1])])
            accepted = mode == "map" or (
                maximum >= self.accept_probability
                and maximum - second >= self.accept_margin
            )
            if accepted:
                final = answers[selected_index]
                decision = "conflict_ranker"
            else:
                decision = "conflict_preserve_initial"
        elif predicted_conflict:
            decision = "conflict_no_choice_set"
        return {
            "initial_answer": initial,
            "final_answer": final,
            "conflict_probability": conflict_probability,
            "conflict_threshold": self.detector.threshold,
            "predicted_conflict": predicted_conflict,
            "decision": decision,
            "candidate_answers": list(answers),
            "candidate_probabilities": probabilities,
            "selected_candidate_index": selected_index,
            "ranker_accepted": accepted,
        }


def fit_two_stage_model(
    examples: Sequence[PreparedExample],
    generations: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]] | None = None,
    domain_metadata: Mapping[str, Mapping] | None = None,
    *,
    detector_prior_scale: float = 1.0,
    ranker_prior_scale: float = 1.0,
    posterior_samples: int = 1000,
    label_smoothing: float = 0.05,
    folds: int = 5,
    seed: int = 42,
    accept_probability: float | None = None,
    accept_margin: float | None = None,
) -> tuple[TwoStageConflictModel, dict]:
    detector, detector_report = fit_conflict_detector(
        examples,
        generations,
        extractions,
        domain_metadata,
        prior_scale=detector_prior_scale,
        folds=folds,
        seed=seed,
    )
    choice_sets, build_report = build_rank_choice_sets(
        examples, generations, extractions, domain_metadata, require_label=True
    )
    ranker = BayesianConditionalLogit(
        RANK_FEATURE_NAMES, prior_scale=ranker_prior_scale,
        posterior_samples=posterior_samples, label_smoothing=label_smoothing, seed=seed,
    )
    ranker_fit = ranker.fit(choice_sets)
    acceptance_report = select_ranker_acceptance_oof(
        choice_sets,
        generations,
        prior_scale=ranker_prior_scale,
        posterior_samples=posterior_samples,
        label_smoothing=label_smoothing,
        folds=folds,
        seed=seed,
    )
    resolved_accept_probability = (
        float(accept_probability)
        if accept_probability is not None
        else float(acceptance_report["accept_probability"])
    )
    resolved_accept_margin = (
        float(accept_margin)
        if accept_margin is not None
        else float(acceptance_report["accept_margin"])
    )
    model = TwoStageConflictModel(
        detector, ranker, accept_probability=resolved_accept_probability,
        accept_margin=resolved_accept_margin, domain_metadata=domain_metadata,
    )
    return model, {
        "detector": detector_report,
        "ranker_build": build_report,
        "ranker_fit": asdict(ranker_fit),
        "ranker_acceptance_oof": acceptance_report,
        "ranker_acceptance_overridden": {
            "probability": accept_probability is not None,
            "margin": accept_margin is not None,
        },
        "rank_features": list(RANK_FEATURE_NAMES),
        "domain_metadata_entries": len(domain_metadata or {}),
    }


def write_two_stage_predictions(
    examples: Sequence[PreparedExample],
    generations: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]],
    model: TwoStageConflictModel,
    output_path: Path,
    *,
    mode: str,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    predicted_conflicts = 0
    accepted = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = generations.get(example.question_id)
            if record is None:
                missing += 1
                result = {
                    "initial_answer": "unknown", "final_answer": "unknown",
                    "conflict_probability": 0.0, "conflict_threshold": model.detector.threshold,
                    "predicted_conflict": False, "decision": "missing_generation",
                    "candidate_answers": [], "candidate_probabilities": [],
                    "selected_candidate_index": None, "ranker_accepted": False,
                }
            else:
                result = model.resolve(
                    example, record, extractions.get(example.question_id, []), mode=mode
                )
            predicted_conflicts += int(result["predicted_conflict"])
            accepted += int(result["ranker_accepted"])
            handle.write(json.dumps({"question_id": example.question_id, **result}, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "examples": len(examples), "missing_generations": missing,
        "predicted_conflicts": predicted_conflicts, "ranker_accepted": accepted,
        "output": str(output_path),
    }


def read_two_stage_predictions(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                result[str(item["question_id"])] = item
    return result


def evaluate_two_stage_predictions(
    examples: Sequence[PreparedExample],
    predictions: Mapping[str, Mapping],
    *,
    references: Mapping[str, Sequence[str]] | None = None,
    conflict_labels: Mapping[str, bool] | None = None,
    output_path: Path | None = None,
) -> dict:
    rows = []
    for example in examples:
        prediction = predictions.get(example.question_id, {})
        if references is None:
            gold = str((example.supervision or {}).get("correct_answer", ""))
            item_references = [gold] if gold else []
        else:
            item_references = list(references.get(example.question_id, []))
        gold_conflict = (
            bool(conflict_labels[example.question_id])
            if conflict_labels is not None and example.question_id in conflict_labels
            else qacc_conflict_label(example)
        )
        initial = str(prediction.get("initial_answer", "unknown"))
        final = str(prediction.get("final_answer", "unknown"))
        row = {
            "question_id": example.question_id,
            "gold_conflict": gold_conflict,
            "predicted_conflict": bool(prediction.get("predicted_conflict", False)),
            "conflict_probability": float(prediction.get("conflict_probability", 0.0)),
            "decision": prediction.get("decision", "missing"),
            "initial_answer": initial,
            "final_answer": final,
            "initial_exact": float(not is_no_answer(initial) and any(exact_match(initial, ref) for ref in item_references)),
            "final_exact": float(not is_no_answer(final) and any(exact_match(final, ref) for ref in item_references)),
            "initial_f1": max((token_f1(initial, ref) for ref in item_references), default=0.0),
            "final_f1": max((token_f1(final, ref) for ref in item_references), default=0.0),
            "initial_abstained": is_no_answer(initial),
            "final_abstained": is_no_answer(final),
        }
        rows.append(row)
    tp = sum(row["gold_conflict"] and row["predicted_conflict"] for row in rows)
    fp = sum(not row["gold_conflict"] and row["predicted_conflict"] for row in rows)
    fn = sum(row["gold_conflict"] and not row["predicted_conflict"] for row in rows)
    tn = sum(not row["gold_conflict"] and not row["predicted_conflict"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    gold_conflict_rows = [row for row in rows if row["gold_conflict"]]
    predicted_conflict_rows = [row for row in rows if row["predicted_conflict"]]

    def answer_metrics(selected: Sequence[dict], prefix: str) -> dict:
        covered = [row for row in selected if not row[f"{prefix}_abstained"]]
        return {
            "examples": len(selected),
            "answer_exact": float(np.mean([row[f"{prefix}_exact"] for row in selected])) if selected else 0.0,
            "answer_token_f1": float(np.mean([row[f"{prefix}_f1"] for row in selected])) if selected else 0.0,
            "coverage": len(covered) / len(selected) if selected else 0.0,
        }

    summary = {
        "conflict_detection": {
            "examples": len(rows), "gold_positive": tp + fn, "predicted_positive": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        },
        "initial_generation": answer_metrics(rows, "initial"),
        "two_stage_final": answer_metrics(rows, "final"),
        "gold_conflict_initial": answer_metrics(gold_conflict_rows, "initial"),
        "gold_conflict_final": answer_metrics(gold_conflict_rows, "final"),
        "predicted_conflict_initial": answer_metrics(predicted_conflict_rows, "initial"),
        "predicted_conflict_final": answer_metrics(predicted_conflict_rows, "final"),
        "changes": {
            "helped": sum(row["initial_exact"] == 0 and row["final_exact"] == 1 for row in rows),
            "hurt": sum(row["initial_exact"] == 1 and row["final_exact"] == 0 for row in rows),
            "changed": sum(not answers_match(row["initial_answer"], row["final_answer"]) for row in rows),
        },
        "decisions": dict(Counter(str(row["decision"]) for row in rows)),
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


__all__ = [
    "CONFLICT_FEATURE_NAMES", "RANK_FEATURE_NAMES", "TWO_STAGE_PROTOCOL_VERSION",
    "BayesianBinaryLogit", "TwoStageConflictModel", "build_rank_choice_sets",
    "conflict_features", "evaluate_two_stage_predictions", "fit_two_stage_model",
    "load_domain_metadata", "qacc_conflict_label", "read_two_stage_predictions",
    "write_two_stage_predictions",
]
