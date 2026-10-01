from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .bayes import BayesianConditionalLogit
from .clustering import cluster_observations
from .cross_model import safe_surface_equivalent
from .dependency import normalize_domain, source_category
from .normalization import (
    exact_match,
    is_no_answer,
    normalize_answer,
    target_matches_candidate,
    token_f1,
)
from .schemas import ChoiceSet, EvidenceObservation, PreparedExample
from .two_stage import qacc_conflict_label


DISAGREEMENT_PROTOCOL_VERSION = "1.0"
NONE_ANSWER = "unknown"

DISAGREEMENT_FEATURE_NAMES = (
    "mini_indicator",
    "none_indicator",
    "mini_x_conflict_strength",
    "none_x_conflict_strength",
    "log_support",
    "log_independent_support",
    "mean_entailment",
    "mean_directness",
    "mean_answer_type_match",
    "mean_temporal_fit",
    "official_support_share",
    "reference_support_share",
    "exact_context_mention_rate",
    "unsupported_token_rate",
    "log_answer_length",
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _hash_fraction(value: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _answer(record: Mapping | None) -> str:
    return str((record or {}).get("answer", NONE_ANSWER)).strip() or NONE_ANSWER


def _conflict_strength(observations: Sequence[EvidenceObservation]) -> float:
    clusters = cluster_observations(observations)
    if len(clusters) <= 1:
        return 0.0
    counts = np.asarray(
        [len({item.context_index for item in cluster.observations}) for cluster in clusters],
        dtype=float,
    )
    probabilities = counts / max(float(counts.sum()), 1.0)
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    normalized_entropy = entropy / math.log(len(clusters))
    answers = [normalize_answer(cluster.answer) for cluster in clusters]
    numeric_disagreement = any(
        set(_NUMBER.findall(left)) != set(_NUMBER.findall(right))
        and _NUMBER.search(left) is not None
        and _NUMBER.search(right) is not None
        for index, left in enumerate(answers)
        for right in answers[index + 1 :]
    )
    return min(1.0, 0.75 * normalized_entropy + 0.25 * float(numeric_disagreement))


def _matches_observation(answer: str, observation: EvidenceObservation) -> bool:
    return (
        safe_surface_equivalent(answer, observation.candidate_answer)
        or target_matches_candidate(answer, observation.candidate_answer)
        or target_matches_candidate(observation.candidate_answer, answer)
    )


def _candidate_row(
    example: PreparedExample,
    answer: str,
    origin: str,
    observations: Sequence[EvidenceObservation],
    conflict_strength: float,
) -> list[float]:
    is_none = origin == "none"
    is_mini = origin == "mini"
    if is_none:
        return [
            0.0, 1.0, 0.0, conflict_strength,
            0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0, 0.0,
        ]

    matched = [item for item in observations if _matches_observation(answer, item)]
    context_by_index = {context.index: context for context in example.contexts}
    support_indices = {item.context_index for item in matched}
    domains = {
        normalize_domain(context_by_index[index].source)
        for index in support_indices
        if index in context_by_index
    }
    categories = Counter(source_category(domain) for domain in domains)
    category_total = len(domains) or 1
    normalized = normalize_answer(answer)
    answer_tokens = normalized.split()
    context_keys = [normalize_answer(context.text) for context in example.contexts]
    mentions = sum(bool(normalized and normalized in key) for key in context_keys)
    context_tokens = set(" ".join(context_keys).split())
    unsupported = (
        sum(token not in context_tokens for token in answer_tokens) / len(answer_tokens)
        if answer_tokens else 1.0
    )

    def mean(name: str) -> float:
        values = [float(getattr(item, name)) for item in matched]
        return float(np.mean(values)) if values else 0.5

    return [
        float(is_mini),
        0.0,
        float(is_mini) * conflict_strength,
        0.0,
        math.log1p(len(support_indices)),
        math.log1p(len(domains)),
        mean("entailment"),
        mean("directness"),
        mean("answer_type_match"),
        mean("temporal_fit"),
        categories["official"] / category_total,
        categories["reference"] / category_total,
        mentions / max(len(example.contexts), 1),
        unsupported,
        math.log1p(len(answer_tokens)),
    ]


def build_disagreement_choice(
    example: PreparedExample,
    luna_record: Mapping | None,
    mini_record: Mapping | None,
    observations: Sequence[EvidenceObservation] = (),
) -> tuple[ChoiceSet | None, dict]:
    luna = _answer(luna_record)
    mini = _answer(mini_record)
    available: list[tuple[str, str]] = []
    if not is_no_answer(luna):
        available.append((luna, "luna"))
    if not is_no_answer(mini):
        existing = next(
            (index for index, (answer, _) in enumerate(available)
             if safe_surface_equivalent(answer, mini)),
            None,
        )
        if existing is None:
            available.append((mini, "mini"))
        else:
            current, origin = available[existing]
            if len(normalize_answer(mini).split()) < len(normalize_answer(current).split()):
                available[existing] = (mini, origin)

    if len(available) <= 1:
        direct = available[0][0] if available else NONE_ANSWER
        return None, {
            "luna_answer": luna,
            "mini_answer": mini,
            "direct_answer": direct,
            "reason": "semantic_agreement" if available else "both_abstained",
        }

    strength = _conflict_strength(observations)
    answers = tuple(answer for answer, _ in available) + (NONE_ANSWER,)
    rows = [
        _candidate_row(example, answer, origin, observations, strength)
        for answer, origin in available
    ]
    rows.append(_candidate_row(example, NONE_ANSWER, "none", observations, strength))

    target = str((example.supervision or {}).get("correct_answer", "")).strip()
    exact = [index for index, answer in enumerate(answers[:-1]) if exact_match(target, answer)]
    relaxed = [
        index for index, answer in enumerate(answers[:-1])
        if target_matches_candidate(target, answer)
    ]
    if exact:
        correct_index = max(exact, key=lambda index: token_f1(target, answers[index]))
        label_kind = "exact_candidate"
    elif relaxed:
        correct_index = max(relaxed, key=lambda index: token_f1(target, answers[index]))
        label_kind = "relaxed_candidate"
    else:
        correct_index = len(answers) - 1
        label_kind = "none_of_candidates"
    return ChoiceSet(
        example.question_id,
        answers,
        np.asarray(rows, dtype=float),
        correct_index=correct_index,
        references=(target,) if target else (),
        has_conflict=qacc_conflict_label(example),
    ), {
        "luna_answer": luna,
        "mini_answer": mini,
        "direct_answer": None,
        "reason": "model_disagreement",
        "label_kind": label_kind,
        "conflict_strength": strength,
    }


def conformal_probability_threshold(
    model: BayesianConditionalLogit,
    calibration: Sequence[ChoiceSet],
    *,
    alpha: float,
) -> tuple[float, dict]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = []
    for item in calibration:
        if item.correct_index is None:
            continue
        probability = float(model.posterior_probabilities(item)[item.correct_index])
        scores.append(1.0 - probability)
    if not scores:
        return 1.0, {"examples": 0, "alpha": alpha, "quantile": 0.0}
    ordered = sorted(scores)
    rank = min(len(ordered) - 1, math.ceil((len(ordered) + 1) * (1.0 - alpha)) - 1)
    quantile = float(ordered[max(rank, 0)])
    return 1.0 - quantile, {
        "examples": len(scores), "alpha": alpha, "quantile": quantile,
        "probability_threshold": 1.0 - quantile,
    }


def _metrics(rows: Sequence[Mapping], answer_key: str) -> dict:
    values = []
    for row in rows:
        answer = str(row.get(answer_key, NONE_ANSWER))
        references = row.get("references", [])
        abstained = is_no_answer(answer)
        exact = float(not abstained and any(exact_match(answer, value) for value in references))
        f1 = 0.0 if abstained else max(
            (token_f1(answer, value) for value in references), default=0.0
        )
        values.append((exact, f1, not abstained))
    covered = [item for item in values if item[2]]
    return {
        "examples": len(values),
        "answer_exact": float(np.mean([item[0] for item in values])) if values else 0.0,
        "answer_token_f1": float(np.mean([item[1] for item in values])) if values else 0.0,
        "coverage": len(covered) / len(values) if values else 0.0,
        "selective_exact": float(np.mean([item[0] for item in covered])) if covered else 0.0,
    }


def run_disagreement_cross_validation(
    examples: Sequence[PreparedExample],
    luna_predictions: Mapping[str, Mapping],
    mini_predictions: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]] | None = None,
    *,
    folds: int = 5,
    calibration_fraction: float = 0.25,
    alpha: float = 0.20,
    prior_scale: float = 0.75,
    posterior_samples: int = 500,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5)")
    extraction_cache = extractions or {}
    built: dict[str, tuple[ChoiceSet | None, dict, PreparedExample]] = {}
    order = {example.question_id: index for index, example in enumerate(examples)}
    for example in examples:
        choice, metadata = build_disagreement_choice(
            example,
            luna_predictions.get(example.question_id),
            mini_predictions.get(example.question_id),
            extraction_cache.get(example.question_id, []),
        )
        built[example.question_id] = (choice, metadata, example)

    assignments = {
        example.question_id: int(_hash_fraction(example.question_id, f"fold-{seed}") * folds)
        for example in examples
    }
    rows: list[dict] = []
    fold_reports = []
    for fold in range(folds):
        held_out = [item for item in examples if assignments[item.question_id] == fold]
        remaining = [item for item in examples if assignments[item.question_id] != fold]
        calibration_examples = [
            item for item in remaining
            if _hash_fraction(item.question_id, f"cal-{seed}-{fold}") < calibration_fraction
        ]
        calibration_ids = {item.question_id for item in calibration_examples}
        fit_choices = [
            built[item.question_id][0] for item in remaining
            if item.question_id not in calibration_ids and built[item.question_id][0] is not None
        ]
        calibration_choices = [
            built[item.question_id][0] for item in calibration_examples
            if built[item.question_id][0] is not None
        ]
        if len(fit_choices) < 5:
            raise ValueError(f"Fold {fold} has too few disagreement examples to fit")
        model = BayesianConditionalLogit(
            DISAGREEMENT_FEATURE_NAMES,
            prior_scale=prior_scale,
            posterior_samples=posterior_samples,
            label_smoothing=0.02,
            seed=seed + fold,
        )
        fit = model.fit(fit_choices)
        temperature = model.calibrate_temperature(calibration_choices)
        threshold, calibration_report = conformal_probability_threshold(
            model, calibration_choices, alpha=alpha
        )
        fold_reports.append({
            "fold": fold,
            "held_out": len(held_out),
            "fit_disagreements": len(fit_choices),
            "calibration_disagreements": len(calibration_choices),
            "fit": asdict(fit),
            "temperature": temperature,
            "conformal": calibration_report,
        })

        for example in held_out:
            choice, metadata, _ = built[example.question_id]
            references = [str((example.supervision or {}).get("correct_answer", ""))]
            luna = metadata["luna_answer"]
            mini = metadata["mini_answer"]
            if choice is None:
                map_answer = conformal_answer = metadata["direct_answer"]
                probabilities: list[float] = []
                prediction_set: list[str] = []
                decision = metadata["reason"]
            else:
                probability_array = model.posterior_probabilities(choice)
                probabilities = [float(value) for value in probability_array]
                best = int(np.argmax(probability_array))
                map_answer = choice.answers[best]
                selected = [
                    index for index, probability in enumerate(probability_array)
                    if float(probability) >= threshold
                ]
                prediction_set = [choice.answers[index] for index in selected]
                if len(selected) == 1 and not is_no_answer(choice.answers[selected[0]]):
                    conformal_answer = choice.answers[selected[0]]
                    decision = "conformal_singleton"
                else:
                    conformal_answer = NONE_ANSWER
                    decision = "conformal_abstain"
            rows.append({
                "question_id": example.question_id,
                "fold": fold,
                "gold_conflict": qacc_conflict_label(example),
                "references": references,
                "luna_answer": luna,
                "mini_answer": mini,
                "map_answer": map_answer,
                "conformal_answer": conformal_answer,
                "decision": decision,
                "candidate_answers": list(choice.answers) if choice else [],
                "correct_index": choice.correct_index if choice else None,
                "label_kind": metadata.get("label_kind"),
                "conflict_strength": metadata.get("conflict_strength", 0.0),
                "probabilities": probabilities,
                "prediction_set": prediction_set,
                "conformal_probability_threshold": threshold if choice else None,
            })

    rows.sort(key=lambda item: order[item["question_id"]])
    disagreement = [row for row in rows if row["candidate_answers"]]
    conflict = [row for row in rows if row["gold_conflict"]]
    nonconflict = [row for row in rows if not row["gold_conflict"]]
    summary = {
        "protocol": "dev_out_of_fold_pilot_not_frozen_test",
        "protocol_version": DISAGREEMENT_PROTOCOL_VERSION,
        "examples": len(rows),
        "routing": {
            "semantic_agreement": len(rows) - len(disagreement),
            "model_disagreement": len(disagreement),
            "disagreement_rate": len(disagreement) / len(rows) if rows else 0.0,
            "candidate_recall_exact": sum(
                row.get("label_kind") == "exact_candidate" for row in disagreement
            ) / len(disagreement) if disagreement else 0.0,
            "candidate_recall_relaxed": sum(
                row.get("label_kind") in {"exact_candidate", "relaxed_candidate"}
                for row in disagreement
            ) / len(disagreement) if disagreement else 0.0,
        },
        "overall": {
            key: _metrics(rows, key)
            for key in ("luna_answer", "mini_answer", "map_answer", "conformal_answer")
        },
        "conflict_only": {
            key: _metrics(conflict, key)
            for key in ("luna_answer", "mini_answer", "map_answer", "conformal_answer")
        },
        "nonconflict_only": {
            key: _metrics(nonconflict, key)
            for key in ("luna_answer", "mini_answer", "map_answer", "conformal_answer")
        },
        "disagreement_only": {
            key: _metrics(disagreement, key)
            for key in ("luna_answer", "mini_answer", "map_answer", "conformal_answer")
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


__all__ = [
    "DISAGREEMENT_FEATURE_NAMES",
    "DISAGREEMENT_PROTOCOL_VERSION",
    "build_disagreement_choice",
    "conformal_probability_threshold",
    "run_disagreement_cross_validation",
]
