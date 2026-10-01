from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .bayes import BayesianConditionalLogit
from .features import FEATURE_NAMES
from .normalization import exact_match, token_f1
from .pipeline import build_choice_sets
from .schemas import ChoiceSet, EvidenceObservation, PreparedExample


ProbabilityFunction = Callable[[ChoiceSet], np.ndarray]

SUPPORT_FEATURES = {"log_support", "support_share"}
EVIDENCE_QUALITY_FEATURES = {
    "mean_directness",
    "max_entailment",
    "mean_entailment",
    "mean_answer_type_match",
    "mean_temporal_fit",
}
SOURCE_CATEGORY_FEATURES = {
    "official_source_share",
    "reference_source_share",
    "news_source_share",
    "social_source_share",
}


def _slice_choice_sets(
    choice_sets: Sequence[ChoiceSet],
    feature_names: Sequence[str],
) -> list[ChoiceSet]:
    indices = [FEATURE_NAMES.index(name) for name in feature_names]
    return [
        ChoiceSet(
            question_id=item.question_id,
            answers=item.answers,
            features=item.features[:, indices],
            correct_index=item.correct_index,
            references=item.references,
            has_conflict=item.has_conflict,
        )
        for item in choice_sets
    ]


def _support_probabilities(item: ChoiceSet) -> np.ndarray:
    # support_share is normalized across candidates by construction.
    values = np.asarray(item.features[:, FEATURE_NAMES.index("support_share")], dtype=float)
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(values), 1.0 / len(values))


def _score_method(
    choice_sets: Sequence[ChoiceSet],
    *,
    expected_examples: int,
    probabilities: ProbabilityFunction,
    abstention_threshold: float,
    gold_answers: Mapping[str, str],
) -> dict:
    mapped_rows: list[tuple[np.ndarray, int, bool, bool]] = []
    covered = 0
    correct_covered = 0
    correct_covered_mapped = 0
    covered_mapped = 0
    answer_exact_total = 0.0
    answer_f1_total = 0.0
    answer_exact_covered = 0.0
    answer_f1_covered = 0.0

    for item in choice_sets:
        values = np.asarray(probabilities(item), dtype=float)
        if values.shape != (len(item.answers),):
            raise ValueError("Probability function returned the wrong shape")
        prediction = int(np.argmax(values))
        confidence = float(values[prediction])
        abstained = confidence < abstention_threshold
        correct = item.correct_index is not None and prediction == item.correct_index
        if not abstained:
            covered += 1
            correct_covered += int(correct)
            predicted_answer = item.answers[prediction]
            gold_answer = gold_answers.get(item.question_id, "")
            answer_exact = float(exact_match(predicted_answer, gold_answer))
            answer_f1 = token_f1(predicted_answer, gold_answer)
            answer_exact_total += answer_exact
            answer_f1_total += answer_f1
            answer_exact_covered += answer_exact
            answer_f1_covered += answer_f1
            if item.correct_index is not None:
                covered_mapped += 1
                correct_covered_mapped += int(correct)
        if item.correct_index is not None:
            mapped_rows.append((values, int(item.correct_index), correct, len(item.answers) >= 2))

    multicandidate = [row for row in mapped_rows if row[3]]
    nll = [
        -float(np.log(max(values[target], 1e-12)))
        for values, target, _, _ in mapped_rows
    ]
    brier = []
    for values, target, _, _ in mapped_rows:
        gold = np.zeros(len(values), dtype=float)
        gold[target] = 1.0
        brier.append(float(np.sum((values - gold) ** 2)))

    return {
        "examples": expected_examples,
        "candidate_sets": len(choice_sets),
        "candidate_recall": len(mapped_rows) / expected_examples if expected_examples else 0.0,
        "answer_exact": answer_exact_total / expected_examples if expected_examples else 0.0,
        "answer_token_f1": answer_f1_total / expected_examples if expected_examples else 0.0,
        "selective_answer_exact": answer_exact_covered / covered if covered else 0.0,
        "selective_answer_token_f1": answer_f1_covered / covered if covered else 0.0,
        "choice_accuracy": float(np.mean([row[2] for row in mapped_rows])) if mapped_rows else 0.0,
        "multicandidate_choice_accuracy": (
            float(np.mean([row[2] for row in multicandidate])) if multicandidate else 0.0
        ),
        "nll": float(np.mean(nll)) if nll else None,
        "multiclass_brier": float(np.mean(brier)) if brier else None,
        "coverage": covered / expected_examples if expected_examples else 0.0,
        "selective_accuracy_end_to_end": correct_covered / covered if covered else 0.0,
        "selective_accuracy_given_candidate_recall": (
            correct_covered_mapped / covered_mapped if covered_mapped else 0.0
        ),
        "abstention_threshold": abstention_threshold,
    }


def _fit_variant(
    train_sets: Sequence[ChoiceSet],
    dev_sets: Sequence[ChoiceSet],
    feature_names: Sequence[str],
    *,
    prior_scale: float,
    posterior_samples: int,
    label_smoothing: float,
    minimum_coverage: float,
    seed: int,
    calibrate_temperature: bool = True,
    probability_method: str = "posterior",
    select_threshold: bool = True,
) -> tuple[BayesianConditionalLogit, dict]:
    train_sliced = _slice_choice_sets(train_sets, feature_names)
    dev_sliced = _slice_choice_sets(dev_sets, feature_names)
    model = BayesianConditionalLogit(
        feature_names,
        prior_scale=prior_scale,
        posterior_samples=posterior_samples,
        label_smoothing=label_smoothing,
        seed=seed,
    )
    fit = model.fit(train_sliced)
    if calibrate_temperature:
        model.calibrate_temperature(
            dev_sliced,
            probability_method=probability_method,
        )
    if select_threshold:
        model.select_abstention_threshold(
            dev_sliced,
            minimum_coverage=minimum_coverage,
            probability_method=probability_method,
        )
    else:
        model.abstention_threshold = 0.0
    metadata = {
        "fit": asdict(fit),
        "feature_names": list(feature_names),
        "label_smoothing": label_smoothing,
        "temperature": model.temperature,
        "abstention_threshold": model.abstention_threshold,
        "probability_method": probability_method,
    }
    model.prediction_method = probability_method
    return model, metadata


def run_dev_experiments(
    train_examples: Sequence[PreparedExample],
    dev_examples: Sequence[PreparedExample],
    train_extractions: Mapping[str, Sequence[EvidenceObservation]],
    dev_extractions: Mapping[str, Sequence[EvidenceObservation]],
    *,
    answer_match_threshold: float = 0.8,
    near_duplicate_jaccard: float = 0.88,
    prior_scale: float = 2.0,
    posterior_samples: int = 1000,
    label_smoothing: float = 0.05,
    minimum_coverage: float = 0.7,
    seed: int = 42,
) -> dict:
    train_sets, train_report = build_choice_sets(
        train_examples,
        train_extractions,
        require_training_label=True,
        answer_match_threshold=answer_match_threshold,
        near_duplicate_jaccard=near_duplicate_jaccard,
    )
    dev_all, dev_all_report = build_choice_sets(
        dev_examples,
        dev_extractions,
        require_training_label=False,
        answer_match_threshold=answer_match_threshold,
        near_duplicate_jaccard=near_duplicate_jaccard,
    )
    dev_labelled = [item for item in dev_all if item.correct_index is not None]

    _, dev_raw_report = build_choice_sets(
        dev_examples,
        dev_extractions,
        require_training_label=False,
        answer_match_threshold=answer_match_threshold,
        near_duplicate_jaccard=near_duplicate_jaccard,
        dependency_correction=False,
    )
    dev_raw, _ = build_choice_sets(
        dev_examples,
        dev_extractions,
        require_training_label=False,
        answer_match_threshold=answer_match_threshold,
        near_duplicate_jaccard=near_duplicate_jaccard,
        dependency_correction=False,
    )

    methods: dict[str, dict] = {}
    dev_gold = {
        item.question_id: str((item.supervision or {}).get("correct_answer", ""))
        for item in dev_examples
    }
    methods["raw_passage_majority"] = {
        "role": "baseline",
        "metrics": _score_method(
            dev_raw,
            expected_examples=len(dev_examples),
            probabilities=_support_probabilities,
            abstention_threshold=0.0,
            gold_answers=dev_gold,
        ),
    }
    methods["independent_source_majority"] = {
        "role": "baseline",
        "metrics": _score_method(
            dev_all,
            expected_examples=len(dev_examples),
            probabilities=_support_probabilities,
            abstention_threshold=0.0,
            gold_answers=dev_gold,
        ),
    }

    model_specs = {
        "full_uacaa": {},
        "non_bayesian_conditional_logit": {"probability_method": "map"},
        "no_support_features": {
            "feature_names": [name for name in FEATURE_NAMES if name not in SUPPORT_FEATURES]
        },
        "no_evidence_quality_features": {
            "feature_names": [
                name for name in FEATURE_NAMES if name not in EVIDENCE_QUALITY_FEATURES
            ]
        },
        "no_source_category_features": {
            "feature_names": [
                name for name in FEATURE_NAMES if name not in SOURCE_CATEGORY_FEATURES
            ]
        },
        "no_temperature_calibration": {"calibrate_temperature": False},
        "no_abstention": {"select_threshold": False},
        "no_label_smoothing": {"label_smoothing": 0.0},
    }

    for name, overrides in model_specs.items():
        feature_names = tuple(overrides.get("feature_names", FEATURE_NAMES))
        probability_method = str(overrides.get("probability_method", "posterior"))
        model, metadata = _fit_variant(
            train_sets,
            dev_labelled,
            feature_names,
            prior_scale=prior_scale,
            posterior_samples=posterior_samples,
            label_smoothing=float(overrides.get("label_smoothing", label_smoothing)),
            minimum_coverage=minimum_coverage,
            seed=seed,
            calibrate_temperature=bool(overrides.get("calibrate_temperature", True)),
            probability_method=probability_method,
            select_threshold=bool(overrides.get("select_threshold", True)),
        )
        dev_sliced = _slice_choice_sets(dev_all, feature_names)
        probability_function = (
            model.map_probabilities
            if probability_method == "map"
            else model.posterior_probabilities
        )
        methods[name] = {
            "role": "baseline" if name == "non_bayesian_conditional_logit" else (
                "primary" if name == "full_uacaa" else "ablation"
            ),
            "model": metadata,
            "metrics": _score_method(
                dev_sliced,
                expected_examples=len(dev_examples),
                probabilities=probability_function,
                abstention_threshold=model.abstention_threshold,
                gold_answers=dev_gold,
            ),
        }

    raw_train, _ = build_choice_sets(
        train_examples,
        train_extractions,
        require_training_label=True,
        answer_match_threshold=answer_match_threshold,
        near_duplicate_jaccard=near_duplicate_jaccard,
        dependency_correction=False,
    )
    raw_dev_labelled = [item for item in dev_raw if item.correct_index is not None]
    raw_model, raw_metadata = _fit_variant(
        raw_train,
        raw_dev_labelled,
        FEATURE_NAMES,
        prior_scale=prior_scale,
        posterior_samples=posterior_samples,
        label_smoothing=label_smoothing,
        minimum_coverage=minimum_coverage,
        seed=seed,
    )
    methods["no_dependency_correction"] = {
        "role": "ablation",
        "model": raw_metadata,
        "metrics": _score_method(
            dev_raw,
            expected_examples=len(dev_examples),
            probabilities=raw_model.posterior_probabilities,
            abstention_threshold=raw_model.abstention_threshold,
            gold_answers=dev_gold,
        ),
    }
    raw_model.select_abstention_threshold(
        raw_dev_labelled,
        minimum_coverage=minimum_coverage,
        probability_method="map",
    )
    selected_metadata = {
        **raw_metadata,
        "abstention_threshold": raw_model.abstention_threshold,
        "probability_method": "map",
        "selection_note": (
            "Combined candidate motivated by the two predeclared dev ablations: "
            "point estimate and no dependency correction"
        ),
    }
    methods["selected_map_no_dependency"] = {
        "role": "selected_candidate",
        "model": selected_metadata,
        "metrics": _score_method(
            dev_raw,
            expected_examples=len(dev_examples),
            probabilities=raw_model.map_probabilities,
            abstention_threshold=raw_model.abstention_threshold,
            gold_answers=dev_gold,
        ),
    }

    return {
        "scope": "train/dev model development only; no official test references used",
        "train_build": asdict(train_report),
        "dev_build": asdict(dev_all_report),
        "raw_dev_build": asdict(dev_raw_report),
        "methods": methods,
        "pending_llm_baselines": ["direct_rag", "qacc_explain_then_answer"],
    }


def write_dev_experiments(result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["run_dev_experiments", "write_dev_experiments"]
