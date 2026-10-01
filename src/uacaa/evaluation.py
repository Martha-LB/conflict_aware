from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .bayes import BayesianConditionalLogit
from .normalization import exact_match, target_matches_candidate, token_f1
from .schemas import ChoiceSet


ProbabilityFunction = Callable[[ChoiceSet], np.ndarray]


def _summarize(rows: Sequence[dict], threshold: float) -> dict:
    covered = [row for row in rows if not row["abstained"]]
    brier = [row["multiclass_brier"] for row in rows if row["multiclass_brier"] is not None]
    count = len(rows)
    return {
        "examples": count,
        "answer_exact": float(np.mean([row["exact"] for row in rows])) if rows else 0.0,
        "answer_token_f1": float(np.mean([row["token_f1"] for row in rows])) if rows else 0.0,
        "candidate_recall": float(np.mean([row["candidate_recalled"] for row in rows])) if rows else 0.0,
        "coverage": len(covered) / count if count else 0.0,
        "selective_exact": float(np.mean([row["exact"] for row in covered])) if covered else 0.0,
        "multiclass_brier": float(np.mean(brier)) if brier else None,
        "abstention_threshold": threshold,
    }


def write_candidate_predictions(
    choice_sets: Sequence[ChoiceSet],
    *,
    expected_question_ids: Sequence[str],
    probabilities: ProbabilityFunction,
    abstention_threshold: float,
    method: str,
    output_path: Path,
) -> dict:
    """Freeze candidate predictions without loading or receiving references."""
    rows: list[dict] = []
    built_ids: set[str] = set()
    for choice_set in choice_sets:
        built_ids.add(choice_set.question_id)
        values = np.asarray(probabilities(choice_set), dtype=float)
        if values.shape != (len(choice_set.answers),):
            raise ValueError("Probability function returned the wrong shape")
        index = int(np.argmax(values))
        confidence = float(values[index])
        abstained = confidence < abstention_threshold
        rows.append(
            {
                "question_id": choice_set.question_id,
                "method": method,
                "candidate_answers": list(choice_set.answers),
                "probabilities": values.tolist(),
                "prediction": None if abstained else choice_set.answers[index],
                "confidence": confidence,
                "abstained": abstained,
                "abstention_threshold": abstention_threshold,
                "failure_reason": None,
            }
        )
    for question_id in expected_question_ids:
        if question_id not in built_ids:
            rows.append(
                {
                    "question_id": question_id,
                    "method": method,
                    "candidate_answers": [],
                    "probabilities": [],
                    "prediction": None,
                    "confidence": 0.0,
                    "abstained": True,
                    "abstention_threshold": abstention_threshold,
                    "failure_reason": "no_extracted_candidates",
                }
            )
    order = {question_id: index for index, question_id in enumerate(expected_question_ids)}
    rows.sort(key=lambda row: order[row["question_id"]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "method": method,
        "examples": len(rows),
        "candidate_sets": len(built_ids),
        "abstained": sum(bool(row["abstained"]) for row in rows),
        "output": str(output_path),
    }


def score_candidate_predictions(
    prediction_path: Path,
    evaluation_metadata: Mapping[str, Mapping],
    *,
    output_path: Path,
) -> dict:
    """Score an already frozen prediction file against separately stored references."""
    predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_ids = [str(row["question_id"]) for row in predictions]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Prediction file contains duplicate question IDs")
    expected_ids = set(evaluation_metadata)
    if set(actual_ids) != expected_ids:
        missing = sorted(expected_ids - set(actual_ids))
        extra = sorted(set(actual_ids) - expected_ids)
        raise ValueError(
            f"Prediction/reference ID mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    rows: list[dict] = []
    for prediction_row in predictions:
        question_id = str(prediction_row["question_id"])
        metadata = evaluation_metadata[question_id]
        references = list(metadata.get("references", []))
        candidates = list(prediction_row.get("candidate_answers", []))
        values = np.asarray(prediction_row.get("probabilities", []), dtype=float)
        predicted = prediction_row.get("prediction")
        exact = float(
            predicted is not None
            and any(exact_match(predicted, reference) for reference in references)
        )
        f1 = (
            max((token_f1(predicted, reference) for reference in references), default=0.0)
            if predicted is not None
            else 0.0
        )
        gold_indices = [
            index
            for index, candidate in enumerate(candidates)
            if any(target_matches_candidate(reference, candidate) for reference in references)
        ]
        brier: float | None = None
        if gold_indices and len(values) == len(candidates):
            target = np.zeros(len(candidates), dtype=float)
            target[gold_indices] = 1.0 / len(gold_indices)
            brier = float(np.sum((values - target) ** 2))
        rows.append(
            {
                **prediction_row,
                "references": references,
                "has_conflict": metadata.get("has_conflict"),
                "exact": exact,
                "token_f1": f1,
                "candidate_recalled": float(bool(gold_indices)),
                "multiclass_brier": brier,
            }
        )

    thresholds = {float(row.get("abstention_threshold", 0.0)) for row in predictions}
    threshold = next(iter(thresholds)) if len(thresholds) == 1 else 0.0
    summary = {
        "overall": _summarize(rows, threshold),
        "multicandidate_only": _summarize(
            [row for row in rows if len(row["candidate_answers"]) >= 2], threshold
        ),
        "conflict_only": _summarize(
            [row for row in rows if row["has_conflict"] is True], threshold
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def evaluate_choice_sets(
    model: BayesianConditionalLogit,
    choice_sets: Sequence[ChoiceSet],
    *,
    output_path: Path | None = None,
    expected_question_ids: Sequence[str] | None = None,
    evaluation_metadata: Mapping[str, Mapping] | None = None,
) -> dict:
    """Score every expected example, including candidate-extraction failures."""
    rows: list[dict] = []
    built_ids: set[str] = set()

    for choice_set in choice_sets:
        built_ids.add(choice_set.question_id)
        probabilities = model.probabilities(choice_set)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        abstained = confidence < model.abstention_threshold
        predicted = None if abstained else choice_set.answers[index]
        references = list(choice_set.references)
        exact = float(
            predicted is not None
            and any(exact_match(predicted, reference) for reference in references)
        )
        f1 = (
            max((token_f1(predicted, reference) for reference in references), default=0.0)
            if predicted is not None
            else 0.0
        )
        recalled = float(
            any(
                target_matches_candidate(reference, candidate)
                for candidate in choice_set.answers
                for reference in references
            )
        )
        gold_indices = [
            candidate_index
            for candidate_index, candidate in enumerate(choice_set.answers)
            if any(target_matches_candidate(reference, candidate) for reference in references)
        ]
        brier: float | None = None
        if gold_indices:
            target = np.zeros(len(choice_set.answers), dtype=float)
            target[gold_indices] = 1.0 / len(gold_indices)
            # Standard multiclass Brier score sums across mutually exclusive
            # classes; averaging over classes would make questions with more
            # candidates look artificially better calibrated.
            brier = float(np.sum((probabilities - target) ** 2))
        rows.append(
            {
                "question_id": choice_set.question_id,
                "candidate_answers": choice_set.answers,
                "probabilities": probabilities.tolist(),
                "prediction": predicted,
                "confidence": confidence,
                "abstained": abstained,
                "references": references,
                "has_conflict": choice_set.has_conflict,
                "exact": exact,
                "token_f1": f1,
                "candidate_recalled": recalled,
                "multiclass_brier": brier,
                "failure_reason": None,
            }
        )

    metadata = evaluation_metadata or {}
    for question_id in expected_question_ids or ():
        if question_id in built_ids:
            continue
        item = metadata.get(question_id, {})
        rows.append(
            {
                "question_id": question_id,
                "candidate_answers": [],
                "probabilities": [],
                "prediction": None,
                "confidence": 0.0,
                "abstained": True,
                "references": list(item.get("references", [])),
                "has_conflict": item.get("has_conflict"),
                "exact": 0.0,
                "token_f1": 0.0,
                "candidate_recalled": 0.0,
                "multiclass_brier": None,
                "failure_reason": "no_extracted_candidates",
            }
        )

    summary = {
        "overall": _summarize(rows, model.abstention_threshold),
        "multicandidate_only": _summarize(
            [row for row in rows if len(row["candidate_answers"]) >= 2],
            model.abstention_threshold,
        ),
        "conflict_only": _summarize(
            [row for row in rows if row["has_conflict"] is True],
            model.abstention_threshold,
        ),
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
