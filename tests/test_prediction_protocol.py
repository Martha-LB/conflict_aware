import json
from pathlib import Path

import numpy as np

from uacaa.evaluation import score_candidate_predictions, write_candidate_predictions
from uacaa.schemas import ChoiceSet


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_predictions_are_frozen_before_references_are_loaded(tmp_path: Path):
    prediction_path = tmp_path / "predictions.jsonl"
    choice_sets = [
        ChoiceSet(
            question_id="q1",
            answers=("Paris", "Lyon"),
            features=np.asarray([[1.0], [0.0]]),
        )
    ]

    report = write_candidate_predictions(
        choice_sets,
        expected_question_ids=("q1", "q2"),
        probabilities=lambda _: np.asarray([0.8, 0.2]),
        abstention_threshold=0.5,
        method="test-method",
        output_path=prediction_path,
    )

    assert report["examples"] == 2
    prediction_rows = _read_jsonl(prediction_path)
    assert all("references" not in row and "exact" not in row for row in prediction_rows)
    assert prediction_rows[0]["prediction"] == "Paris"
    assert prediction_rows[1]["failure_reason"] == "no_extracted_candidates"

    scored_path = tmp_path / "scored.jsonl"
    summary = score_candidate_predictions(
        prediction_path,
        {
            "q1": {"references": ["Paris"], "has_conflict": True},
            "q2": {"references": ["Rome"], "has_conflict": False},
        },
        output_path=scored_path,
    )
    assert summary["overall"]["examples"] == 2
    assert summary["overall"]["answer_exact"] == 0.5
    assert summary["overall"]["candidate_recall"] == 0.5
    assert summary["overall"]["coverage"] == 0.5


def test_scoring_rejects_mismatched_question_ids(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "candidate_answers": ["Paris"],
                "probabilities": [1.0],
                "prediction": "Paris",
                "abstained": False,
                "abstention_threshold": 0.0,
            }
        )
        + "\n"
    )
    try:
        score_candidate_predictions(
            path,
            {"q2": {"references": ["Rome"]}},
            output_path=tmp_path / "scored.jsonl",
        )
    except ValueError as error:
        assert "ID mismatch" in str(error)
    else:
        raise AssertionError("Expected scoring to reject mismatched IDs")
