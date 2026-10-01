import json
from pathlib import Path

import numpy as np

from uacaa.schemas import Context, EvidenceObservation, PreparedExample
from uacaa.two_stage import (
    BayesianBinaryLogit,
    TwoStageConflictModel,
    conflict_features,
    evaluate_two_stage_predictions,
    fit_two_stage_model,
    load_domain_metadata,
    qacc_conflict_label,
    rank_feature_matrix,
)


def _example(question_id: str, gold: str, *, conflict: bool) -> PreparedExample:
    candidates = [{"answer": gold, "context_indices": [0]}]
    if conflict:
        candidates.append({"answer": "Lyon", "context_indices": [1]})
    return PreparedExample(
        question_id=question_id,
        split="train",
        question="What was the capital in 2020?",
        contexts=[
            Context(0, "In 2020 Paris was the capital.", "service.gov"),
            Context(1, "In 2019 Lyon was the capital.", "social.example"),
        ],
        supervision={"correct_answer": gold, "candidates": candidates},
    )


def _generation(primary: str = "Lyon") -> dict:
    alternative = "Paris" if primary == "Lyon" else "Lyon"
    primary_index = 1 if primary == "Lyon" else 0
    alternative_index = 0 if primary == "Lyon" else 1
    return {
        "answer": primary,
        "evidence": [{"context_index": primary_index, "quote": f"In {2019 if primary_index else 2020} {primary} was the capital"}],
        "alternatives": [{
            "answer": alternative,
            "evidence": [{"context_index": alternative_index, "quote": f"In {2019 if alternative_index else 2020} {alternative} was the capital"}],
        }],
        "contradictions": [{"answer_a": primary, "answer_b": alternative, "context_indices": [0, 1]}],
        "evidence_sufficient": True,
        "confidence": "medium",
    }


def test_conflict_target_and_temporal_domain_features(tmp_path: Path):
    example = _example("q1", "Paris", conflict=True)
    assert qacc_conflict_label(example)
    values = conflict_features(_generation(), example)
    assert values.shape == (24,)
    metadata_path = tmp_path / "domains.json"
    metadata_path.write_text(json.dumps({"service.gov": {"authority": 0.9, "category": "official"}}))
    metadata = load_domain_metadata(metadata_path)
    answers, matrix = rank_feature_matrix(
        _generation(),
        example,
        [
            EvidenceObservation(0, "Paris", directness=1, entailment=1, answer_type_match=1, temporal_fit=1),
            EvidenceObservation(1, "Lyon", directness=1, entailment=1, answer_type_match=1, temporal_fit=0),
        ],
        metadata,
    )
    assert answers == ("Lyon", "Paris")
    assert matrix.shape == (2, 21)
    # The Paris evidence matches the requested year and has frozen metadata.
    assert matrix[1, 11] == 1.0
    assert matrix[1, 18] == 0.9
    assert matrix[1, 19] == 1.0


def test_two_stage_model_fits_routes_and_evaluates():
    examples = []
    generations = {}
    extractions = {}
    for index in range(12):
        conflict = index % 2 == 0
        gold = "Paris"
        example = _example(f"q{index}", gold, conflict=conflict)
        examples.append(example)
        generation = _generation("Lyon" if conflict else "Paris")
        if not conflict:
            generation["alternatives"] = []
            generation["contradictions"] = []
            generation["evidence"] = [{"context_index": 0, "quote": "In 2020 Paris was the capital"}]
        generations[example.question_id] = {"generation": generation}
        extractions[example.question_id] = [
            EvidenceObservation(0, "Paris", directness=1, entailment=1, answer_type_match=1, temporal_fit=1),
            EvidenceObservation(1, "Lyon", directness=0.7, entailment=0.8, answer_type_match=1, temporal_fit=0),
        ]
    model, report = fit_two_stage_model(
        examples, generations, extractions, folds=3, posterior_samples=100, seed=7
    )
    assert report["detector"]["fit"]["examples"] == 12
    assert report["ranker_fit"]["questions"] == 6
    record = model.resolve(examples[0], generations["q0"], extractions["q0"], mode="map")
    assert record["predicted_conflict"]
    assert record["decision"] == "conflict_ranker"
    predictions = {"q0": record}
    summary = evaluate_two_stage_predictions([examples[0]], predictions)
    assert summary["conflict_detection"]["tp"] == 1
    assert summary["two_stage_final"]["examples"] == 1


def test_binary_model_round_trip_payload():
    matrix = np.asarray([[0.0], [0.2], [0.8], [1.0]])
    labels = np.asarray([0, 0, 1, 1])
    model = BayesianBinaryLogit(("signal",))
    model.fit(matrix, labels)
    loaded = BayesianBinaryLogit.from_dict(model.to_dict())
    np.testing.assert_allclose(
        model.predict_probability(matrix), loaded.predict_probability(matrix)
    )
