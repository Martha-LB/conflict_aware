from pathlib import Path

from uacaa.disagreement import (
    DISAGREEMENT_FEATURE_NAMES,
    build_disagreement_choice,
    run_disagreement_cross_validation,
)
from uacaa.schemas import Context, EvidenceObservation, PreparedExample


def _example(index: int, gold: str, conflict: bool = False) -> PreparedExample:
    candidates = [{"answer": gold, "context_indices": [0]}]
    if conflict:
        candidates.append({"answer": "wrong", "context_indices": [1]})
    return PreparedExample(
        question_id=f"q{index}",
        split="dev",
        question="Which answer is supported?",
        contexts=[
            Context(0, f"The supported answer is {gold}.", "agency.gov"),
            Context(1, "A different page says wrong.", "example.com"),
        ],
        supervision={"correct_answer": gold, "candidates": candidates},
    )


def test_disagreement_choice_has_none_candidate_and_evidence_features():
    example = _example(1, "Paris", conflict=True)
    choice, metadata = build_disagreement_choice(
        example,
        {"answer": "Lyon"},
        {"answer": "Paris"},
        [EvidenceObservation(0, "Paris", entailment=0.95, directness=0.9)],
    )
    assert choice is not None
    assert choice.answers == ("Lyon", "Paris", "unknown")
    assert choice.correct_index == 1
    assert choice.features.shape == (3, len(DISAGREEMENT_FEATURE_NAMES))
    assert metadata["label_kind"] == "exact_candidate"


def test_semantic_surface_agreement_bypasses_ranker():
    example = _example(2, "October")
    choice, metadata = build_disagreement_choice(
        example, {"answer": "October"}, {"answer": "Every October."}
    )
    assert choice is None
    assert metadata["direct_answer"] == "October"
    assert metadata["reason"] == "semantic_agreement"


def test_out_of_fold_pilot_writes_complete_report(tmp_path: Path):
    examples = []
    luna = {}
    mini = {}
    extractions = {}
    for index in range(30):
        gold = "Paris" if index % 2 else "Lyon"
        example = _example(index, gold, conflict=index % 3 == 0)
        examples.append(example)
        luna[example.question_id] = {"answer": "Lyon"}
        mini[example.question_id] = {"answer": "Paris"}
        extractions[example.question_id] = [
            EvidenceObservation(0, gold, entailment=0.9, directness=0.9)
        ]
    output = tmp_path / "pilot.jsonl"
    summary = run_disagreement_cross_validation(
        examples, luna, mini, extractions,
        folds=3, calibration_fraction=0.2, posterior_samples=30,
        output_path=output,
    )
    assert summary["examples"] == 30
    assert summary["routing"]["model_disagreement"] == 30
    assert summary["protocol"] == "dev_out_of_fold_pilot_not_frozen_test"
    assert output.exists()
    assert output.with_suffix(".summary.json").exists()
