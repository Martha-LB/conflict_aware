from pathlib import Path

from uacaa.schemas import Context, EvidenceObservation, PreparedExample
from uacaa.uncertainty_routing import (
    UQ_FEATURE_NAMES,
    build_uq_features,
    evaluate_frozen_uncertainty_routing_predictions,
    run_uncertainty_routing_cross_validation,
    run_uncertainty_routing_train_dev,
    write_frozen_uncertainty_routing_predictions,
)


def _example(index: int, gold: str, conflict: bool = False) -> PreparedExample:
    candidates = [{"answer": gold, "context_indices": [0]}]
    if conflict:
        candidates.append({"answer": "wrong", "context_indices": [1]})
    return PreparedExample(
        question_id=f"q{index}",
        split="dev",
        question="In which year was the supported event?",
        contexts=[
            Context(0, f"The answer in 2020 was {gold}.", "agency.gov"),
            Context(1, "A copied page from 1990 says wrong.", "example.com"),
        ],
        supervision={"correct_answer": gold, "candidates": candidates},
    )


def test_uq_features_include_conflict_source_and_time_signals():
    example = _example(1, "Paris", conflict=True)
    features, metadata = build_uq_features(
        example,
        {"answer": "Lyon"},
        {"answer": "Paris"},
        [
            EvidenceObservation(0, "Paris", entailment=0.95, temporal_fit=0.9),
            EvidenceObservation(1, "Lyon", entailment=0.8, temporal_fit=0.4),
        ],
    )
    assert features.shape == (len(UQ_FEATURE_NAMES),)
    assert metadata["model_disagreement"] is True
    assert metadata["evidence_semantic_entropy"] > 0
    assert features[UQ_FEATURE_NAMES.index("context_year_dispersion")] > 0
    assert features[UQ_FEATURE_NAMES.index("official_domain_share")] > 0


def test_uq_cross_validation_writes_oof_report(tmp_path: Path):
    examples = []
    luna = {}
    mini = {}
    extractions = {}
    for index in range(60):
        gold = "Paris" if index % 3 else "Lyon"
        example = _example(index, gold, conflict=index % 4 == 0)
        examples.append(example)
        luna[example.question_id] = {"answer": "Lyon" if index % 5 else gold}
        mini[example.question_id] = {"answer": "Paris" if index % 7 else gold}
        extractions[example.question_id] = [
            EvidenceObservation(0, gold, entailment=0.9, directness=0.9)
        ]
    output = tmp_path / "uq.jsonl"
    summary = run_uncertainty_routing_cross_validation(
        examples,
        luna,
        mini,
        extractions,
        folds=3,
        calibration_fraction=0.25,
        seed=7,
        output_path=output,
    )
    assert summary["examples"] == 60
    assert summary["protocol"] == "labelled_dev_out_of_fold_uq_pilot_not_frozen_test"
    assert "auroc_error_detection" in summary["risk_quality"]
    assert "conservative_uq_route" in summary["overall"]
    assert output.exists()
    assert output.with_suffix(".summary.json").exists()


def test_train_dev_router_is_symmetric_and_writes_model(tmp_path: Path):
    train = []
    dev = []
    primary = {}
    secondary = {}
    extractions = {}
    for index in range(80):
        gold = "Paris" if index % 3 else "Lyon"
        example = _example(index, gold, conflict=index % 4 == 0)
        if index >= 50:
            example.split = "dev"
            dev.append(example)
        else:
            example.split = "train"
            train.append(example)
        primary[example.question_id] = {"answer": "Lyon" if index % 5 else gold}
        secondary[example.question_id] = {"answer": "Paris" if index % 7 else gold}
        extractions[example.question_id] = [
            EvidenceObservation(0, gold, entailment=0.9, directness=0.9)
        ]
    output = tmp_path / "train-dev.jsonl"
    summary = run_uncertainty_routing_train_dev(
        train,
        dev,
        primary,
        secondary,
        primary,
        secondary,
        extractions,
        extractions,
        primary_name="primary-model",
        secondary_name="secondary-model",
        output_path=output,
    )
    assert summary["protocol"] == "train_fit_dev_calibration_and_selection_not_test"
    assert summary["train_examples"] == 50
    assert summary["dev_examples"] == 30
    assert summary["configuration"]["primary_name"] == "primary-model"
    assert output.exists()
    assert output.with_suffix(".summary.json").exists()
    assert output.with_suffix(".model.json").exists()

    test_examples = []
    test_primary = {}
    test_secondary = {}
    evaluation = {}
    for index in range(100, 112):
        gold = "Paris" if index % 2 else "Lyon"
        example = _example(index, gold, conflict=index % 4 == 0)
        example.split = "test"
        example.supervision = None
        test_examples.append(example)
        test_primary[example.question_id] = {"answer": "Lyon"}
        test_secondary[example.question_id] = {"answer": "Paris"}
        evaluation[example.question_id] = {
            "references": [gold],
            "has_conflict": index % 4 == 0,
        }
    prediction_path = tmp_path / "frozen.jsonl"
    import json

    model = json.loads(output.with_suffix(".model.json").read_text())
    progress = write_frozen_uncertainty_routing_predictions(
        test_examples,
        test_primary,
        test_secondary,
        extractions,
        model,
        prediction_path,
    )
    assert progress["examples"] == 12
    frozen = {
        row["question_id"]: row
        for row in map(json.loads, prediction_path.read_text().splitlines())
    }
    scored_path = tmp_path / "scored.jsonl"
    scored = evaluate_frozen_uncertainty_routing_predictions(
        test_examples,
        frozen,
        evaluation,
        output_path=scored_path,
    )
    assert scored["protocol"] == "frozen_official_test_evaluation"
    assert scored["examples"] == 12
    assert scored_path.exists()
    assert scored_path.with_suffix(".summary.json").exists()
