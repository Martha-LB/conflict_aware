"""Tests for QACC parsing and test-label isolation."""

import json
from pathlib import Path

import pytest

from uacaa.extraction import annotated_observations
from uacaa.protocol import (
    prepare_protocol,
    read_evaluation_metadata,
    read_prepared,
    read_references,
)
from uacaa.qacc import (
    audit_rows,
    evaluation_references,
    load_ambigqa_references,
    prepare_example,
    qacc_conflict_label,
)


def _row(split="train", *, fourth="B", correct="Paris"):
    """Create one row in the unusual schema used by QACC."""

    return {
        "annotation_task_id": 1,
        "split": split,
        "question": "What is the capital of France?",
        "contexts": ["Paris is the capital.", "Some unrelated text."],
        "sources": ["example.gov", "example.com"],
        "firstAnswer": "Paris",
        "firstContext": "['A']",
        "secondAnswerExist": "B",
        "secondAnswer": float("nan"),
        "secondContext": float("nan"),
        "thirdAnswerExist": float("nan"),
        "thirdAnswer": float("nan"),
        "thirdContext": float("nan"),
        "fourthAnswerExist": fourth,
        "correctAnswer": correct,
        "reasons": "['B']",
        "explanation": "The first context states the answer.",
        "ambigqa_answer": ["Paris"] if split == "test" else None,
    }


def test_audit_distinguishes_unassigned_context_from_missing_candidate():
    # An irrelevant context is valid data, not a missing answer field.
    report = audit_rows([_row()])
    assert report["splits"]["train"]["contexts"] == 2
    assert report["splits"]["train"]["attributed_contexts"] == 1
    assert report["splits"]["train"]["attribution_coverage"] == 0.5


def test_prepare_protocol_preserves_official_test_and_filters_strict(tmp_path: Path):
    # Test input must not contain its answer.
    raw = tmp_path / "qacc.json"
    raw.write_text(json.dumps([_row(), _row("train", fourth="A"), _row("test")]))
    output = tmp_path / "derived"
    manifest = prepare_protocol(raw, output, revision="abc123")
    assert manifest["upstream_revision"] == "abc123"
    assert len(read_prepared(output / "train.weak.jsonl")) == 2
    assert len(read_prepared(output / "train.strict.jsonl")) == 1
    test_inputs = read_prepared(output / "test.official.inputs.jsonl")
    assert len(test_inputs) == 1
    assert test_inputs[0].supervision is None
    assert test_inputs[0].evaluation is None
    assert read_references(output / "test.official.references.jsonl")[test_inputs[0].question_id] == ("Paris",)
    # Test labels are still available to the separate evaluator.
    metadata = read_evaluation_metadata(output / "test.official.references.jsonl")
    assert metadata[test_inputs[0].question_id]["has_conflict"] is False


def test_prepare_example_preserves_author_conflict_label_directly():
    row = _row()
    row["secondAnswerExist"] = "A"
    # Deliberately leave secondAnswer missing. The target is the authors'
    # judgment, not a value reconstructed from the answer fields.
    example = prepare_example(row, 0)
    assert example.supervision["has_conflict"] is True
    assert qacc_conflict_label(example) is True


def test_prepare_protocol_restores_ambigqa_multi_references(tmp_path: Path):
    # Only singleAnswer aliases belong to the original QACC question.
    raw = tmp_path / "qacc.json"
    raw.write_text(json.dumps([_row(), _row("dev"), _row("test")]))
    ambigqa = tmp_path / "train_light.json"
    ambigqa.write_text(
        json.dumps(
            [
                {
                    "question": "What is the capital of France?",
                    "annotations": [
                        {
                            "type": "singleAnswer",
                            "answer": ["Paris", "City of Paris"],
                        },
                        {
                            "type": "multipleQAs",
                            "qaPairs": [
                                {"question": "Which Paris?", "answer": ["Paris, Texas"]}
                            ],
                        },
                    ],
                }
            ]
        )
    )
    index = load_ambigqa_references([ambigqa])
    assert index["what is the capital of france"] == ("Paris", "City of Paris")

    output = tmp_path / "derived"
    manifest = prepare_protocol(raw, output, ambigqa_paths=[ambigqa])
    train = read_prepared(output / "train.strict.jsonl")[0]
    assert evaluation_references(train) == ["Paris", "City of Paris"]
    assert manifest["protocol_version"] == "1.2"
    assert manifest["reference_coverage"]["train"]["restored_from_ambigqa"] == 1


def test_oracle_attribution_is_forbidden_on_test():
    # Gold context attribution is never a legal test-time feature.
    example = prepare_example(_row("test"), 0)
    with pytest.raises(ValueError):
        annotated_observations(example)
