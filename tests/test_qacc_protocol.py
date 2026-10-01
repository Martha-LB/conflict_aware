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
from uacaa.qacc import audit_rows, prepare_example


def _row(split="train", *, fourth="B", correct="Paris"):
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
    report = audit_rows([_row()])
    assert report["splits"]["train"]["contexts"] == 2
    assert report["splits"]["train"]["attributed_contexts"] == 1
    assert report["splits"]["train"]["attribution_coverage"] == 0.5


def test_prepare_protocol_preserves_official_test_and_filters_strict(tmp_path: Path):
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
    metadata = read_evaluation_metadata(output / "test.official.references.jsonl")
    assert metadata[test_inputs[0].question_id]["has_conflict"] is False


def test_oracle_attribution_is_forbidden_on_test():
    example = prepare_example(_row("test"), 0)
    with pytest.raises(ValueError):
        annotated_observations(example)
