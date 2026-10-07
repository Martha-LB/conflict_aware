"""Tests for baseline prompts, cache, and scores."""

from pathlib import Path

from conflict_maxent.answer_baselines import (
    DIRECT_RAG_SYSTEM_PROMPT,
    NO_CONTEXT_SYSTEM_PROMPT,
    QACC_EXPLAIN_SYSTEM_PROMPT,
    WEIGHTED_RAG_SYSTEM_PROMPT,
    evaluate_answer_cache,
    read_answer_cache,
    write_answer_cache,
    weighted_answer_prompt,
)
from conflict_maxent.schemas import Context, PreparedExample


class _Predictor:
    """A fake predictor. It avoids real API calls."""
    def __init__(self):
        self.calls = 0

    def predict(self, example):
        self.calls += 1
        return {"answer": "Paris", "explanation": ""}


def _examples():
    """Create two small labelled questions for local tests."""

    return [
        PreparedExample(
            str(index),
            "dev",
            "Capital?",
            [Context(0, "Paris is the capital.")],
            supervision={"correct_answer": "Paris"},
        )
        for index in range(2)
    ]


def test_answer_baseline_cache_resumes_and_evaluates(tmp_path: Path):
    # First write one answer, then check that resume calls the model only once.
    output = tmp_path / "answers.jsonl"
    first = _Predictor()
    assert write_answer_cache(_examples()[:1], first, output)["extracted"] == 1
    second = _Predictor()
    progress = write_answer_cache(_examples(), second, output)
    assert progress == {"requested": 2, "reused": 1, "extracted": 1}
    assert second.calls == 1
    summary = evaluate_answer_cache(_examples(), read_answer_cache(output))
    assert summary["answer_exact"] == 1.0
    assert summary["coverage"] == 1.0


def test_answer_cache_accepts_only_explicit_seed_records(tmp_path: Path):
    # An identical-prompt caller may seed one answer; the other still calls the model.
    output = tmp_path / "seeded.jsonl"
    predictor = _Predictor()
    progress = write_answer_cache(
        _examples(),
        predictor,
        output,
        seed_records={"0": {"question_id": "0", "answer": "Paris"}},
    )
    assert progress == {"requested": 2, "reused": 1, "extracted": 1}
    assert predictor.calls == 1


def test_answer_baseline_prompts_require_minimal_answer_span():
    # Long answer sentences can lower QACC exact match.
    assert "minimal slot-filling answer span" in DIRECT_RAG_SYSTEM_PROMPT
    assert "minimal slot-filling answer span" in QACC_EXPLAIN_SYSTEM_PROMPT
    assert "reasoning only" in QACC_EXPLAIN_SYSTEM_PROMPT
    assert "own knowledge" in NO_CONTEXT_SYSTEM_PROMPT
    assert "weight may be wrong" in WEIGHTED_RAG_SYSTEM_PROMPT


def test_weighted_prompt_changes_only_weight_values():
    # Passage order and text stay fixed across all MaxEnt controls.
    example = PreparedExample(
        "q",
        "test",
        "Capital?",
        [
            Context(0, "Paris is the capital.", "one.example"),
            Context(1, "Lyon is the capital.", "two.example"),
        ],
    )
    uniform = weighted_answer_prompt(example, [0.5, 0.5])
    weighted = weighted_answer_prompt(example, [0.8, 0.2])
    assert uniform.index("Paris") < uniform.index("Lyon")
    assert weighted.index("Paris") < weighted.index("Lyon")
    assert uniform.replace("0.500000", "WEIGHT") == weighted.replace(
        "0.800000", "WEIGHT"
    ).replace("0.200000", "WEIGHT")


def test_answer_baseline_reports_conflict_slice():
    # The labelled conflict result must remain visible beside the full score.
    examples = _examples()
    predictions = {
        "0": {"answer": "Paris", "explanation": ""},
        "1": {"answer": "unknown", "explanation": ""},
    }
    summary = evaluate_answer_cache(
        examples,
        predictions,
        references={"0": ["Paris"], "1": ["Paris"]},
        evaluation_metadata={
            "0": {"has_conflict": True},
            "1": {"has_conflict": False},
        },
    )
    assert summary["answer_exact"] == 0.5
    assert summary["conflict_only"]["examples"] == 1
    assert summary["conflict_only"]["answer_exact"] == 1.0
