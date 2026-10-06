from pathlib import Path

from uacaa.answer_baselines import (
    DIRECT_RAG_SYSTEM_PROMPT,
    NO_CONTEXT_SYSTEM_PROMPT,
    QACC_EXPLAIN_SYSTEM_PROMPT,
    evaluate_answer_cache,
    read_answer_cache,
    write_answer_cache,
)
from uacaa.schemas import Context, PreparedExample


class _Predictor:
    def __init__(self):
        self.calls = 0

    def predict(self, example):
        self.calls += 1
        return {"answer": "Paris", "explanation": ""}


def _examples():
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


def test_answer_baseline_prompts_require_minimal_answer_span():
    assert "minimal slot-filling answer span" in DIRECT_RAG_SYSTEM_PROMPT
    assert "minimal slot-filling answer span" in QACC_EXPLAIN_SYSTEM_PROMPT
    assert "reasoning only" in QACC_EXPLAIN_SYSTEM_PROMPT
    assert "own knowledge" in NO_CONTEXT_SYSTEM_PROMPT


def test_answer_baseline_reports_conflict_slice():
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
