from pathlib import Path

from uacaa.adaptive_gca import (
    AdaptiveGCAPredictor,
    evaluate_gca_cache,
    trigger_reasons,
    write_gca_cache,
)
from uacaa.schemas import Context, PreparedExample


def _example() -> PreparedExample:
    return PreparedExample(
        question_id="q1",
        split="dev",
        question="What is the capital of France?",
        contexts=[
            Context(0, "Paris is the capital of France.", "example.gov"),
            Context(1, "The capital is Lyon.", "example.com"),
        ],
        supervision={"correct_answer": "Paris"},
    )


class _FakePredictor(AdaptiveGCAPredictor):
    def __init__(self, *, conflict: bool):
        super().__init__("fake-model")
        self.conflict = conflict
        self.calls: list[str] = []

    def _request(self, *, name, schema, system, prompt):
        self.calls.append(name)
        if name == "qacc_joint_generation":
            if self.conflict:
                return (
                    {
                        "answer": "Lyon",
                        "evidence": [{"context_index": 1, "quote": "The capital is Lyon"}],
                        "alternatives": [
                            {
                                "answer": "Paris",
                                "evidence": [
                                    {
                                        "context_index": 0,
                                        "quote": "Paris is the capital of France",
                                    }
                                ],
                            }
                        ],
                        "contradictions": [
                            {
                                "answer_a": "Lyon",
                                "answer_b": "Paris",
                                "context_indices": [0, 1],
                            }
                        ],
                        "evidence_sufficient": True,
                        "confidence": "medium",
                    },
                    {"usage": {"total_tokens": 100}},
                )
            return (
                {
                    "answer": "Paris",
                    "evidence": [
                        {"context_index": 0, "quote": "Paris is the capital of France"}
                    ],
                    "alternatives": [],
                    "contradictions": [],
                    "evidence_sufficient": True,
                    "confidence": "high",
                },
                {"usage": {"total_tokens": 80}},
            )
        return (
            {
                "answer": "Paris",
                "selected_candidate_id": "C1",
                "decisive_evidence": [
                    {"context_index": 0, "quote": "Paris is the capital of France"}
                ],
                "strongest_counterevidence": [
                    {"context_index": 1, "quote": "The capital is Lyon"}
                ],
                "evidence_sufficient": True,
                "confidence": "high",
            },
            {"usage": {"total_tokens": 90}},
        )


def test_adaptive_gca_triggers_and_repairs_wrong_initial_answer(tmp_path: Path):
    predictor = _FakePredictor(conflict=True)
    record = predictor.predict(_example())
    assert predictor.calls == ["qacc_joint_generation", "qacc_blind_adjudication"]
    assert record["triggered"] is True
    assert "explicit_contradiction" in record["trigger_reasons"]
    assert "supported_alternative" in record["trigger_reasons"]
    assert record["initial_answer"] == "Lyon"
    assert record["final_answer"] == "Paris"
    assert all("primary" not in item for item in record["blind_candidates"])

    summary = evaluate_gca_cache(
        [_example()],
        {"q1": {"question_id": "q1", **record}},
        output_path=tmp_path / "scored.jsonl",
    )
    assert summary["initial_generation"]["answer_exact"] == 0.0
    assert summary["adaptive_final"]["answer_exact"] == 1.0
    assert summary["trigger"]["mean_llm_calls_per_question"] == 2.0
    assert summary["trigger"]["helped"] == 1
    assert summary["citation_validation"]["valid_rate"] == 1.0
    assert summary["usage"]["total_tokens"] == 190


def test_adaptive_gca_skips_second_call_when_evidence_is_consistent(tmp_path: Path):
    predictor = _FakePredictor(conflict=False)
    output = tmp_path / "gca.jsonl"
    progress = write_gca_cache([_example()], predictor, output)
    assert predictor.calls == ["qacc_joint_generation"]
    assert progress["llm_calls"] == 1
    assert progress["triggered"] == 0
    assert write_gca_cache([_example()], predictor, output)["reused"] == 1
    assert predictor.calls == ["qacc_joint_generation"]


def test_trigger_flags_invalid_or_insufficient_primary_evidence():
    reasons = trigger_reasons(
        {
            "answer": "Paris",
            "evidence": [],
            "alternatives": [],
            "contradictions": [],
            "evidence_sufficient": False,
        }
    )
    assert reasons == ["insufficient_evidence", "unsupported_primary_answer"]
