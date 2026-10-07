"""Tests for the final comparison report."""

from pathlib import Path

from conflict_maxent.identifiability import (
    evaluate_identifiability,
    read_jsonl_records,
    write_candidate_pool,
    write_identifiability_report,
)
from conflict_maxent.schemas import Context, EvidenceObservation, PreparedExample


def _examples():
    """Create one conflict and one non-conflict test row."""

    return [
        PreparedExample("a", "test", "Q1?", [Context(0, "Alpha")]),
        PreparedExample("b", "test", "Q2?", [Context(0, "Beta")]),
    ]


def test_identifiability_report_separates_candidate_availability_and_selection(tmp_path: Path):
    # The oracle has a gold answer available. The resolver still must select it.
    systems = {
        "no_context": ({"a": {"answer": "wrong"}, "b": {"answer": "Beta"}}, "answer"),
        "explain": ({"a": {"answer": "Alpha"}, "b": {"answer": "wrong"}}, "answer"),
        "robust": (
            {
                "a": {
                    "primary_answer": "Alpha",
                    "robust_answer": "Alpha",
                    "candidate_cards": [{"answer": "wrong"}],
                },
                "b": {
                    "primary_answer": "wrong",
                    "robust_answer": "Beta",
                    "candidate_cards": [{"answer": "Beta"}],
                },
            },
            "robust_answer",
        ),
        "oracle": (
            {
                "a": {
                    "primary_answer": "Alpha",
                    "candidate_cards": [{"answer": "wrong"}],
                },
                "b": {
                    "primary_answer": "wrong",
                    "candidate_cards": [{"answer": "Beta"}],
                },
            },
            "oracle_candidate",
        ),
    }
    # The report should show both resolver gain and remaining oracle gap.
    summary, rows = evaluate_identifiability(
        _examples(),
        {"a": ["Alpha"], "b": ["Beta"]},
        {"a": True, "b": False},
        systems,
        primary_name="explain",
        no_context_name="no_context",
        resolver_name="robust",
        oracle_name="oracle",
    )
    assert summary["metrics"]["overall"]["explain"]["answer_exact"] == 0.5
    assert summary["metrics"]["overall"]["robust"]["answer_exact"] == 1.0
    assert summary["identifiability_gaps"]["overall"]["oracle_minus_resolver"] == 0.0
    assert summary["comparisons_to_primary"]["overall"]["robust"]["helped"] == 1
    assert summary["metrics"]["qacc_conflict"]["explain"]["examples"] == 1
    # Human and machine-readable outputs must be written together.
    output = tmp_path / "report.json"
    write_identifiability_report(output, summary, rows)
    assert output.exists()
    assert output.with_suffix(".rows.jsonl").exists()
    assert output.with_suffix(".md").exists()


def test_candidate_pool_uses_models_and_contexts_without_references(tmp_path: Path):
    # The diagnostic pool contains observable answers but never gold references.
    examples = _examples()
    output = tmp_path / "candidates.jsonl"
    progress = write_candidate_pool(
        examples,
        {"a": {"answer": "Alpha"}, "b": {"answer": "wrong"}},
        {"a": {"answer": "other"}, "b": {"answer": "Beta"}},
        {"a": [EvidenceObservation(0, "context answer")]},
        output,
    )
    records = read_jsonl_records(output)
    assert progress["examples"] == 2
    assert [card["answer"] for card in records["a"]["candidate_cards"]] == [
        "Alpha",
        "other",
        "context answer",
    ]
    assert "references" not in records["a"]
