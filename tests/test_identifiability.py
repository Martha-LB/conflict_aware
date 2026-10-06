from pathlib import Path

from uacaa.identifiability import evaluate_identifiability, write_identifiability_report
from uacaa.schemas import Context, PreparedExample


def _examples():
    return [
        PreparedExample("a", "test", "Q1?", [Context(0, "Alpha")]),
        PreparedExample("b", "test", "Q2?", [Context(0, "Beta")]),
    ]


def test_identifiability_report_separates_candidate_availability_and_selection(tmp_path: Path):
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
    output = tmp_path / "report.json"
    write_identifiability_report(output, summary, rows)
    assert output.exists()
    assert output.with_suffix(".rows.jsonl").exists()
    assert output.with_suffix(".md").exists()
