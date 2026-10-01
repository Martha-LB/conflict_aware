from pathlib import Path

from uacaa.error_analysis import analyze_unmapped, write_unmapped_analysis
from uacaa.schemas import Context, EvidenceObservation, PreparedExample


def _example(question_id: str, gold: str) -> PreparedExample:
    return PreparedExample(
        question_id,
        "train",
        "How many?",
        [Context(0, f"The answer is {gold}.", "example.test")],
        supervision={
            "correct_answer": gold,
            "candidates": [{"answer": gold, "context_indices": [0]}],
        },
    )


def test_unmapped_analysis_separates_mapping_and_extraction_failures(tmp_path: Path):
    examples = [
        _example("mapped", "Paris"),
        _example("number-word", "two"),
        _example("missing", "Rome"),
    ]
    extractions = {
        "mapped": [EvidenceObservation(0, "Paris")],
        "number-word": [EvidenceObservation(0, "2")],
        "missing": [],
    }
    rows, report = analyze_unmapped(examples, extractions)

    assert report.examples == 3
    assert report.mapped == 1
    assert report.unmapped == 2
    assert report.no_candidates == 1
    assert report.label_unmapped == 1
    assert report.candidate_recall == 1 / 3
    by_id = {row["question_id"]: row for row in rows}
    assert by_id["number-word"]["lexical_relation"] == "numeric_format_variant"
    assert by_id["number-word"]["extraction_status"] == "gold_context_answered"
    assert by_id["missing"]["extraction_status"] == "no_candidates"

    outputs = write_unmapped_analysis(rows, report, tmp_path / "unmapped.jsonl")
    assert Path(outputs["details"]).exists()
    assert Path(outputs["summary"]).exists()
    assert Path(outputs["review_csv"]).exists()
