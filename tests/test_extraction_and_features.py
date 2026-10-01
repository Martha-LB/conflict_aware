import numpy as np
from pathlib import Path
import io
import urllib.error

from uacaa.clustering import cluster_observations
from uacaa.extraction import (
    OpenAICompatibleExtractor,
    align_evidence_quote,
    evidence_quote_is_verbatim,
    parse_context_assessments,
    parse_extraction_response,
    read_extraction_cache,
    write_extraction_cache,
)
from uacaa.features import FEATURE_NAMES, candidate_feature_matrix
from uacaa.pipeline import build_choice_sets
from uacaa.schemas import Context, EvidenceObservation, PreparedExample


def test_parse_cluster_and_dependency_correct_features():
    response = json_text = """{
      "observations": [
        {"context_index": 0, "answerable": true, "candidate_answer": "Paris", "evidence_quote": "Paris", "directness": 1, "entailment": 0.9, "answer_type_match": 1, "temporal_fit": 0.5},
        {"context_index": 1, "answerable": true, "candidate_answer": "The Paris", "evidence_quote": "Paris", "directness": 0.8, "entailment": 0.8, "answer_type_match": 1, "temporal_fit": 0.5},
        {"context_index": 2, "answerable": true, "candidate_answer": "Lyon", "evidence_quote": "Lyon", "directness": 0.9, "entailment": 0.7, "answer_type_match": 1, "temporal_fit": 0.5}
      ]
    }"""
    observations = parse_extraction_response(json_text, 3)
    candidates = cluster_observations(observations)
    assert len(candidates) == 2
    contexts = [
        Context(0, "Paris is the capital of France", "a.example.com"),
        Context(1, "Paris is the capital of France", "a.example.com"),
        Context(2, "Lyon is the capital of France", "b.example.com"),
    ]
    matrix = candidate_feature_matrix(candidates, contexts)
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()


def test_parser_requires_one_assessment_per_context():
    response = """{
      "observations": [
        {"context_index": 0, "answerable": false, "candidate_answer": "", "evidence_quote": "", "directness": 0, "entailment": 0, "answer_type_match": 0, "temporal_fit": 0}
      ]
    }"""
    try:
        parse_extraction_response(response, 2)
    except ValueError as error:
        assert "omitted context_indices: [1]" in str(error)
    else:
        raise AssertionError("Expected omitted context validation")


def test_parser_can_quarantine_missing_assessment_after_retries():
    response = """{
      "observations": [
        {"context_index": 0, "answerable": false, "candidate_answer": "", "evidence_quote": "", "directness": 0, "entailment": 0, "answer_type_match": 0, "temporal_fit": 0}
      ]
    }"""
    observations, assessments = parse_context_assessments(
        response, 2, allow_missing=True
    )
    assert observations == []
    assert len(assessments) == 2
    assert assessments[1]["context_index"] == 1
    assert assessments[1]["omitted_by_model"] is True


def test_parser_can_quarantine_malformed_answerable_after_retries():
    response = """{
      "observations": [
        {"context_index": 0, "answerable": true, "candidate_answer": "", "evidence_quote": "", "directness": 0.8, "entailment": 0.8, "answer_type_match": 1, "temporal_fit": 1}
      ]
    }"""
    observations, assessments = parse_context_assessments(
        response,
        1,
        quarantine_malformed_answerable=True,
    )
    assert observations == []
    assert assessments[0]["malformed_answerable"] is True


def test_parser_can_quarantine_duplicate_assessment_after_retries():
    response = """{
      "observations": [
        {"context_index": 0, "answerable": false, "candidate_answer": "", "evidence_quote": "", "directness": 0, "entailment": 0, "answer_type_match": 0, "temporal_fit": 0},
        {"context_index": 0, "answerable": false, "candidate_answer": "", "evidence_quote": "", "directness": 0, "entailment": 0, "answer_type_match": 0, "temporal_fit": 0}
      ]
    }"""
    observations, assessments = parse_context_assessments(
        response,
        1,
        quarantine_structural_errors=True,
    )
    assert observations == []
    assert len(assessments) == 1
    assert assessments[0]["duplicate_assessment"] is True


def test_verbatim_quote_normalizes_case_and_whitespace_only():
    assert evidence_quote_is_verbatim("PARIS  is the capital", "Paris is\tthe capital.")
    assert not evidence_quote_is_verbatim("Paris was the capital", "Paris is the capital.")


def test_quote_alignment_allows_punctuation_only_and_returns_context_span():
    context = "The rifles use 22 caliber long rifle ammunition."
    assert align_evidence_quote(".22 caliber long rifle", context) == "22 caliber long rifle"
    assert align_evidence_quote("rifles use ammunition", context) is None


def test_extraction_cache_resumes_without_repeating_calls(tmp_path: Path):
    examples = [
        PreparedExample(f"q-{index}", "train", f"Question {index}", [Context(0, "text")])
        for index in range(2)
    ]

    class Extractor:
        def __init__(self):
            self.calls = 0

        def extract(self, example):
            self.calls += 1
            return [EvidenceObservation(0, example.question)]

    output = tmp_path / "cache.jsonl"
    first = Extractor()
    progress = write_extraction_cache(examples[:1], first, output)
    assert progress == {"requested": 1, "reused": 0, "extracted": 1}

    second = Extractor()
    progress = write_extraction_cache(examples, second, output)
    assert progress == {"requested": 2, "reused": 1, "extracted": 1}
    assert second.calls == 1
    assert set(read_extraction_cache(output)) == {"q-0", "q-1"}


def test_http_400_surfaces_api_message_without_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        body = io.BytesIO(b'{"error":{"message":"Unsupported parameter: temperature"}}')
        raise urllib.error.HTTPError("https://example.test", 400, "Bad Request", {}, body)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    extractor = OpenAICompatibleExtractor("gpt-test", retries=3)
    example = PreparedExample("q", "train", "Question", [Context(0, "text")])
    try:
        extractor.extract(example)
    except RuntimeError as error:
        assert "Unsupported parameter: temperature" in str(error)
    else:
        raise AssertionError("Expected a surfaced API error")
    assert calls == 1


def test_build_report_counts_single_candidates_only_after_label_filtering():
    examples = [
        PreparedExample(
            "mapped",
            "train",
            "Question",
            [Context(0, "Paris")],
            supervision={"correct_answer": "Paris"},
        ),
        PreparedExample(
            "unmapped",
            "train",
            "Question",
            [Context(0, "Lyon")],
            supervision={"correct_answer": "Paris"},
        ),
    ]
    extractions = {
        "mapped": [EvidenceObservation(0, "Paris")],
        "unmapped": [EvidenceObservation(0, "Lyon")],
    }
    _, report = build_choice_sets(
        examples, extractions, require_training_label=True
    )
    assert report.built == 1
    assert report.label_unmapped == 1
    assert report.single_candidate == 1
