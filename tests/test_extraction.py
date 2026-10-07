"""Tests for answer extraction and cache resume."""

import io
import urllib.error
from pathlib import Path

from conflict_maxent.clustering import cluster_observations
from conflict_maxent.extraction import (
    OpenAICompatibleExtractor,
    align_evidence_quote,
    evidence_quote_is_verbatim,
    parse_context_assessments,
    parse_extraction_response,
    read_extraction_cache,
    write_extraction_cache,
)
from conflict_maxent.schemas import Context, EvidenceObservation, PreparedExample


def test_parse_and_cluster_context_answers():
    # Paris and The Paris form one candidate; Lyon remains separate.
    response = """{
      "observations": [
        {"context_index": 0, "answerable": true, "candidate_answer": "Paris", "evidence_quote": "Paris", "directness": 1, "entailment": 0.9, "answer_type_match": 1, "temporal_fit": 0.5},
        {"context_index": 1, "answerable": true, "candidate_answer": "The Paris", "evidence_quote": "Paris", "directness": 0.8, "entailment": 0.8, "answer_type_match": 1, "temporal_fit": 0.5},
        {"context_index": 2, "answerable": true, "candidate_answer": "Lyon", "evidence_quote": "Lyon", "directness": 0.9, "entailment": 0.7, "answer_type_match": 1, "temporal_fit": 0.5}
      ]
    }"""
    observations = parse_extraction_response(response, 3)
    assert len(cluster_observations(observations)) == 2


def test_parser_requires_one_assessment_per_context():
    # Missing a context is an error before the final quarantine attempt.
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


def test_parser_quarantines_missing_or_malformed_assessments():
    # Quarantine keeps valid contexts instead of dropping the whole question.
    missing = """{
      "observations": [
        {"context_index": 0, "answerable": false, "candidate_answer": "", "evidence_quote": "", "directness": 0, "entailment": 0, "answer_type_match": 0, "temporal_fit": 0}
      ]
    }"""
    observations, assessments = parse_context_assessments(
        missing, 2, allow_missing=True
    )
    assert observations == []
    assert assessments[1]["omitted_by_model"] is True

    malformed = """{
      "observations": [
        {"context_index": 0, "answerable": true, "candidate_answer": "", "evidence_quote": "", "directness": 0.8, "entailment": 0.8, "answer_type_match": 1, "temporal_fit": 1}
      ]
    }"""
    observations, assessments = parse_context_assessments(
        malformed, 1, quarantine_malformed_answerable=True
    )
    assert observations == []
    assert assessments[0]["malformed_answerable"] is True


def test_quote_validation_and_alignment():
    # Safe alignment fixes formatting, but never skips words.
    assert evidence_quote_is_verbatim(
        "PARIS  is the capital", "Paris is\tthe capital."
    )
    assert not evidence_quote_is_verbatim(
        "Paris was the capital", "Paris is the capital."
    )
    context = "The rifles use 22 caliber long rifle ammunition."
    assert align_evidence_quote(
        ".22 caliber long rifle", context
    ) == "22 caliber long rifle"
    assert align_evidence_quote("rifles use ammunition", context) is None


def test_extraction_cache_resumes_without_repeating_calls(tmp_path: Path):
    # Fake extraction makes resume behavior deterministic and free.
    examples = [
        PreparedExample(f"q-{index}", "train", f"Question {index}", [Context(0, "text")])
        for index in range(2)
    ]

    class Extractor:
        """Count calls while returning one simple observation."""

        def __init__(self):
            self.calls = 0

        def extract(self, example):
            self.calls += 1
            return [EvidenceObservation(0, example.question)]

    output = tmp_path / "cache.jsonl"
    first = Extractor()
    assert write_extraction_cache(examples[:1], first, output) == {
        "requested": 1,
        "reused": 0,
        "extracted": 1,
    }
    second = Extractor()
    assert write_extraction_cache(examples, second, output) == {
        "requested": 2,
        "reused": 1,
        "extracted": 1,
    }
    assert second.calls == 1
    assert set(read_extraction_cache(output)) == {"q-0", "q-1"}


def test_http_400_surfaces_api_message_without_retry(monkeypatch):
    # A bad request should stop fast and show the API message.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = 0

    def fail(*args, **kwargs):
        """Return the same client error on every attempted request."""

        nonlocal calls
        calls += 1
        body = io.BytesIO(b'{"error":{"message":"Unsupported parameter: temperature"}}')
        raise urllib.error.HTTPError(
            "https://example.test", 400, "Bad Request", {}, body
        )

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
