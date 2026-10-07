"""Tests for command-line cache safety helpers."""

from conflict_maxent.cli import _provenance_mismatches


def test_moved_input_is_safe_when_hash_is_identical():
    previous = {"input": "/old/data.jsonl", "input_sha256": "same", "model": "m"}
    current = {"input": "/new/data.jsonl", "input_sha256": "same", "model": "m"}
    assert _provenance_mismatches(previous, current) == []


def test_changed_content_or_model_is_not_safe_to_resume():
    previous = {"input": "/data.jsonl", "input_sha256": "old", "model": "m1"}
    current = {"input": "/data.jsonl", "input_sha256": "new", "model": "m2"}
    assert _provenance_mismatches(previous, current) == ["input_sha256", "model"]
