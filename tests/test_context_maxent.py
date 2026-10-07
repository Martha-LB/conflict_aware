"""Tests for notebook-aligned context MaxEnt features."""

import numpy as np

from conflict_maxent.context_maxent import build_context_maxent_record
from conflict_maxent.schemas import Context, EvidenceObservation, PreparedExample


class _Runtime:
    """Return fixed embeddings and NLI scores without loading real models."""

    def encode(self, texts):
        values = {
            "Who won?": [1.0, 0.0, 0.0],
            "Alice won.": [1.0, 0.0, 0.0],
            "Alice was the winner.": [0.7, 0.7, 0.0],
            "Bob won.": [0.0, 1.0, 0.0],
        }
        return np.asarray([values[text] for text in texts], dtype=float)

    def contradiction_probabilities(self, pairs):
        def score(pair):
            left, right = pair
            same_answer = ("Alice" in left) == ("Alice" in right)
            return 0.1 if same_answer else 0.9

        return np.asarray([score(pair) for pair in pairs], dtype=float)


def _example() -> PreparedExample:
    return PreparedExample(
        "q1",
        "test",
        "Who won?",
        [
            Context(0, "Alice won.", "https://reuters.com/a"),
            Context(1, "Alice was the winner.", "https://one.example/a"),
            Context(2, "Bob won.", "https://reddit.com/a"),
        ],
    )


def _observations() -> list[EvidenceObservation]:
    return [
        EvidenceObservation(0, "Alice", "Alice won"),
        EvidenceObservation(1, "Alice", "Alice was"),
        EvidenceObservation(2, "Bob", "Bob won"),
    ]


def test_context_maxent_has_uniform_prior_and_notebook_features():
    record = build_context_maxent_record(_example(), _observations(), _Runtime())
    for field in ("uniform_weights", "support_voting_weights", "maxent_weights"):
        assert len(record[field]) == 3
        assert np.isclose(sum(record[field]), 1.0)
        assert all(value >= 0 for value in record[field])
    assert record["projection"]["prior"] == "uniform_document_mass"
    constraints = record["projection"]["constraints"]
    assert "g2_contradiction" in constraints
    assert "g6_information_gain" in constraints
    assert "g4_source_reliability" in constraints
    assert any(name.startswith("h_") for name in constraints)
    # Independent support is a separate baseline, not the MaxEnt prior.
    assert record["support_voting_weights"] != record["uniform_weights"]


def test_context_maxent_never_reads_gold_labels():
    plain = build_context_maxent_record(_example(), _observations(), _Runtime())
    labelled = _example()
    labelled.supervision = {"correct_answer": "Bob"}
    labelled.evaluation = {"references": ["Bob"], "has_conflict": True}
    with_labels = build_context_maxent_record(labelled, _observations(), _Runtime())
    assert plain["maxent_weights"] == with_labels["maxent_weights"]


def test_each_original_context_is_one_evidence_unit():
    record = build_context_maxent_record(_example(), _observations(), _Runtime())
    assert [item["context_index"] for item in record["passage_features"]] == [0, 1, 2]
    assert len(record["maxent_weights"]) == 3
