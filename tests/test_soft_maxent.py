"""Tests for the soft projection and independent-source vote."""

import numpy as np

from conflict_maxent.schemas import Context, EvidenceObservation, PreparedExample
from conflict_maxent.soft_maxent import soft_information_projection, support_voting_weights


def _example() -> PreparedExample:
    """Make two independent sources support Alice and one support Bob."""

    return PreparedExample(
        question_id="q1",
        split="test",
        question="Who won?",
        contexts=[
            Context(0, "Alice won.", "https://one.example/item"),
            Context(1, "Alice was the winner.", "https://two.example/item"),
            Context(2, "Bob won.", "https://three.example/item"),
        ],
    )


def _observations() -> list[EvidenceObservation]:
    return [
        EvidenceObservation(0, "Alice", "Alice won"),
        EvidenceObservation(1, "Alice", "Alice was"),
        EvidenceObservation(2, "Bob", "Bob won"),
    ]


def test_soft_projection_is_valid_when_constraints_compete():
    # Slack keeps the problem valid even when both targets are impossible together.
    prior = np.asarray([0.5, 0.5])
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = soft_information_projection(
        prior, matrix, [0.9, 0.9], slack_penalty=1.0
    )
    assert np.isclose(result.distribution.sum(), 1.0)
    assert np.all(result.distribution > 0)
    assert np.all((0 <= result.lambdas) & (result.lambdas <= 1.0))
    assert result.max_violation > 0


def test_soft_projection_supports_upper_and_lower_constraints():
    # The first feature is preferred high. The second is preferred low.
    result = soft_information_projection(
        [0.5, 0.5],
        np.asarray([[1.0, 1.0], [0.0, 0.0]]),
        [0.7, 0.3],
        signs=[1.0, -1.0],
        slack_penalty=5.0,
    )
    # Competing constraints are symmetric, so slack leaves the prior unchanged.
    assert np.allclose(result.distribution, [0.5, 0.5], atol=1e-4)
    assert result.violated_constraints == 2


def test_support_vote_is_separate_and_label_free():
    weights, features = support_voting_weights(_example(), _observations())
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[2]
    assert weights[1] > weights[2]
    assert set(features[0]) == {
        "context_index",
        "source",
        "domain",
        "candidate_answer",
        "independent_answer_support",
    }

    labelled = _example()
    labelled.supervision = {"correct_answer": "Bob"}
    labelled.evaluation = {"references": ["Bob"], "has_conflict": True}
    labelled_weights, _ = support_voting_weights(labelled, _observations())
    assert np.allclose(weights, labelled_weights)
