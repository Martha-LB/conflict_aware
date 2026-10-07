"""Group similar context answers into one candidate."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .normalization import answers_match, normalize_answer
from .schemas import Candidate, EvidenceObservation


class _UnionFind:
    """Small helper for joining matching answers."""
    def __init__(self, size: int) -> None:
        """Start with every observation in its own group."""

        self.parent = list(range(size))

    def find(self, index: int) -> int:
        """Find the group root and shorten the path for later calls."""

        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        """Join two answer groups when their aliases match."""

        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_observations(
    observations: Sequence[EvidenceObservation],
    *,
    answer_match_threshold: float = 0.8,
) -> list[Candidate]:
    """Merge equivalent extracted answers into candidate groups."""

    # Compare all answer pairs, then build groups.
    if not observations:
        return []
    union = _UnionFind(len(observations))
    for left in range(len(observations)):
        for right in range(left + 1, len(observations)):
            if answers_match(
                observations[left].candidate_answer,
                observations[right].candidate_answer,
                threshold=answer_match_threshold,
            ):
                union.union(left, right)
    # Collect all observations after transitive alias matching.
    groups: dict[int, list[EvidenceObservation]] = defaultdict(list)
    for index, observation in enumerate(observations):
        groups[union.find(index)].append(observation)

    candidates: list[Candidate] = []
    for group in groups.values():
        aliases = list(dict.fromkeys(item.candidate_answer.strip() for item in group))
        # Prefer the most common alias, then the shortest surface form.
        representative = max(
            aliases,
            key=lambda value: (
                sum(normalize_answer(value) == normalize_answer(item.candidate_answer) for item in group),
                -len(value),
            ),
        )
        candidates.append(Candidate(answer=representative, aliases=aliases, observations=list(group)))
    candidates.sort(key=lambda item: normalize_answer(item.answer))
    return candidates
