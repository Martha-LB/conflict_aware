from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from .dependency import evidence_dependency_groups, source_category
from .schemas import Candidate, Context


FEATURE_NAMES = (
    "log_support",
    "support_share",
    "mean_directness",
    "max_entailment",
    "mean_entailment",
    "mean_answer_type_match",
    "mean_temporal_fit",
    "official_source_share",
    "reference_source_share",
    "news_source_share",
    "social_source_share",
    "redundancy_ratio",
)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def candidate_feature_matrix(
    candidates: Sequence[Candidate],
    contexts: Sequence[Context],
    *,
    near_duplicate_jaccard: float = 0.88,
    dependency_correction: bool = True,
) -> np.ndarray:
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    by_index = {context.index: context for context in contexts}
    group_counts = []
    for candidate in candidates:
        raw_count = len({item.context_index for item in candidate.observations})
        group_counts.append(
            len(
                evidence_dependency_groups(
                    candidate,
                    contexts,
                    near_duplicate_jaccard=near_duplicate_jaccard,
                )
            )
            if dependency_correction
            else raw_count
        )
    total_groups = sum(group_counts)
    rows: list[list[float]] = []
    for candidate, independent_count in zip(candidates, group_counts):
        observations = candidate.observations
        raw_count = len({item.context_index for item in observations})
        categories = Counter(
            source_category(by_index[item.context_index].source)
            for item in observations
            if item.context_index in by_index
        )
        category_total = sum(categories.values()) or 1
        entailments = [item.entailment for item in observations]
        rows.append(
            [
                math.log1p(independent_count),
                independent_count / total_groups if total_groups else 0.0,
                _mean([item.directness for item in observations]),
                max(entailments, default=0.0),
                _mean(entailments),
                _mean([item.answer_type_match for item in observations], default=0.5),
                _mean([item.temporal_fit for item in observations], default=0.5),
                categories["official"] / category_total,
                categories["reference"] / category_total,
                categories["news"] / category_total,
                categories["social"] / category_total,
                raw_count / independent_count if independent_count else 0.0,
            ]
        )
    return np.asarray(rows, dtype=float)
