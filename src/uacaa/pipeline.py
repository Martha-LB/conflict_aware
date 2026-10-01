from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .clustering import cluster_observations
from .features import FEATURE_NAMES, candidate_feature_matrix
from .normalization import best_answer_index, target_matches_candidate
from .schemas import ChoiceSet, EvidenceObservation, PreparedExample


@dataclass(frozen=True)
class ChoiceSetBuildReport:
    examples: int
    built: int
    no_candidates: int
    label_unmapped: int
    single_candidate: int


def build_choice_set(
    example: PreparedExample,
    observations: Sequence[EvidenceObservation],
    *,
    answer_match_threshold: float = 0.8,
    near_duplicate_jaccard: float = 0.88,
    dependency_correction: bool = True,
    require_training_label: bool = False,
    references: Sequence[str] | None = None,
    has_conflict: bool | None = None,
) -> ChoiceSet | None:
    candidates = cluster_observations(
        observations,
        answer_match_threshold=answer_match_threshold,
    )
    if not candidates:
        return None
    answers = tuple(candidate.answer for candidate in candidates)
    features = candidate_feature_matrix(
        candidates,
        example.contexts,
        near_duplicate_jaccard=near_duplicate_jaccard,
        dependency_correction=dependency_correction,
    )

    correct_index: int | None = None
    if example.supervision and example.supervision.get("correct_answer"):
        target = example.supervision["correct_answer"]
        alias_sets = [candidate.aliases or [candidate.answer] for candidate in candidates]
        matches = [
            index
            for index, aliases in enumerate(alias_sets)
            if any(
                target_matches_candidate(
                    target,
                    alias,
                    threshold=answer_match_threshold,
                )
                for alias in aliases
            )
        ]
        if matches:
            correct_index = matches[0]
        else:
            correct_index = best_answer_index(
                target,
                answers,
                threshold=answer_match_threshold,
            )
    if require_training_label and correct_index is None:
        return None

    resolved_references = tuple(
        str(value)
        for value in (
            references
            if references is not None
            else (example.evaluation or {}).get("references", [])
        )
        if str(value).strip()
    )
    return ChoiceSet(
        question_id=example.question_id,
        answers=answers,
        features=features,
        correct_index=correct_index,
        references=resolved_references,
        has_conflict=has_conflict,
    )


def build_choice_sets(
    examples: Sequence[PreparedExample],
    extractions: Mapping[str, Sequence[EvidenceObservation]],
    *,
    require_training_label: bool = False,
    answer_match_threshold: float = 0.8,
    near_duplicate_jaccard: float = 0.88,
    dependency_correction: bool = True,
    references: Mapping[str, Sequence[str]] | None = None,
    evaluation_metadata: Mapping[str, Mapping] | None = None,
) -> tuple[list[ChoiceSet], ChoiceSetBuildReport]:
    choice_sets: list[ChoiceSet] = []
    no_candidates = 0
    label_unmapped = 0
    single_candidate = 0
    for example in examples:
        observations = extractions.get(example.question_id, [])
        item_metadata = (evaluation_metadata or {}).get(example.question_id, {})
        unlabelled = build_choice_set(
            example,
            observations,
            answer_match_threshold=answer_match_threshold,
            near_duplicate_jaccard=near_duplicate_jaccard,
            dependency_correction=dependency_correction,
            require_training_label=False,
            references=(references or {}).get(
                example.question_id, item_metadata.get("references")
            ),
            has_conflict=item_metadata.get("has_conflict"),
        )
        if unlabelled is None:
            no_candidates += 1
            continue
        if require_training_label and unlabelled.correct_index is None:
            label_unmapped += 1
            continue
        if len(unlabelled.answers) == 1:
            single_candidate += 1
        choice_sets.append(unlabelled)
    return choice_sets, ChoiceSetBuildReport(
        examples=len(examples),
        built=len(choice_sets),
        no_candidates=no_candidates,
        label_unmapped=label_unmapped,
        single_candidate=single_candidate,
    )


__all__ = ["FEATURE_NAMES", "build_choice_set", "build_choice_sets"]
