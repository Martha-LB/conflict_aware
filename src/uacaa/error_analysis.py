from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .clustering import cluster_observations
from .normalization import normalize_answer, target_matches_candidate, token_f1
from .pipeline import build_choice_set
from .schemas import EvidenceObservation, PreparedExample


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "first": "1",
    "second": "2",
    "third": "3",
}


@dataclass(frozen=True)
class UnmappedAnalysisReport:
    examples: int
    mapped: int
    unmapped: int
    no_candidates: int
    label_unmapped: int
    candidate_recall: float
    by_extraction_status: dict[str, int]
    by_lexical_relation: dict[str, int]
    by_review_priority: dict[str, int]


def _number_set(value: object) -> set[str]:
    return set(re.findall(r"\d+", normalize_answer(value)))


def _canonical_tokens(value: object) -> list[str]:
    return [_NUMBER_WORDS.get(token, token) for token in normalize_answer(value).split()]


def _counter_f1(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return float(list(left) == list(right))
    overlap = sum((Counter(left) & Counter(right)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall)


def _lexical_relation(target: str, candidate: str) -> tuple[str, float, float]:
    score = token_f1(target, candidate)
    canonical_score = _counter_f1(
        _canonical_tokens(target),
        _canonical_tokens(candidate),
    )
    target_tokens = Counter(normalize_answer(target).split())
    candidate_tokens = Counter(normalize_answer(candidate).split())
    target_numbers = _number_set(target)
    candidate_numbers = _number_set(candidate)

    if canonical_score >= 0.5 and canonical_score > score + 0.2:
        return "numeric_format_variant", score, canonical_score
    if target_numbers and candidate_numbers and target_numbers != candidate_numbers:
        return "number_conflict", score, canonical_score
    if candidate_tokens and all(
        target_tokens[token] >= count for token, count in candidate_tokens.items()
    ):
        return "candidate_subspan_of_gold", score, canonical_score
    if target_tokens and all(
        candidate_tokens[token] >= count for token, count in target_tokens.items()
    ):
        return "gold_subspan_of_candidate", score, canonical_score
    if score >= 0.67:
        return "high_lexical_overlap", score, canonical_score
    if score >= 0.35:
        return "partial_lexical_overlap", score, canonical_score
    return "lexically_disjoint", score, canonical_score


def _gold_context_indices(example: PreparedExample, target: str) -> list[int]:
    if not example.supervision:
        return []
    indices: set[int] = set()
    for annotation in example.supervision.get("candidates", []):
        answer = str(annotation.get("answer", ""))
        if target_matches_candidate(target, answer) or target_matches_candidate(answer, target):
            indices.update(int(index) for index in annotation.get("context_indices", []))
    return sorted(indices)


def _review_priority(extraction_status: str, lexical_relation: str) -> str:
    if extraction_status in {"no_candidates", "gold_context_unanswered"}:
        return "extraction_recall_review"
    if lexical_relation == "numeric_format_variant":
        return "normalization_review"
    if lexical_relation == "number_conflict":
        return "numeric_or_temporal_conflict_review"
    if lexical_relation in {
        "candidate_subspan_of_gold",
        "gold_subspan_of_candidate",
        "high_lexical_overlap",
        "partial_lexical_overlap",
    }:
        return "alias_or_granularity_review"
    if extraction_status == "gold_context_answered":
        return "label_or_extractor_disagreement_review"
    return "semantic_alias_review"


def analyze_unmapped(
    examples: Sequence[PreparedExample],
    extractions: Mapping[str, Sequence[EvidenceObservation]],
    *,
    answer_match_threshold: float = 0.8,
) -> tuple[list[dict], UnmappedAnalysisReport]:
    rows: list[dict] = []
    mapped = 0
    no_candidates = 0

    for example in examples:
        supervision = example.supervision or {}
        target = str(supervision.get("correct_answer", "")).strip()
        if not target:
            continue
        observations = list(extractions.get(example.question_id, []))
        choice_set = build_choice_set(
            example,
            observations,
            answer_match_threshold=answer_match_threshold,
            require_training_label=False,
        )
        if choice_set is not None and choice_set.correct_index is not None:
            mapped += 1
            continue

        candidates = cluster_observations(
            observations,
            answer_match_threshold=answer_match_threshold,
        )
        if not candidates:
            no_candidates += 1

        candidate_details = [
            {
                "answer": candidate.answer,
                "aliases": candidate.aliases,
                "context_indices": sorted(
                    {item.context_index for item in candidate.observations}
                ),
                "supporting_contexts": len(
                    {item.context_index for item in candidate.observations}
                ),
            }
            for candidate in candidates
        ]
        alias_rows = [
            (alias, detail["answer"])
            for candidate, detail in zip(candidates, candidate_details)
            for alias in candidate.aliases
        ]
        if alias_rows:
            best_alias, best_representative = max(
                alias_rows,
                key=lambda item: token_f1(target, item[0]),
            )
            lexical_relation, best_f1, canonical_f1 = _lexical_relation(
                target, best_alias
            )
        else:
            best_alias = None
            best_representative = None
            lexical_relation = "no_candidate"
            best_f1 = 0.0
            canonical_f1 = 0.0

        gold_indices = _gold_context_indices(example, target)
        answered_indices = sorted({item.context_index for item in observations})
        answered_gold_indices = sorted(set(gold_indices) & set(answered_indices))
        if not candidates:
            extraction_status = "no_candidates"
        elif not gold_indices:
            extraction_status = "no_gold_context_annotation"
        elif answered_gold_indices:
            extraction_status = "gold_context_answered"
        else:
            extraction_status = "gold_context_unanswered"

        priority = _review_priority(extraction_status, lexical_relation)
        context_by_index = {context.index: context for context in example.contexts}
        gold_contexts = [
            {
                "context_index": index,
                "source": context_by_index[index].source,
                "text": context_by_index[index].text,
            }
            for index in gold_indices
            if index in context_by_index
        ]
        rows.append(
            {
                "split": example.split,
                "question_id": example.question_id,
                "question": example.question,
                "gold_answer": target,
                "gold_answer_normalized": normalize_answer(target),
                "qacc_candidate_annotations": supervision.get("candidates", []),
                "gold_context_indices": gold_indices,
                "answered_context_indices": answered_indices,
                "answered_gold_context_indices": answered_gold_indices,
                "gold_contexts": gold_contexts,
                "candidate_count": len(candidates),
                "candidates": candidate_details,
                "best_candidate": best_representative,
                "best_alias": best_alias,
                "best_token_f1": best_f1,
                "best_canonical_number_f1": canonical_f1,
                "gold_numbers": sorted(_number_set(target)),
                "candidate_numbers": (
                    sorted(_number_set(best_alias)) if best_alias is not None else []
                ),
                "extraction_status": extraction_status,
                "lexical_relation": lexical_relation,
                "review_priority": priority,
                "manual_resolution": None,
                "manual_notes": None,
            }
        )

    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[field]) for row in rows).items()))

    report = UnmappedAnalysisReport(
        examples=len(examples),
        mapped=mapped,
        unmapped=len(rows),
        no_candidates=no_candidates,
        label_unmapped=len(rows) - no_candidates,
        candidate_recall=mapped / len(examples) if examples else 0.0,
        by_extraction_status=counts("extraction_status"),
        by_lexical_relation=counts("lexical_relation"),
        by_review_priority=counts("review_priority"),
    )
    return rows, report


def write_unmapped_analysis(
    rows: Sequence[dict],
    report: UnmappedAnalysisReport,
    output_path: Path,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    review_path = output_path.with_suffix(".review.csv")
    fields = [
        "split",
        "question_id",
        "question",
        "gold_answer",
        "best_alias",
        "best_token_f1",
        "best_canonical_number_f1",
        "candidate_count",
        "extraction_status",
        "lexical_relation",
        "review_priority",
        "manual_resolution",
        "manual_notes",
    ]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "details": str(output_path),
        "summary": str(summary_path),
        "review_csv": str(review_path),
        "report": asdict(report),
    }


__all__ = ["UnmappedAnalysisReport", "analyze_unmapped", "write_unmapped_analysis"]
