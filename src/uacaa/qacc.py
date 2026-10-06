from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .normalization import (
    best_answer_index,
    is_no_answer,
    normalize_answer,
    qacc_normalize_text,
)
from .schemas import Context, PreparedExample


ORDINALS = ("first", "second", "third")


def qacc_conflict_label(example: PreparedExample) -> bool:
    """Return the QACC author conflict label, with a legacy answer-count fallback."""
    supervision = example.supervision or {}
    if "has_conflict" in supervision:
        return bool(supervision["has_conflict"])
    answers = {
        re.sub(r"\s+", " ", str(item.get("answer", "")).strip().lower())
        for item in supervision.get("candidates", [])
        if str(item.get("answer", "")).strip()
    }
    return len(answers) >= 2


def _question_key(value: object) -> str:
    return " ".join(str(value or "").casefold().split()).rstrip("?").strip()


def _deduplicate_references(values: Iterable[object]) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()
    for value in values:
        if is_missing(value):
            continue
        answer = str(value).strip()
        key = qacc_normalize_text(answer)
        if not key or key in seen:
            continue
        seen.add(key)
        references.append(answer)
    return references


def load_ambigqa_references(paths: Sequence[Path]) -> dict[str, tuple[str, ...]]:
    """Index acceptable single-answer aliases from official AmbigQA light data.

    QACC was sampled from AmbigQA, but its released train/dev rows omit the
    ``ambigqa_answer`` list retained in the QACC test split.  The QACC paper's
    evaluator scores against that upstream answer list, taking the maximum
    over references.  This loader restores the corresponding aliases without
    exposing them to inference.
    """

    collected: dict[str, list[str]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"AmbigQA file must contain a JSON list: {path}")
        for record in payload:
            if not isinstance(record, dict):
                continue
            key = _question_key(record.get("question"))
            if not key:
                continue
            answers: list[object] = []
            for annotation in record.get("annotations", []):
                if not isinstance(annotation, dict):
                    continue
                # QACC targets questions treated as unambiguous.  Do not mix
                # answers to rewritten disambiguated qaPairs into the original
                # question's reference set.
                if annotation.get("type") == "singleAnswer":
                    answers.extend(annotation.get("answer") or [])
            merged = [*collected.get(key, []), *answers]
            collected[key] = _deduplicate_references(merged)
    return {key: tuple(values) for key, values in collected.items() if values}


def evaluation_references(example: PreparedExample) -> list[str]:
    """Return frozen evaluation references, falling back to QACC annotation."""

    evaluation = example.evaluation or {}
    supervision = example.supervision or {}
    values = evaluation.get("references") or supervision.get("references") or []
    references = _deduplicate_references(values)
    if references:
        return references
    answer = supervision.get("correct_answer")
    return [] if is_missing(answer) else [str(answer).strip()]


def is_missing(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or not str(value).strip()
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_qacc(path: Path) -> list[dict[str, Any]]:
    # Python's JSON parser deliberately accepts the non-standard NaN tokens in
    # the official QACC release. They are normalized by is_missing downstream.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("QACC must be a JSON array")
    return payload


def parse_context_labels(value: Any, context_count: int) -> list[int]:
    if is_missing(value):
        return []
    try:
        labels = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid QACC context label list: {value!r}") from error
    if not isinstance(labels, list):
        raise ValueError(f"QACC context labels must be a list: {value!r}")
    indices: list[int] = []
    for label in labels:
        key = str(label).strip().upper()
        if len(key) != 1 or not "A" <= key <= "J":
            raise ValueError(f"Invalid QACC context label: {label!r}")
        index = ord(key) - ord("A")
        if index >= context_count:
            raise ValueError(f"Context label {key} exceeds {context_count} contexts")
        indices.append(index)
    return list(dict.fromkeys(indices))


def candidate_annotations(row: dict[str, Any]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    context_count = len(row.get("contexts") or [])
    for ordinal in ORDINALS:
        answer = row.get(f"{ordinal}Answer")
        if is_missing(answer):
            continue
        annotations.append(
            {
                "answer": str(answer).strip(),
                "context_indices": parse_context_labels(
                    row.get(f"{ordinal}Context"), context_count
                ),
            }
        )
    return annotations


def row_quality_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    annotations = candidate_annotations(row)
    if row.get("fourthAnswerExist") == "A":
        flags.append("truncated_fourth_answer")
    if any(not item["context_indices"] for item in annotations):
        flags.append("candidate_without_context")
    if any(is_no_answer(item["answer"]) for item in annotations):
        flags.append("no_answer_candidate")

    assigned = [index for item in annotations for index in item["context_indices"]]
    if len(assigned) != len(set(assigned)):
        flags.append("cross_candidate_context_overlap")

    candidate_keys = [normalize_answer(item["answer"]) for item in annotations]
    if len(candidate_keys) != len(set(candidate_keys)):
        flags.append("duplicate_candidate_answer")

    correct = row.get("correctAnswer")
    if best_answer_index(correct, [item["answer"] for item in annotations]) is None:
        flags.append("correct_answer_unmapped")
    return flags


def _question_id(row: dict[str, Any], position: int) -> str:
    raw_id = row.get("annotation_task_id")
    if not is_missing(raw_id):
        # Position keeps repeated official rows distinct in extraction caches
        # and evaluation outputs while the raw annotation id remains visible.
        return f"qacc-{row.get('split', 'unknown')}-{raw_id}-{position}"
    digest = hashlib.sha256(
        f"{position}|{row.get('question', '')}".encode("utf-8")
    ).hexdigest()[:16]
    return f"qacc-{digest}"


def prepare_example(
    row: dict[str, Any],
    position: int,
    *,
    upstream_references: Sequence[str] | None = None,
) -> PreparedExample:
    texts = [str(value).strip() for value in row.get("contexts") or []]
    sources = row.get("sources") or []
    contexts = [
        Context(
            index=index,
            text=text,
            source=(str(sources[index]).strip() if index < len(sources) and not is_missing(sources[index]) else None),
        )
        for index, text in enumerate(texts)
    ]
    annotations = candidate_annotations(row)
    restored_references = _deduplicate_references(upstream_references or [])
    supervision = {
        "candidates": annotations,
        # Preserve the authors' human annotation directly. Do not infer the
        # target from candidate count, because malformed or missing answer
        # fields must not silently rewrite the original judgment.
        "has_conflict": row.get("secondAnswerExist") == "A",
        "correct_answer": None if is_missing(row.get("correctAnswer")) else str(row["correctAnswer"]).strip(),
        # Evaluation-only aliases restored from upstream AmbigQA for train/dev.
        # They are not inputs to extraction or answer generation.
        "references": restored_references,
        "reasons": None if is_missing(row.get("reasons")) else str(row["reasons"]),
        "explanation": None if is_missing(row.get("explanation")) else str(row["explanation"]).strip(),
    }
    raw_references = row.get("ambigqa_answer")
    references = (
        _deduplicate_references(raw_references)
        if isinstance(raw_references, list)
        else []
    )
    evaluation = (
        {
            "references": references,
            # This annotation is evaluation-only. It is physically removed from
            # test inputs and retained beside the references for subgroup scores.
            "has_conflict": row.get("secondAnswerExist") == "A",
        }
        if references
        else None
    )
    return PreparedExample(
        question_id=_question_id(row, position),
        split=str(row.get("split") or "unknown"),
        question=str(row.get("question") or "").strip(),
        contexts=contexts,
        supervision=supervision,
        evaluation=evaluation,
        quality_flags=row_quality_flags(row),
    )


def audit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    split_summary: dict[str, Any] = {}
    for split in sorted({str(row.get("split") or "unknown") for row in materialized}):
        selected = [row for row in materialized if str(row.get("split") or "unknown") == split]
        context_total = sum(len(row.get("contexts") or []) for row in selected)
        attributed = 0
        flag_counts: Counter[str] = Counter()
        candidate_counts: Counter[int] = Counter()
        questions: Counter[str] = Counter()
        ids: Counter[str] = Counter()
        for position, row in enumerate(selected):
            annotations = candidate_annotations(row)
            candidate_counts[len(annotations)] += 1
            attributed += len(
                {index for item in annotations for index in item["context_indices"]}
            )
            flag_counts.update(row_quality_flags(row))
            questions[normalize_answer(row.get("question"))] += 1
            raw_id = row.get("annotation_task_id")
            ids[f"{type(raw_id).__name__}:{raw_id}"] += 1
        split_summary[split] = {
            "rows": len(selected),
            "contexts": context_total,
            "candidate_count_distribution": dict(sorted(candidate_counts.items())),
            "attributed_contexts": attributed,
            "attribution_coverage": attributed / context_total if context_total else 0.0,
            "quality_flags": dict(sorted(flag_counts.items())),
            "duplicate_question_rows": sum(count - 1 for count in questions.values() if count > 1),
            "duplicate_id_rows": sum(count - 1 for count in ids.values() if count > 1),
        }
    return {"total_rows": len(materialized), "splits": split_summary}
