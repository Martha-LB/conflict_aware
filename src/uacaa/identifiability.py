from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .normalization import is_no_answer, qacc_exact_match, qacc_normalize_text, qacc_token_f1
from .schemas import PreparedExample


EVALUATION_PROTOCOL = "qacc_public_normalized_em_f1_max_over_references"


def read_jsonl_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                question_id = str(record["question_id"])
                if question_id in records:
                    raise ValueError(f"Duplicate question_id in {path}: {question_id}")
                records[question_id] = record
    return records


def _field(record: Mapping, field: str) -> object:
    value: object = record
    for key in field.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _score(answer: object, references: Sequence[str]) -> tuple[float, float, bool]:
    text = str(answer or "").strip()
    abstained = not text or is_no_answer(text)
    if abstained:
        return 0.0, 0.0, True
    exact = float(any(qacc_exact_match(text, reference) for reference in references))
    f1 = max((qacc_token_f1(text, reference) for reference in references), default=0.0)
    return exact, f1, False


def _oracle_candidate(record: Mapping, references: Sequence[str]) -> str:
    candidates = [str(record.get("primary_answer", ""))]
    candidates.extend(
        str(card.get("answer", ""))
        for card in record.get("candidate_cards", [])
        if isinstance(card, Mapping)
    )
    candidates = [candidate for candidate in candidates if candidate.strip()]
    if not candidates:
        return "unknown"
    return max(
        enumerate(candidates),
        key=lambda item: (*_score(item[1], references)[:2], -item[0]),
    )[1]


def _answer(record: Mapping, field: str, references: Sequence[str]) -> str:
    if field == "oracle_candidate":
        return _oracle_candidate(record, references)
    value = _field(record, field)
    return str(value).strip() if value is not None else "unknown"


def evaluate_identifiability(
    examples: Sequence[PreparedExample],
    references: Mapping[str, Sequence[str]],
    conflict_labels: Mapping[str, bool],
    systems: Mapping[str, tuple[Mapping[str, Mapping], str]],
    *,
    primary_name: str,
    no_context_name: str | None = None,
    resolver_name: str | None = None,
    oracle_name: str | None = None,
) -> tuple[dict, list[dict]]:
    if primary_name not in systems:
        raise ValueError(f"Unknown primary system: {primary_name}")
    for role, name in {
        "no-context": no_context_name,
        "resolver": resolver_name,
        "oracle": oracle_name,
    }.items():
        if name is not None and name not in systems:
            raise ValueError(f"Unknown {role} system: {name}")
    expected_ids = {example.question_id for example in examples}
    missing_references = sorted(expected_ids - set(references))
    if missing_references:
        raise ValueError(
            "Missing evaluation references for question IDs: "
            + ", ".join(missing_references[:5])
        )

    rows: list[dict] = []
    for example in examples:
        question_id = example.question_id
        item_references = [str(value) for value in references.get(question_id, ())]
        row: dict = {
            "question_id": question_id,
            "has_conflict": bool(conflict_labels.get(question_id, False)),
            "references": item_references,
        }
        for name, (records, field) in systems.items():
            record = records.get(question_id, {})
            answer = _answer(record, field, item_references)
            exact, f1, abstained = _score(answer, item_references)
            row[name] = {
                "answer": None if abstained else answer,
                "exact": exact,
                "token_f1": f1,
                "abstained": abstained,
                "missing_prediction": question_id not in records,
            }
        rows.append(row)

    def metrics(selected: Sequence[Mapping], name: str) -> dict:
        covered = [row for row in selected if not row[name]["abstained"]]
        return {
            "examples": len(selected),
            "answer_exact": float(np.mean([row[name]["exact"] for row in selected]))
            if selected
            else 0.0,
            "answer_token_f1": float(
                np.mean([row[name]["token_f1"] for row in selected])
            )
            if selected
            else 0.0,
            "coverage": len(covered) / len(selected) if selected else 0.0,
            "missing_predictions": sum(row[name]["missing_prediction"] for row in selected),
        }

    slices = {
        "overall": rows,
        "qacc_conflict": [row for row in rows if row["has_conflict"]],
        "qacc_non_conflict": [row for row in rows if not row["has_conflict"]],
    }

    def comparison(selected: Sequence[Mapping], name: str) -> dict:
        changed = [
            row
            for row in selected
            if qacc_normalize_text(row[primary_name]["answer"])
            != qacc_normalize_text(row[name]["answer"])
        ]
        helped = sum(
            row[primary_name]["exact"] == 0 and row[name]["exact"] == 1
            for row in selected
        )
        hurt = sum(
            row[primary_name]["exact"] == 1 and row[name]["exact"] == 0
            for row in selected
        )
        primary_em = metrics(selected, primary_name)["answer_exact"]
        method_em = metrics(selected, name)["answer_exact"]
        return {
            "delta_exact": method_em - primary_em,
            "changed": len(changed),
            "helped": helped,
            "hurt": hurt,
            "net_exact_changes": helped - hurt,
            "switch_precision": helped / (helped + hurt) if helped + hurt else None,
        }

    summary = {
        "protocol": "closed_context_answer_identifiability",
        "protocol_version": "1.0",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "roles": {
            "primary": primary_name,
            "no_context": no_context_name,
            "resolver": resolver_name,
            "oracle": oracle_name,
        },
        "metrics": {
            slice_name: {name: metrics(selected, name) for name in systems}
            for slice_name, selected in slices.items()
        },
        "comparisons_to_primary": {
            slice_name: {
                name: comparison(selected, name)
                for name in systems
                if name != primary_name
            }
            for slice_name, selected in slices.items()
        },
        "identifiability_gaps": {},
    }
    for slice_name in slices:
        slice_metrics = summary["metrics"][slice_name]
        gaps = {}
        if no_context_name is not None:
            gaps["context_gain_over_no_context"] = (
                slice_metrics[primary_name]["answer_exact"]
                - slice_metrics[no_context_name]["answer_exact"]
            )
        if resolver_name is not None:
            gaps["resolver_gain_over_primary"] = (
                slice_metrics[resolver_name]["answer_exact"]
                - slice_metrics[primary_name]["answer_exact"]
            )
        if oracle_name is not None:
            gaps["oracle_minus_primary"] = (
                slice_metrics[oracle_name]["answer_exact"]
                - slice_metrics[primary_name]["answer_exact"]
            )
            if resolver_name is not None:
                gaps["oracle_minus_resolver"] = (
                    slice_metrics[oracle_name]["answer_exact"]
                    - slice_metrics[resolver_name]["answer_exact"]
                )
        summary["identifiability_gaps"][slice_name] = gaps
    return summary, rows


def write_identifiability_report(path: Path, summary: Mapping, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    rows_path = path.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    system_names = list(summary["metrics"]["overall"])
    lines = [
        "# Closed-context answer identifiability",
        "",
        "All scores use the released QACC normalized EM/F1 evaluator.",
        "",
        "| Slice | System | n | EM | Token F1 | Coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for slice_name in ("overall", "qacc_conflict", "qacc_non_conflict"):
        for name in system_names:
            item = summary["metrics"][slice_name][name]
            lines.append(
                f"| {slice_name} | {name} | {item['examples']} | "
                f"{item['answer_exact']:.4f} | {item['answer_token_f1']:.4f} | "
                f"{item['coverage']:.4f} |"
            )
    lines.extend(["", "## Identifiability gaps", ""])
    for slice_name, gaps in summary["identifiability_gaps"].items():
        lines.append(
            f"- {slice_name}: "
            + ", ".join(f"{key}={value:+.4f}" for key, value in gaps.items())
        )
    lines.extend(
        [
            "",
            "These empirical gaps diagnose the selected systems under the frozen QACC "
            "protocol; they do not constitute a mathematical proof of non-identifiability.",
            "",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "evaluate_identifiability",
    "read_jsonl_records",
    "write_identifiability_report",
]
