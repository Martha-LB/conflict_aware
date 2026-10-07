"""Compare final systems with the official QACC score."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .clustering import cluster_observations
from .normalization import is_no_answer, qacc_exact_match, qacc_normalize_text, qacc_token_f1
from .schemas import EvidenceObservation, PreparedExample


EVALUATION_PROTOCOL = "qacc_public_normalized_em_f1_max_over_references"


def _paired_exact_p(helped: int, hurt: int) -> float:
    """Two-sided exact McNemar test for paired binary outcomes."""

    discordant = helped + hurt
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(helped, hurt) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _bootstrap_interval(differences: np.ndarray, seed: int = 42) -> list[float]:
    """Paired bootstrap interval for a mean EM difference."""

    if not len(differences):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(5000, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def read_jsonl_records(path: Path) -> dict[str, dict]:
    """Read unique prediction records keyed by question id."""

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


def write_candidate_pool(
    examples: Sequence[PreparedExample],
    primary_predictions: Mapping[str, Mapping],
    secondary_predictions: Mapping[str, Mapping],
    extractions: Mapping[str, Sequence[EvidenceObservation]],
    output_path: Path,
) -> dict:
    """Save candidates that are available without reading gold references."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            primary = str(
                primary_predictions.get(example.question_id, {}).get("answer", "unknown")
            ).strip()
            secondary = str(
                secondary_predictions.get(example.question_id, {}).get("answer", "unknown")
            ).strip()

            # Context answers are grouped before they enter the oracle pool.
            clusters = cluster_observations(extractions.get(example.question_id, ()))
            cards: list[dict] = []
            seen: set[str] = set()
            for answer, origin in [
                (primary, "primary_model"),
                (secondary, "secondary_model"),
                *((cluster.answer, "context_extraction") for cluster in clusters),
            ]:
                key = qacc_normalize_text(answer)
                if not key or is_no_answer(answer) or key in seen:
                    continue
                seen.add(key)
                cards.append(
                    {
                        "candidate_id": f"C{len(cards) + 1}",
                        "answer": answer,
                        "origin": origin,
                    }
                )
            record = {
                "question_id": example.question_id,
                "primary_answer": primary or "unknown",
                "candidate_cards": cards,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return {"examples": len(examples), "output": str(output_path)}


def _field(record: Mapping, field: str) -> object:
    """Read a nested value from a dot-separated field name."""

    value: object = record
    for key in field.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _score(answer: object, references: Sequence[str]) -> tuple[float, float, bool]:
    """Return QACC EM, F1, and abstention for one answer."""

    text = str(answer or "").strip()
    abstained = not text or is_no_answer(text)
    if abstained:
        return 0.0, 0.0, True
    exact = float(any(qacc_exact_match(text, reference) for reference in references))
    f1 = max((qacc_token_f1(text, reference) for reference in references), default=0.0)
    return exact, f1, False


def _oracle_candidate(record: Mapping, references: Sequence[str]) -> str:
    # Gold references make this an upper bound only.
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
    """Read a normal answer field or compute the oracle candidate."""

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
    """Measure what evidence exists and what each resolver can identify."""

    # Keep every system score on the same question row.
    if primary_name not in systems:
        raise ValueError(f"Unknown primary system: {primary_name}")
    for role, name in {
        "no-context": no_context_name,
        "resolver": resolver_name,
        "oracle": oracle_name,
    }.items():
        if name is not None and name not in systems:
            raise ValueError(f"Unknown {role} system: {name}")
    # Missing references make the main outcome impossible to score.
    expected_ids = {example.question_id for example in examples}
    missing_references = sorted(expected_ids - set(references))
    if missing_references:
        raise ValueError(
            "Missing evaluation references for question IDs: "
            + ", ".join(missing_references[:5])
        )

    rows: list[dict] = []
    # Store all system outcomes on the same row for paired comparisons.
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
        """Summarize one system on one fixed data slice."""

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

    # Conflict and non-conflict rows answer different parts of the RQ.
    slices = {
        "overall": rows,
        "qacc_conflict": [row for row in rows if row["has_conflict"]],
        "qacc_non_conflict": [row for row in rows if not row["has_conflict"]],
    }

    def pairwise_comparison(
        selected: Sequence[Mapping], left: str, right: str
    ) -> dict:
        """Compare the right system with the left system on paired rows."""

        changed = [
            row
            for row in selected
            if qacc_normalize_text(row[left]["answer"])
            != qacc_normalize_text(row[right]["answer"])
        ]
        helped = sum(
            row[left]["exact"] == 0 and row[right]["exact"] == 1
            for row in selected
        )
        hurt = sum(
            row[left]["exact"] == 1 and row[right]["exact"] == 0
            for row in selected
        )
        differences = np.asarray(
            [row[right]["exact"] - row[left]["exact"] for row in selected],
            dtype=float,
        )
        return {
            "left": left,
            "right": right,
            "delta_exact": float(differences.mean()) if len(differences) else 0.0,
            "changed": len(changed),
            "helped": helped,
            "hurt": hurt,
            "net_exact_changes": helped - hurt,
            "switch_precision": helped / (helped + hurt) if helped + hurt else None,
            "bootstrap_95pct_em_delta": _bootstrap_interval(differences),
            "mcnemar_exact_p_two_sided": _paired_exact_p(helped, hurt),
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
                name: pairwise_comparison(selected, primary_name, name)
                for name in systems
                if name != primary_name
            }
            for slice_name, selected in slices.items()
        },
        "pairwise_comparisons": {
            slice_name: {
                f"{right}_vs_{left}": pairwise_comparison(selected, left, right)
                for left_index, left in enumerate(systems)
                for right in list(systems)[left_index + 1 :]
            }
            for slice_name, selected in slices.items()
        },
        "identifiability_gaps": {},
    }
    # Oracle gaps separate candidate availability from answer selection.
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
    """Save machine-readable results and a short human-readable table."""

    # Save JSON data and one short Markdown table.
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
    lines.extend(
        [
            "",
            "## Overall paired EM comparisons",
            "",
            "Positive delta means the right system is better.",
            "",
            "| Comparison | EM delta | 95% bootstrap interval | McNemar p |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in summary["pairwise_comparisons"]["overall"].items():
        low, high = item["bootstrap_95pct_em_delta"]
        lines.append(
            f"| {name} | {item['delta_exact']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | "
            f"{item['mcnemar_exact_p_two_sided']:.4f} |"
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
    "write_candidate_pool",
    "write_identifiability_report",
]
