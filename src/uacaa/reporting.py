from __future__ import annotations

import json
from pathlib import Path


METHOD_ORDER = (
    "direct_rag",
    "qacc_explain_then_answer",
    "raw_passage_majority",
    "independent_source_majority",
    "non_bayesian_conditional_logit",
    "full_uacaa",
    "selected_map_no_dependency",
    "no_dependency_correction",
    "no_support_features",
    "no_evidence_quality_features",
    "no_source_category_features",
    "no_temperature_calibration",
    "no_abstention",
    "no_label_smoothing",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def merge_dev_results(
    candidate_experiments: Path,
    direct_rag_summary: Path,
    qacc_summary: Path,
    output_path: Path,
) -> dict:
    result = _load(candidate_experiments)
    methods = result["methods"]
    methods["direct_rag"] = {
        "role": "baseline",
        "metric_scope": "direct answer against noisy QACC train/dev correctAnswer",
        "metrics": _load(direct_rag_summary),
    }
    methods["qacc_explain_then_answer"] = {
        "role": "baseline",
        "metric_scope": "direct answer against noisy QACC train/dev correctAnswer",
        "metrics": _load(qacc_summary),
    }
    result["pending_llm_baselines"] = []
    result["metric_notes"] = {
        "answer_exact": "normalized exact string equality over all dev questions",
        "answer_token_f1": "maximum token F1 against the dev weak answer label",
        "multicandidate_choice_accuracy": (
            "candidate methods only; accuracy conditional on a recalled gold candidate "
            "and at least two extracted candidates"
        ),
        "multiclass_brier": (
            "candidate methods only; standard summed multiclass Brier conditional on "
            "a recalled gold candidate"
        ),
        "coverage": "fraction of all dev questions receiving a non-abstained answer",
        "selective_answer_exact": "strict answer EM among covered questions",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    for name in METHOD_ORDER:
        if name not in methods:
            continue
        entry = methods[name]
        metrics = entry["metrics"]
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    entry["role"],
                    _format(metrics.get("answer_exact")),
                    _format(metrics.get("answer_token_f1")),
                    _format(metrics.get("multicandidate_choice_accuracy")),
                    _format(metrics.get("multiclass_brier")),
                    _format(metrics.get("coverage")),
                    _format(
                        metrics.get(
                            "selective_answer_exact",
                            metrics.get("selective_exact"),
                        )
                    ),
                ]
            )
            + " |"
        )
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(
        "\n".join(
            [
                "# QACC dev baselines and ablations",
                "",
                "These are train/dev development results against QACC's noisy `correctAnswer` field, not official test estimates.",
                "",
                "| Method | Role | EM | Token F1 | Multi-candidate accuracy | Brier ↓ | Coverage | Selective EM |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
                *rows,
                "",
                "Direct-answer baselines do not expose candidate probabilities, so multi-candidate accuracy and Brier are not applicable.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"json": str(output_path), "markdown": str(markdown_path), "methods": len(methods)}


__all__ = ["merge_dev_results"]
