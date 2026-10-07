"""Build data files without test-answer leakage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .qacc import (
    _question_key,
    audit_rows,
    evaluation_references,
    file_sha256,
    load_ambigqa_references,
    load_raw_qacc,
    prepare_example,
)
from .schemas import PreparedExample


STRICT_EXCLUSION_FLAGS = {
    "truncated_fourth_answer",
    "candidate_without_context",
    "no_answer_candidate",
    "correct_answer_unmapped",
    "duplicate_candidate_answer",
}


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    # Write a temp file first, then replace.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_prepared(path: Path) -> list[PreparedExample]:
    """Read the project JSONL format into typed examples."""

    examples: list[PreparedExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(PreparedExample.from_dict(json.loads(line)))
    return examples


def read_references(path: Path) -> dict[str, tuple[str, ...]]:
    """Load hidden evaluation answers by question id."""

    references: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                references[str(record["question_id"])] = tuple(
                    str(value) for value in record.get("references", [])
                )
    return references


def read_evaluation_metadata(path: Path) -> dict[str, dict]:
    """Load references and conflict labels used only for scoring."""

    metadata: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                metadata[str(record["question_id"])] = {
                    "references": tuple(
                        str(value) for value in record.get("references", [])
                    ),
                    "has_conflict": record.get("has_conflict"),
                }
    return metadata


def _test_input(example: PreparedExample) -> PreparedExample:
    # Remove labels before test prediction.
    return PreparedExample(
        question_id=example.question_id,
        split=example.split,
        question=example.question,
        contexts=example.contexts,
        supervision=None,
        evaluation=None,
        quality_flags=example.quality_flags,
    )


def _write_test_pair(prefix: Path, examples: list[PreparedExample]) -> None:
    """Write separate public inputs and private evaluation labels."""

    write_jsonl(
        Path(str(prefix) + ".inputs.jsonl"),
        (_test_input(item).to_dict() for item in examples),
    )
    write_jsonl(
        Path(str(prefix) + ".references.jsonl"),
        (
            {
                "question_id": item.question_id,
                "references": list((item.evaluation or {}).get("references", [])),
                "has_conflict": (item.evaluation or {}).get("has_conflict"),
            }
            for item in examples
        ),
    )


def is_strict_training_example(example: PreparedExample) -> bool:
    """Accept only labelled train/dev rows without known data faults."""

    return (
        example.split in {"train", "dev"}
        and not (set(example.quality_flags) & STRICT_EXCLUSION_FLAGS)
        and example.supervision is not None
    )


def prepare_protocol(
    raw_path: Path,
    output_dir: Path,
    *,
    revision: str | None = None,
    ambigqa_paths: Iterable[Path] | None = None,
) -> dict:
    """Create all frozen train, dev, and test protocol files."""

    # Raw QACC stays unchanged.
    rows = load_raw_qacc(raw_path)
    ambigqa_files = list(ambigqa_paths or [])
    # AmbigQA aliases improve scoring but never enter an inference prompt.
    upstream = load_ambigqa_references(ambigqa_files) if ambigqa_files else {}
    examples = [
        prepare_example(
            row,
            position,
            upstream_references=upstream.get(_question_key(row.get("question")), ()),
        )
        for position, row in enumerate(rows)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Weak keeps all rows; strict removes rows with annotation defects.
    for split in ("train", "dev"):
        weak = [example for example in examples if example.split == split]
        strict = [example for example in weak if is_strict_training_example(example)]
        write_jsonl(output_dir / f"{split}.weak.jsonl", (item.to_dict() for item in weak))
        write_jsonl(output_dir / f"{split}.strict.jsonl", (item.to_dict() for item in strict))

    test = [example for example in examples if example.split == "test"]
    # The official test retains the exact rows and quality flags from QACC.
    _write_test_pair(output_dir / "test.official", test)

    audit = audit_rows(rows)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reference_coverage = {}
    for split in ("train", "dev", "test"):
        selected = [item for item in examples if item.split == split]
        with_multiple = sum(len(evaluation_references(item)) > 1 for item in selected)
        restored = sum(bool((item.supervision or {}).get("references")) for item in selected)
        reference_coverage[split] = {
            "examples": len(selected),
            "restored_from_ambigqa": restored,
            "multiple_references": with_multiple,
        }

    # Record hashes and label boundaries so the study can be reproduced.
    manifest = {
        "protocol_version": "1.2",
        "raw_path": str(raw_path.resolve()),
        "raw_sha256": file_sha256(raw_path),
        "upstream_revision": revision,
        "ambigqa_reference_files": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in ambigqa_files
        ],
        "reference_coverage": reference_coverage,
        "raw_rows": len(rows),
        "strict_exclusion_flags": sorted(STRICT_EXCLUSION_FLAGS),
        "files": {
            "train_weak": "train.weak.jsonl",
            "train_strict": "train.strict.jsonl",
            "dev_weak": "dev.weak.jsonl",
            "dev_strict": "dev.strict.jsonl",
            "test_official_inputs": "test.official.inputs.jsonl",
            "test_official_references": "test.official.references.jsonl",
            "audit": "audit.json",
        },
        "leakage_policy": {
            "inference_inputs": ["question", "contexts.text", "contexts.source"],
            "train_dev_only": [
                "supervision",
                "supervision.has_conflict (QACC secondAnswerExist)",
            ],
            "evaluation_only": [
                "supervision.references (restored AmbigQA aliases; never model input)",
                "evaluation.references",
                "evaluation.has_conflict",
            ],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
