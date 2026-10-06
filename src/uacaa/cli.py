from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .answer_baselines import (
    OpenAIAnswerBaseline,
    evaluate_answer_cache,
    read_answer_cache,
    write_answer_cache,
)
from .extraction import (
    EXTRACTION_PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    OpenAICompatibleExtractor,
    read_extraction_cache,
    write_extraction_cache,
)
from .identifiability import (
    evaluate_identifiability,
    read_jsonl_records,
    write_identifiability_report,
)
from .protocol import prepare_protocol, read_evaluation_metadata, read_prepared
from .qacc import audit_rows, file_sha256, load_raw_qacc
from .uncertainty_routing import (
    evaluate_frozen_uncertainty_routing_predictions,
    run_uncertainty_routing_train_dev,
    write_frozen_uncertainty_routing_predictions,
)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = str(record["question_id"])
            if question_id in records:
                raise ValueError(f"Duplicate question_id in {path}: {question_id}")
            records[question_id] = record
    return records


def _reasoning_effort(value: str) -> str | None:
    return None if value == "omit" else value


def command_audit(args: argparse.Namespace) -> None:
    report = audit_rows(load_raw_qacc(args.raw))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print(report)


def command_prepare(args: argparse.Namespace) -> None:
    _print(
        prepare_protocol(
            args.raw,
            args.output_dir,
            revision=args.revision,
            ambigqa_paths=args.ambigqa,
        )
    )


def command_extract(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    extractor = OpenAICompatibleExtractor(
        args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        structured_output=not args.json_mode,
        reasoning_effort=_reasoning_effort(args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "model": args.model,
        "endpoint": args.endpoint,
        "structured_output": not args.json_mode,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing extraction cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            key for key, value in identity.items()
            if previous.get(key, value) != value
            and not (
                key == "extraction_protocol_version"
                and previous.get(key) == "3.1"
                and value == "3.2"
            )
        ]
        if mismatches:
            raise ValueError(
                "Cannot resume extraction with different provenance: "
                + ", ".join(mismatches)
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {**identity, "offset": args.offset, "limit": args.limit, "status": "running"},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    progress = write_extraction_cache(
        examples, extractor, args.output, resume=not args.no_resume
    )
    manifest_path.write_text(
        json.dumps(
            {
                **identity,
                "offset": args.offset,
                "limit": args.limit,
                "progress": progress,
                "status": "complete",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _print({**progress, "output": str(args.output), "manifest": str(manifest_path)})


def command_answer_baseline(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    predictor = OpenAIAnswerBaseline(
        args.model,
        args.style,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        reasoning_effort=_reasoning_effort(args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "model": args.model,
        "style": args.style,
        "endpoint": args.endpoint,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt_sha256": hashlib.sha256(
            predictor.system_prompt.encode()
        ).hexdigest(),
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing answer cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in identity.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(
                "Cannot resume answer cache with different provenance: "
                + ", ".join(mismatches)
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {**identity, "offset": args.offset, "limit": args.limit, "status": "running"},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    progress = write_answer_cache(
        examples, predictor, args.output, resume=not args.no_resume
    )
    manifest_path.write_text(
        json.dumps(
            {
                **identity,
                "offset": args.offset,
                "limit": args.limit,
                "progress": progress,
                "status": "complete",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _print({**progress, "output": str(args.output), "manifest": str(manifest_path)})


def command_evaluate_answer(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    references = None
    metadata = None
    if any(example.split == "test" for example in examples):
        if args.references is None:
            raise ValueError("Test evaluation requires --references")
        metadata = read_evaluation_metadata(args.references)
        references = {
            question_id: item["references"] for question_id, item in metadata.items()
        }
    summary = evaluate_answer_cache(
        examples,
        read_answer_cache(args.predictions),
        references=references,
        evaluation_metadata=metadata,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def command_arsr_train_dev(args: argparse.Namespace) -> None:
    summary = run_uncertainty_routing_train_dev(
        read_prepared(args.train_data),
        read_prepared(args.dev_data),
        read_answer_cache(args.train_primary_predictions),
        read_answer_cache(args.train_secondary_predictions),
        read_answer_cache(args.dev_primary_predictions),
        read_answer_cache(args.dev_secondary_predictions),
        read_extraction_cache(args.train_extractions) if args.train_extractions else {},
        read_extraction_cache(args.dev_extractions) if args.dev_extractions else {},
        primary_name=args.primary_name,
        secondary_name=args.secondary_name,
        prior_scale=args.prior_scale,
        intervention_quantile=args.intervention_quantile,
        delta_margin=args.delta_margin,
        selective_coverage=args.selective_coverage,
        pre_feature_mode=args.pre_feature_mode,
        seed=args.seed,
        output_path=args.output,
    )
    input_paths = {
        "train_data": args.train_data,
        "dev_data": args.dev_data,
        "train_primary_predictions": args.train_primary_predictions,
        "train_secondary_predictions": args.train_secondary_predictions,
        "dev_primary_predictions": args.dev_primary_predictions,
        "dev_secondary_predictions": args.dev_secondary_predictions,
        "train_extractions": args.train_extractions,
        "dev_extractions": args.dev_extractions,
    }
    manifest = {
        "protocol": summary["protocol"],
        "protocol_version": summary["protocol_version"],
        "configuration": summary["configuration"],
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
            if path is not None
        },
        "outputs": {
            "rows": str(args.output.resolve()),
            "summary": str(args.output.with_suffix(".summary.json").resolve()),
            "model": str(args.output.with_suffix(".model.json").resolve()),
        },
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print({"output": str(args.output), "manifest": str(manifest_path), "metrics": summary})


def command_arsr_predict(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(f"Frozen prediction output already exists: {args.output}")
    model = json.loads(args.model.read_text(encoding="utf-8"))
    progress = write_frozen_uncertainty_routing_predictions(
        read_prepared(args.data),
        read_answer_cache(args.primary_predictions),
        read_answer_cache(args.secondary_predictions),
        read_extraction_cache(args.extractions),
        model,
        args.output,
    )
    inputs = {
        "data": args.data,
        "primary_predictions": args.primary_predictions,
        "secondary_predictions": args.secondary_predictions,
        "extractions": args.extractions,
        "model": args.model,
    }
    manifest = {
        "protocol": "frozen_official_test_prediction_without_references",
        "protocol_version": model["protocol_version"],
        "configuration": model["configuration"],
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in inputs.items()
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": file_sha256(args.output),
        },
        "progress": progress,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print({**progress, "manifest": str(manifest_path)})


def command_arsr_evaluate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(f"Scored output already exists: {args.output}")
    summary = evaluate_frozen_uncertainty_routing_predictions(
        read_prepared(args.data),
        _read_jsonl(args.predictions),
        read_evaluation_metadata(args.references),
        seed=args.seed,
        output_path=args.output,
    )
    inputs = {
        "data": args.data,
        "predictions": args.predictions,
        "references": args.references,
    }
    manifest = {
        "protocol": summary["protocol"],
        "protocol_version": summary["protocol_version"],
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in inputs.items()
        },
        "outputs": {
            "scored_rows": str(args.output.resolve()),
            "summary": str(args.output.with_suffix(".summary.json").resolve()),
        },
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print({"output": str(args.output), "manifest": str(manifest_path), "metrics": summary})


def command_identifiability_report(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    metadata = read_evaluation_metadata(args.references)
    references = {
        question_id: item["references"] for question_id, item in metadata.items()
    }
    conflict_labels = {
        question_id: bool(item.get("has_conflict", False))
        for question_id, item in metadata.items()
    }
    systems: dict[str, tuple[dict[str, dict], str]] = {}
    for name, path_text, field in args.prediction:
        if name in systems:
            raise ValueError(f"Duplicate prediction system name: {name}")
        systems[name] = (read_jsonl_records(Path(path_text)), field)
    summary, rows = evaluate_identifiability(
        examples,
        references,
        conflict_labels,
        systems,
        primary_name=args.primary_name,
        no_context_name=args.no_context_name,
        resolver_name=args.resolver_name,
        oracle_name=args.oracle_name,
    )
    write_identifiability_report(args.output, summary, rows)
    _print(
        {
            "output": str(args.output),
            "rows": str(args.output.with_suffix(".rows.jsonl")),
            "markdown": str(args.output.with_suffix(".md")),
            "metrics": summary,
        }
    )


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "omit"),
        default="none",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uacaa")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Audit the untouched QACC file")
    audit.add_argument("--raw", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(function=command_audit)

    prepare = commands.add_parser("prepare", help="Create leakage-safe QACC splits")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--revision")
    prepare.add_argument("--ambigqa", type=Path, nargs="+", default=[])
    prepare.set_defaults(function=command_prepare)

    extract = commands.add_parser("extract", help="Extract one answer per context")
    extract.add_argument("--data", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--offset", type=int, default=0)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--no-resume", action="store_true")
    extract.add_argument("--json-mode", action="store_true")
    _add_api_arguments(extract)
    extract.set_defaults(function=command_extract)

    baseline = commands.add_parser(
        "answer-baseline", help="Run Direct-RAG or the QACC reference prompt"
    )
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--model", required=True)
    baseline.add_argument(
        "--style",
        choices=("direct-rag", "qacc-explain-then-answer", "no-context"),
        required=True,
    )
    baseline.add_argument("--offset", type=int, default=0)
    baseline.add_argument("--limit", type=int)
    baseline.add_argument("--no-resume", action="store_true")
    _add_api_arguments(baseline)
    baseline.set_defaults(function=command_answer_baseline)

    evaluate = commands.add_parser(
        "evaluate-answer-baseline", help="Evaluate a cached answer baseline"
    )
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--references", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(function=command_evaluate_answer)

    train = commands.add_parser(
        "arsr-train-dev", help="Fit ARSR on train and calibrate it on dev"
    )
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--dev-data", type=Path, required=True)
    train.add_argument("--train-primary-predictions", type=Path, required=True)
    train.add_argument("--train-secondary-predictions", type=Path, required=True)
    train.add_argument("--dev-primary-predictions", type=Path, required=True)
    train.add_argument("--dev-secondary-predictions", type=Path, required=True)
    train.add_argument("--train-extractions", type=Path)
    train.add_argument("--dev-extractions", type=Path)
    train.add_argument("--primary-name", required=True)
    train.add_argument("--secondary-name", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--prior-scale", type=float, default=0.75)
    train.add_argument("--intervention-quantile", type=float, default=0.75)
    train.add_argument("--delta-margin", type=float, default=0.10)
    train.add_argument("--selective-coverage", type=float, default=0.80)
    train.add_argument(
        "--pre-feature-mode",
        choices=("full", "no-source-time", "answer-only"),
        default="answer-only",
    )
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(function=command_arsr_train_dev)

    predict = commands.add_parser(
        "arsr-predict", help="Apply a frozen ARSR model without test references"
    )
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--primary-predictions", type=Path, required=True)
    predict.add_argument("--secondary-predictions", type=Path, required=True)
    predict.add_argument("--extractions", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.set_defaults(function=command_arsr_predict)

    arsr_evaluate = commands.add_parser(
        "arsr-evaluate", help="Evaluate frozen ARSR predictions"
    )
    arsr_evaluate.add_argument("--data", type=Path, required=True)
    arsr_evaluate.add_argument("--predictions", type=Path, required=True)
    arsr_evaluate.add_argument("--references", type=Path, required=True)
    arsr_evaluate.add_argument("--output", type=Path, required=True)
    arsr_evaluate.add_argument("--seed", type=int, default=42)
    arsr_evaluate.set_defaults(function=command_arsr_evaluate)

    report = commands.add_parser(
        "identifiability-report", help="Build the final QACC comparison"
    )
    report.add_argument("--data", type=Path, required=True)
    report.add_argument("--references", type=Path, required=True)
    report.add_argument(
        "--prediction",
        nargs=3,
        action="append",
        metavar=("NAME", "JSONL", "ANSWER_FIELD"),
        required=True,
    )
    report.add_argument("--primary-name", required=True)
    report.add_argument("--no-context-name")
    report.add_argument("--resolver-name")
    report.add_argument("--oracle-name")
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(function=command_identifiability_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
