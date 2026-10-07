"""Command line entry for the soft-MaxEnt QACC experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .answer_baselines import (
    OpenAIAnswerBaseline,
    OpenAIWeightedAnswerer,
    evaluate_answer_cache,
    read_answer_cache,
    write_answer_cache,
)
from .context_maxent import (
    CONTEXT_MAXENT_PROTOCOL_VERSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NLI_MODEL,
    SentenceTransformersRuntime,
    write_context_maxent_cache,
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
    write_candidate_pool,
    write_identifiability_report,
)
from .protocol import prepare_protocol, read_evaluation_metadata, read_prepared
from .qacc import audit_rows, file_sha256, load_raw_qacc
from .soft_maxent import read_weight_cache


# Small helpers used by all commands.
def _print(value: object) -> None:
    """Print stable, readable JSON for shell users."""

    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _provenance_mismatches(previous: dict, current: dict) -> list[str]:
    """Compare cache settings while accepting moved, byte-identical inputs."""

    path_hashes = {
        "input": "input_sha256",
        "weights": "weights_sha256",
        "extractions": "extractions_sha256",
    }
    mismatches: list[str] = []
    for key, value in current.items():
        hash_key = path_hashes.get(key)
        if hash_key and previous.get(hash_key) == current.get(hash_key):
            # A path is only descriptive when its content hash is unchanged.
            continue
        if previous.get(key) != value:
            mismatches.append(key)
    return mismatches


def _reasoning_effort(value: str) -> str | None:
    """Translate the CLI word 'omit' to an absent API field."""

    return None if value == "omit" else value


def command_audit(args: argparse.Namespace) -> None:
    """Inspect the untouched QACC release and save its data report."""

    report = audit_rows(load_raw_qacc(args.raw))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print(report)


def command_prepare(args: argparse.Namespace) -> None:
    """Create the leakage-safe files used by all later commands."""

    _print(
        prepare_protocol(
            args.raw,
            args.output_dir,
            revision=args.revision,
            ambigqa_paths=args.ambigqa,
        )
    )


def command_extract(args: argparse.Namespace) -> None:
    """Extract per-context answers with resumable API calls."""

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
    # Provenance prevents an old cache from being reused with new settings.
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
    # A resume is valid only when all output-changing inputs still match.
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing extraction cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            key
            for key in _provenance_mismatches(previous, identity)
            if not (
                key == "extraction_protocol_version"
                and previous.get(key) in {None, "3.1"}
                and identity[key] == "3.2"
            )
        ]
        if mismatches:
            raise ValueError(
                "Cannot resume extraction with different provenance: "
                + ", ".join(mismatches)
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Mark the run before API calls, then replace this state on completion.
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
    """Run one frozen answer prompt and cache every response."""

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
    # The prompt hash makes prompt changes visible in saved results.
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
    # Refuse to mix models, prompts, or input versions in one cache.
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing answer cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = _provenance_mismatches(previous, identity)
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
    """Score an answer cache and keep test references out of generation."""

    examples = read_prepared(args.data)
    references = None
    metadata = None
    # Official test answers live in a separate file by design.
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


def command_maxent_build(args: argparse.Namespace) -> None:
    """Create all three passage-weight conditions without references."""

    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    identity = {
        "protocol": "notebook_aligned_context_soft_maxent",
        "protocol_version": CONTEXT_MAXENT_PROTOCOL_VERSION,
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "extractions": str(args.extractions.resolve()),
        "extractions_sha256": file_sha256(args.extractions),
        "embedding_model": args.embedding_model,
        "nli_model": args.nli_model,
        "uniform_mix": args.uniform_mix,
        "cluster_floor": args.cluster_floor,
        "slack_penalty": args.slack_penalty,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "offset": args.offset,
        "limit": args.limit,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    had_cache = args.output.exists() or temporary_path.exists()
    if not args.no_resume and had_cache:
        if not manifest_path.exists():
            raise ValueError("Existing MaxEnt cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = _provenance_mismatches(previous, identity)
        if mismatches:
            raise ValueError(
                "Cannot resume MaxEnt cache with different provenance: "
                + ", ".join(mismatches)
            )
        cached = read_weight_cache(args.output) if args.output.exists() else {}
        if all(example.question_id in cached for example in examples):
            _print(
                {
                    "requested": len(examples),
                    "reused": len(examples),
                    "extracted": 0,
                    "output": str(args.output),
                    "manifest": str(manifest_path),
                }
            )
            return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {**identity, "status": "running"},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = SentenceTransformersRuntime(
        embedding_model=args.embedding_model,
        nli_model=args.nli_model,
        device=None if args.device == "auto" else args.device,
        batch_size=args.batch_size,
    )
    progress = write_context_maxent_cache(
        examples,
        read_extraction_cache(args.extractions),
        runtime,
        args.output,
        resume=not args.no_resume,
        uniform_mix=args.uniform_mix,
        cluster_floor=args.cluster_floor,
        slack_penalty=args.slack_penalty,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
    )
    manifest_path.write_text(
        json.dumps(
            {
                **identity,
                "output_sha256": file_sha256(args.output),
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
    _print({**progress, "manifest": str(manifest_path)})


def command_answer_weighted(args: argparse.Namespace) -> None:
    """Answer with frozen uniform, support-voting, or MaxEnt weights."""

    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    weights = read_weight_cache(args.weights)
    predictor = OpenAIWeightedAnswerer(
        args.model,
        weights,
        args.condition,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        reasoning_effort=_reasoning_effort(args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "weights": str(args.weights.resolve()),
        "weights_sha256": file_sha256(args.weights),
        "model": args.model,
        "condition": args.condition,
        "endpoint": args.endpoint,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt_sha256": hashlib.sha256(
            predictor.system_prompt.encode()
        ).hexdigest(),
    }
    if bool(args.reuse_equivalent_answers) != bool(args.reuse_equivalent_weights):
        raise ValueError(
            "--reuse-equivalent-answers and --reuse-equivalent-weights must be used together"
        )
    seed_records: dict[str, dict] = {}
    if args.reuse_equivalent_answers:
        source_manifest_path = Path(
            str(args.reuse_equivalent_answers) + ".manifest.json"
        )
        if not source_manifest_path.exists():
            raise ValueError("Equivalent-answer source has no provenance manifest")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_weights_hash = file_sha256(args.reuse_equivalent_weights)
        required = {
            "input_sha256": identity["input_sha256"],
            "model": identity["model"],
            "endpoint": identity["endpoint"],
            "reasoning_effort": identity["reasoning_effort"],
            "system_prompt_sha256": identity["system_prompt_sha256"],
            "weights_sha256": source_weights_hash,
        }
        mismatches = [
            key for key, value in required.items() if source_manifest.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Equivalent-answer source has different provenance: "
                + ", ".join(mismatches)
            )
        source_weights = read_weight_cache(args.reuse_equivalent_weights)
        source_answers = read_answer_cache(args.reuse_equivalent_answers)
        source_predictor = OpenAIWeightedAnswerer(
            args.model,
            source_weights,
            str(source_manifest["condition"]),
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            retries=args.retries,
            reasoning_effort=_reasoning_effort(args.reasoning_effort),
        )
        for example in examples:
            if (
                example.question_id in source_answers
                and predictor.user_prompt(example) == source_predictor.user_prompt(example)
            ):
                seed_records[example.question_id] = source_answers[example.question_id]
        identity["equivalent_reuse"] = {
            "answers": str(args.reuse_equivalent_answers.resolve()),
            "answers_sha256": file_sha256(args.reuse_equivalent_answers),
            "weights": str(args.reuse_equivalent_weights.resolve()),
            "weights_sha256": source_weights_hash,
            "identical_prompts": len(seed_records),
        }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing weighted answer cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = _provenance_mismatches(previous, identity)
        if mismatches:
            raise ValueError(
                "Cannot resume weighted answer cache with different provenance: "
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
        examples,
        predictor,
        args.output,
        resume=not args.no_resume,
        seed_records=seed_records,
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


def command_identifiability_report(args: argparse.Namespace) -> None:
    """Compare primary, resolver, no-context, and oracle systems."""

    examples = read_prepared(args.data)
    metadata = read_evaluation_metadata(args.references)
    references = {
        question_id: item["references"] for question_id, item in metadata.items()
    }
    conflict_labels = {
        question_id: bool(item.get("has_conflict", False))
        for question_id, item in metadata.items()
    }
    # A field name lets the report read answers from different artifact shapes.
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


def command_candidate_pool(args: argparse.Namespace) -> None:
    """Build the reference-free candidate set used by the oracle diagnostic."""

    if args.output.exists():
        raise ValueError(f"Candidate pool output already exists: {args.output}")
    progress = write_candidate_pool(
        read_prepared(args.data),
        read_answer_cache(args.primary_predictions),
        read_answer_cache(args.secondary_predictions),
        read_extraction_cache(args.extractions),
        args.output,
    )
    _print(progress)


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same network options to all model commands."""

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
    """Build the small public CLI for the final experiment."""

    # Only final project commands are kept.
    parser = argparse.ArgumentParser(prog="conflict-maxent")
    commands = parser.add_subparsers(dest="command", required=True)

    # Data preparation commands do not call a model.
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

    # Model calls are cached because they are slow and cost money.
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

    # These three answer conditions share one prompt and differ only in weights.
    maxent = commands.add_parser(
        "maxent-build", help="Build uniform, support-voting, and soft-MaxEnt weights"
    )
    maxent.add_argument("--data", type=Path, required=True)
    maxent.add_argument("--extractions", type=Path, required=True)
    maxent.add_argument("--output", type=Path, required=True)
    maxent.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    maxent.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    maxent.add_argument("--device", default="cpu")
    maxent.add_argument("--batch-size", type=int, default=8)
    maxent.add_argument("--uniform-mix", type=float, default=0.1)
    maxent.add_argument("--cluster-floor", type=float, default=0.05)
    maxent.add_argument("--slack-penalty", type=float, default=5.0)
    maxent.add_argument("--max-iter", type=int, default=200)
    maxent.add_argument("--tolerance", type=float, default=1e-6)
    maxent.add_argument("--offset", type=int, default=0)
    maxent.add_argument("--limit", type=int)
    maxent.add_argument("--no-resume", action="store_true")
    maxent.set_defaults(function=command_maxent_build)

    weighted = commands.add_parser(
        "answer-weighted", help="Answer with one frozen passage-weight condition"
    )
    weighted.add_argument("--data", type=Path, required=True)
    weighted.add_argument("--weights", type=Path, required=True)
    weighted.add_argument("--output", type=Path, required=True)
    weighted.add_argument("--model", required=True)
    weighted.add_argument(
        "--condition",
        choices=("uniform", "support-voting", "soft-maxent"),
        required=True,
    )
    weighted.add_argument("--offset", type=int, default=0)
    weighted.add_argument("--limit", type=int)
    weighted.add_argument("--no-resume", action="store_true")
    weighted.add_argument("--reuse-equivalent-answers", type=Path)
    weighted.add_argument("--reuse-equivalent-weights", type=Path)
    _add_api_arguments(weighted)
    weighted.set_defaults(function=command_answer_weighted)

    pool = commands.add_parser(
        "candidate-pool", help="Build a reference-free candidate oracle input"
    )
    pool.add_argument("--data", type=Path, required=True)
    pool.add_argument("--primary-predictions", type=Path, required=True)
    pool.add_argument("--secondary-predictions", type=Path, required=True)
    pool.add_argument("--extractions", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.set_defaults(function=command_candidate_pool)

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
    """Parse one command and run its bound handler."""

    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
