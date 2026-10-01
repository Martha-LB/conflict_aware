from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .adaptive_gca import (
    ADJUDICATION_SYSTEM_PROMPT,
    GCA_PROTOCOL_VERSION,
    GENERATION_SYSTEM_PROMPT,
    AdaptiveGCAPredictor,
    evaluate_gca_cache,
    read_gca_cache,
    write_gca_cache,
)
from .answer_baselines import (
    OpenAIAnswerBaseline,
    evaluate_answer_cache,
    read_answer_cache,
    write_answer_cache,
)
from .bayes import BayesianConditionalLogit
from .error_analysis import analyze_unmapped, write_unmapped_analysis
from .cross_model import (
    CANDIDATE_LOCKED_VERIFIER_SYSTEM_PROMPT,
    CROSS_MODEL_PROTOCOL_VERSION,
    SECONDARY_GENERATION_SYSTEM_PROMPT,
    CandidateLockedVerifier,
    CrossModelSecondaryGenerator,
    evaluate_cross_model_predictions,
    read_jsonl_cache,
    write_cross_model_predictions,
    write_secondary_cache,
)
from .disagreement import run_disagreement_cross_validation
from .evaluation import (
    evaluate_choice_sets,
    score_candidate_predictions,
    write_candidate_predictions,
)
from .experiments import run_dev_experiments, write_dev_experiments
from .extraction import (
    EXTRACTION_PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    OpenAICompatibleExtractor,
    annotated_observations,
    read_extraction_cache,
    write_extraction_cache,
)
from .features import FEATURE_NAMES
from .map_gca import (
    MAP_GCA_PROTOCOL_VERSION,
    MAP_GUIDED_SYSTEM_PROMPT,
    BayesianRelationMAP,
    MAPGuidedResolver,
    evaluate_map_resolutions,
    write_map_resolutions,
)
from .pipeline import build_choice_sets
from .protocol import prepare_protocol, read_evaluation_metadata, read_prepared
from .qacc import audit_rows, file_sha256, load_raw_qacc
from .reporting import merge_dev_results
from .two_stage import (
    TwoStageConflictModel,
    evaluate_two_stage_predictions,
    fit_two_stage_model,
    load_domain_metadata,
    read_two_stage_predictions,
    write_two_stage_predictions,
)
from .uncertainty_routing import (
    evaluate_frozen_uncertainty_routing_predictions,
    run_uncertainty_routing_cross_validation,
    run_uncertainty_routing_train_dev,
    write_frozen_uncertainty_routing_predictions,
)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _oracle_cache(examples):
    return {example.question_id: annotated_observations(example) for example in examples}


def command_audit(args: argparse.Namespace) -> None:
    report = audit_rows(load_raw_qacc(args.raw))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print(report)


def command_prepare(args: argparse.Namespace) -> None:
    _print(prepare_protocol(args.raw, args.output_dir, revision=args.revision))


def command_extract(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    extractor = OpenAICompatibleExtractor(
        args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        structured_output=not args.json_mode,
        reasoning_effort=(None if args.reasoning_effort == "omit" else args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "model": args.model,
        "endpoint": args.endpoint,
        "structured_output": not args.json_mode,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError(
                "Existing extraction cache has no manifest; use a new output path "
                "or --no-resume after verifying its provenance"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = []
        for key, value in identity.items():
            previous_value = previous.get(key, value)
            compatible_protocol_upgrade = (
                key == "extraction_protocol_version"
                and previous_value in {"3.1"}
                and value == "3.2"
            )
            if previous_value != value and not compatible_protocol_upgrade:
                mismatches.append(key)
        if mismatches:
            raise ValueError(
                "Cannot resume a cache with different provenance fields: "
                + ", ".join(mismatches)
            )
    preliminary_manifest = {
        **identity,
        "offset": args.offset,
        "limit": args.limit,
        "status": "running",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(preliminary_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress = write_extraction_cache(
        examples,
        extractor,
        args.output,
        resume=not args.no_resume,
    )
    manifest = {
        **identity,
        "offset": args.offset,
        "limit": args.limit,
        "progress": progress,
        "status": "complete",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print({**progress, "output": str(args.output), "manifest": str(manifest_path)})


def _load_oracle_or_cache(examples, path: Path | None):
    return read_extraction_cache(path) if path else _oracle_cache(examples)


def command_train(args: argparse.Namespace) -> None:
    train_examples = read_prepared(args.train)
    dev_examples = read_prepared(args.dev)
    train_extractions = _load_oracle_or_cache(train_examples, args.train_extractions)
    dev_extractions = _load_oracle_or_cache(dev_examples, args.dev_extractions)
    train_sets, train_report = build_choice_sets(
        train_examples,
        train_extractions,
        require_training_label=True,
        answer_match_threshold=args.answer_match_threshold,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        dependency_correction=not args.no_dependency_correction,
    )
    dev_sets, dev_report = build_choice_sets(
        dev_examples,
        dev_extractions,
        require_training_label=True,
        answer_match_threshold=args.answer_match_threshold,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        dependency_correction=not args.no_dependency_correction,
    )
    model = BayesianConditionalLogit(
        FEATURE_NAMES,
        prior_scale=args.prior_scale,
        posterior_samples=args.posterior_samples,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )
    fit = model.fit(train_sets)
    model.calibrate_temperature(
        dev_sets,
        probability_method=args.probability_method,
    )
    model.select_abstention_threshold(
        dev_sets,
        minimum_coverage=args.minimum_coverage,
        probability_method=args.probability_method,
    )
    model.prediction_method = args.probability_method
    model.dependency_correction = not args.no_dependency_correction
    model.save(args.output)
    _print(
        {
            "train_build": train_report.__dict__,
            "dev_build": dev_report.__dict__,
            "fit": fit.__dict__,
            "temperature": model.temperature,
            "abstention_threshold": model.abstention_threshold,
            "prediction_method": model.prediction_method,
            "dependency_correction": not args.no_dependency_correction,
            "output": str(args.output),
            "warning": (
                "No extraction cache supplied: train/dev oracle attributions were used. "
                "This is a diagnostic upper bound, not a deployable pipeline."
                if not args.train_extractions or not args.dev_extractions
                else None
            ),
        }
    )


def command_analyze_unmapped(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    if any(example.split not in {"train", "dev"} for example in examples):
        raise ValueError("analyze-unmapped is restricted to train/dev protocol files")
    extractions = read_extraction_cache(args.extractions)
    rows, report = analyze_unmapped(
        examples,
        extractions,
        answer_match_threshold=args.answer_match_threshold,
    )
    _print(write_unmapped_analysis(rows, report, args.output))


def command_dev_experiments(args: argparse.Namespace) -> None:
    train_examples = read_prepared(args.train)
    dev_examples = read_prepared(args.dev)
    if any(example.split != "train" for example in train_examples):
        raise ValueError("--train must contain only train protocol rows")
    if any(example.split != "dev" for example in dev_examples):
        raise ValueError("--dev must contain only dev protocol rows")
    result = run_dev_experiments(
        train_examples,
        dev_examples,
        read_extraction_cache(args.train_extractions),
        read_extraction_cache(args.dev_extractions),
        answer_match_threshold=args.answer_match_threshold,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        prior_scale=args.prior_scale,
        posterior_samples=args.posterior_samples,
        label_smoothing=args.label_smoothing,
        minimum_coverage=args.minimum_coverage,
        seed=args.seed,
    )
    write_dev_experiments(result, args.output)
    _print({"output": str(args.output), **result})


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
        reasoning_effort=(None if args.reasoning_effort == "omit" else args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "model": args.model,
        "style": args.style,
        "endpoint": args.endpoint,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt_sha256": hashlib.sha256(
            predictor.system_prompt.encode("utf-8")
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
                "Cannot resume an answer cache with different provenance fields: "
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
    )
    manifest_path.write_text(
        json.dumps(
            {**identity, "offset": args.offset, "limit": args.limit, "progress": progress, "status": "complete"},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _print({**progress, "output": str(args.output), "manifest": str(manifest_path)})


def command_evaluate_answer_baseline(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    references = None
    metadata = None
    if any(example.split == "test" for example in examples):
        if args.references is None:
            raise ValueError("Test answer-baseline evaluation requires --references")
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


def command_gca_run(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    predictor = AdaptiveGCAPredictor(
        args.model,
        mode=args.mode,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        reasoning_effort=(None if args.reasoning_effort == "omit" else args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "model": args.model,
        "mode": args.mode,
        "endpoint": args.endpoint,
        "reasoning_effort": args.reasoning_effort,
        "protocol_version": GCA_PROTOCOL_VERSION,
        "generation_prompt_sha256": hashlib.sha256(
            GENERATION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "adjudication_prompt_sha256": hashlib.sha256(
            ADJUDICATION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing Adaptive GCA cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in identity.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(
                "Cannot resume an Adaptive GCA cache with different provenance fields: "
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
    progress = write_gca_cache(
        examples,
        predictor,
        args.output,
        resume=not args.no_resume,
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


def command_gca_evaluate(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    references = None
    if any(example.split == "test" for example in examples):
        if args.references is None:
            raise ValueError("Test Adaptive GCA evaluation requires --references")
        metadata = read_evaluation_metadata(args.references)
        references = {
            question_id: item["references"] for question_id, item in metadata.items()
        }
    summary = evaluate_gca_cache(
        examples,
        read_gca_cache(args.predictions),
        references=references,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def command_gca_map_fit(args: argparse.Namespace) -> None:
    examples = read_prepared(args.train)
    if any(example.split != "train" for example in examples):
        raise ValueError("gca-map-fit requires a train protocol file")
    model = BayesianRelationMAP(
        alpha=args.alpha,
        dependency_correction=not args.no_dependency_correction,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        accept_probability=args.accept_probability,
        accept_margin=args.accept_margin,
    )
    report = model.fit(examples, read_gca_cache(args.train_generations))
    model.save(args.output)
    _print(
        {
            "fit": report,
            "accept_probability": model.accept_probability,
            "accept_margin": model.accept_margin,
            "dependency_correction": model.dependency_correction,
            "output": str(args.output),
        }
    )


def command_gca_map_resolve(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    relation_model = BayesianRelationMAP.load(args.map_model)
    resolver = MAPGuidedResolver(
        relation_model,
        args.model,
        mode=args.mode,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        reasoning_effort=(None if args.reasoning_effort == "omit" else args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "generations": str(args.generations.resolve()),
        "generations_sha256": file_sha256(args.generations),
        "map_model": str(args.map_model.resolve()),
        "map_model_sha256": file_sha256(args.map_model),
        "llm_model": args.model,
        "mode": args.mode,
        "endpoint": args.endpoint,
        "reasoning_effort": args.reasoning_effort,
        "protocol_version": MAP_GCA_PROTOCOL_VERSION,
        "adjudication_prompt_sha256": hashlib.sha256(
            MAP_GUIDED_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing MAP-GCA cache has no provenance manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in identity.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(
                "Cannot resume a MAP-GCA cache with different provenance fields: "
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
    progress = write_map_resolutions(
        examples,
        read_gca_cache(args.generations),
        resolver,
        args.output,
        resume=not args.no_resume,
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


def command_gca_map_evaluate(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    references = None
    if any(example.split == "test" for example in examples):
        if args.references is None:
            raise ValueError("Test MAP-GCA evaluation requires --references")
        metadata = read_evaluation_metadata(args.references)
        references = {
            question_id: item["references"] for question_id, item in metadata.items()
        }
    summary = evaluate_map_resolutions(
        examples,
        read_gca_cache(args.predictions),
        references=references,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def command_two_stage_fit(args: argparse.Namespace) -> None:
    examples = read_prepared(args.train)
    if any(example.split != "train" for example in examples):
        raise ValueError("two-stage-fit requires a train protocol file")
    generations = read_gca_cache(args.train_generations)
    extractions = (
        read_extraction_cache(args.train_extractions)
        if args.train_extractions is not None
        else {}
    )
    domain_metadata = load_domain_metadata(args.domain_metadata)
    model, report = fit_two_stage_model(
        examples,
        generations,
        extractions,
        domain_metadata,
        detector_prior_scale=args.detector_prior_scale,
        ranker_prior_scale=args.ranker_prior_scale,
        posterior_samples=args.posterior_samples,
        label_smoothing=args.label_smoothing,
        folds=args.folds,
        seed=args.seed,
        accept_probability=args.accept_probability,
        accept_margin=args.accept_margin,
    )
    model.save(args.output)
    _print({"output": str(args.output), **report})


def command_two_stage_resolve(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    model = TwoStageConflictModel.load(args.two_stage_model)
    extractions = (
        read_extraction_cache(args.extractions) if args.extractions is not None else {}
    )
    report = write_two_stage_predictions(
        examples,
        read_gca_cache(args.generations),
        extractions,
        model,
        args.output,
        mode=args.mode,
    )
    manifest = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "generations": str(args.generations.resolve()),
        "generations_sha256": file_sha256(args.generations),
        "two_stage_model": str(args.two_stage_model.resolve()),
        "two_stage_model_sha256": file_sha256(args.two_stage_model),
        "extractions": str(args.extractions.resolve()) if args.extractions else None,
        "extractions_sha256": file_sha256(args.extractions) if args.extractions else None,
        "mode": args.mode,
        "offset": args.offset,
        "limit": args.limit,
        "status": "complete",
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print({**report, "manifest": str(manifest_path)})


def command_two_stage_evaluate(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    references = None
    conflict_labels = None
    if any(example.split == "test" for example in examples):
        if args.references is None:
            raise ValueError("Test two-stage evaluation requires --references")
        metadata = read_evaluation_metadata(args.references)
        references = {
            question_id: item["references"] for question_id, item in metadata.items()
        }
        conflict_labels = {
            question_id: bool(item.get("has_conflict", False))
            for question_id, item in metadata.items()
        }
    summary = evaluate_two_stage_predictions(
        examples,
        read_two_stage_predictions(args.predictions),
        references=references,
        conflict_labels=conflict_labels,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def command_cross_model_generate(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    risk_model = TwoStageConflictModel.load(args.risk_model)
    generator = CrossModelSecondaryGenerator(
        args.model,
        risk_model,
        risk_mode=args.risk_mode,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        reasoning_effort=(None if args.reasoning_effort == "omit" else args.reasoning_effort),
    )
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "primary_generations": str(args.primary_generations.resolve()),
        "primary_generations_sha256": file_sha256(args.primary_generations),
        "risk_model": str(args.risk_model.resolve()),
        "risk_model_sha256": file_sha256(args.risk_model),
        "extractions": str(args.extractions.resolve()) if args.extractions else None,
        "extractions_sha256": file_sha256(args.extractions) if args.extractions else None,
        "model": args.model,
        "endpoint": args.endpoint,
        "reasoning_effort": args.reasoning_effort,
        "risk_mode": args.risk_mode,
        "protocol_version": CROSS_MODEL_PROTOCOL_VERSION,
        "prompt_sha256": hashlib.sha256(
            SECONDARY_GENERATION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing secondary-generation cache has no manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in identity.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(
                "Cannot resume secondary cache with different provenance: "
                + ", ".join(mismatches)
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {**identity, "offset": args.offset, "limit": args.limit, "status": "running"},
            indent=2, ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    progress = write_secondary_cache(
        examples,
        read_gca_cache(args.primary_generations),
        read_extraction_cache(args.extractions) if args.extractions else {},
        generator,
        args.output,
        resume=not args.no_resume,
    )
    manifest_path.write_text(
        json.dumps(
            {**identity, "offset": args.offset, "limit": args.limit,
             "progress": progress, "status": "complete"},
            indent=2, ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    _print({**progress, "output": str(args.output), "manifest": str(manifest_path)})


def command_cross_model_verify(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    verifier = CandidateLockedVerifier(
        args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        reasoning_effort=(None if args.reasoning_effort == "omit" else args.reasoning_effort),
    )
    domain_metadata = load_domain_metadata(args.domain_metadata)
    identity = {
        "input": str(args.data.resolve()),
        "input_sha256": file_sha256(args.data),
        "primary_generations": str(args.primary_generations.resolve()),
        "primary_generations_sha256": file_sha256(args.primary_generations),
        "secondary_generations": str(args.secondary_generations.resolve()),
        "secondary_generations_sha256": file_sha256(args.secondary_generations),
        "extractions": str(args.extractions.resolve()) if args.extractions else None,
        "extractions_sha256": file_sha256(args.extractions) if args.extractions else None,
        "domain_metadata": str(args.domain_metadata.resolve()) if args.domain_metadata else None,
        "domain_metadata_sha256": file_sha256(args.domain_metadata) if args.domain_metadata else None,
        "model": args.model,
        "endpoint": args.endpoint,
        "reasoning_effort": args.reasoning_effort,
        "mode": args.mode,
        "protocol_version": CROSS_MODEL_PROTOCOL_VERSION,
        "prompt_sha256": hashlib.sha256(
            CANDIDATE_LOCKED_VERIFIER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if not args.no_resume and (args.output.exists() or temporary_path.exists()):
        if not manifest_path.exists():
            raise ValueError("Existing cross-model verification cache has no manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in identity.items() if previous.get(key) != value]
        # The secondary cache is intentionally appendable as LIMIT grows, so
        # its whole-file digest changes even when all existing records are
        # unchanged. write_cross_model_predictions verifies a per-record digest
        # before reusing each cached verification.
        append_only_secondary_growth = (
            mismatches == ["secondary_generations_sha256"]
            and previous.get("secondary_generations") == identity["secondary_generations"]
        )
        if mismatches and not append_only_secondary_growth:
            raise ValueError(
                "Cannot resume verification cache with different provenance: "
                + ", ".join(mismatches)
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {**identity, "offset": args.offset, "limit": args.limit, "status": "running"},
            indent=2, ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    progress = write_cross_model_predictions(
        examples,
        read_gca_cache(args.primary_generations),
        read_jsonl_cache(args.secondary_generations),
        read_extraction_cache(args.extractions) if args.extractions else {},
        verifier,
        args.output,
        domain_metadata=domain_metadata,
        mode=args.mode,
        resume=not args.no_resume,
    )
    manifest_path.write_text(
        json.dumps(
            {**identity, "offset": args.offset, "limit": args.limit,
             "progress": progress, "status": "complete"},
            indent=2, ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    _print({**progress, "output": str(args.output), "manifest": str(manifest_path)})


def command_cross_model_evaluate(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    references = None
    conflict_labels = None
    if any(example.split == "test" for example in examples):
        if args.references is None:
            raise ValueError("Test cross-model evaluation requires --references")
        metadata = read_evaluation_metadata(args.references)
        references = {key: item["references"] for key, item in metadata.items()}
        conflict_labels = {key: bool(item.get("has_conflict", False)) for key, item in metadata.items()}
    summary = evaluate_cross_model_predictions(
        examples,
        read_jsonl_cache(args.predictions),
        references=references,
        conflict_labels=conflict_labels,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def command_disagreement_cv(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    if any(example.split == "test" for example in examples):
        raise ValueError("disagreement-cv is a labelled train/dev pilot, not a test command")
    summary = run_disagreement_cross_validation(
        examples,
        read_answer_cache(args.luna_predictions),
        read_answer_cache(args.mini_predictions),
        read_extraction_cache(args.extractions) if args.extractions else {},
        folds=args.folds,
        calibration_fraction=args.calibration_fraction,
        alpha=args.alpha,
        prior_scale=args.prior_scale,
        posterior_samples=args.posterior_samples,
        seed=args.seed,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def command_uq_routing_cv(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    if any(example.split == "test" for example in examples):
        raise ValueError("uq-routing-cv is a labelled train/dev pilot, not a test command")
    summary = run_uncertainty_routing_cross_validation(
        examples,
        read_answer_cache(args.luna_predictions),
        read_answer_cache(args.mini_predictions),
        read_extraction_cache(args.extractions) if args.extractions else {},
        folds=args.folds,
        calibration_fraction=args.calibration_fraction,
        prior_scale=args.prior_scale,
        intervention_quantile=args.intervention_quantile,
        delta_margin=args.delta_margin,
        selective_coverage=args.selective_coverage,
        pre_feature_mode=args.pre_feature_mode,
        seed=args.seed,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def command_uq_routing_train_dev(args: argparse.Namespace) -> None:
    train_examples = read_prepared(args.train_data)
    dev_examples = read_prepared(args.dev_data)
    summary = run_uncertainty_routing_train_dev(
        train_examples,
        dev_examples,
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


def command_compare_uq_routing(args: argparse.Namespace) -> None:
    def compact(path: Path) -> dict:
        summary = json.loads(path.read_text(encoding="utf-8"))
        route = summary["overall"]["conservative_uq_route"]
        return {
            "primary": summary["configuration"]["primary_name"],
            "secondary": summary["configuration"]["secondary_name"],
            "primary_em": summary["overall"]["primary_direct_rag"]["answer_exact"],
            "routed_em": route["answer_exact"],
            "routed_f1": route["answer_token_f1"],
            "em_delta_vs_primary": route["em_delta_vs_primary"],
            "switches": route["switches"],
            "helped": route["helped"],
            "hurt": route["hurt"],
            "mcnemar_p": route["mcnemar_exact_p_two_sided"],
            "error_detection_auroc": summary["risk_quality"]["auroc_error_detection"],
            "conflict_em": summary["conflict_only"]["conservative_uq_route"]["answer_exact"],
            "nonconflict_em": summary["nonconflict_only"]["conservative_uq_route"]["answer_exact"],
            "mean_llm_calls": summary["usage"]["mean_llm_calls_per_question"],
        }

    _print({
        "protocol": "dev_calibration_and_direction_selection_not_test",
        "luna_first": compact(args.luna_first),
        "mini_first": compact(args.mini_first),
    })


def command_uq_routing_predict(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(
            f"Frozen prediction output already exists: {args.output}; "
            "use a new versioned path rather than overwriting it"
        )
    examples = read_prepared(args.data)
    model_artifact = json.loads(args.model.read_text(encoding="utf-8"))
    progress = write_frozen_uncertainty_routing_predictions(
        examples,
        read_answer_cache(args.primary_predictions),
        read_answer_cache(args.secondary_predictions),
        read_extraction_cache(args.extractions),
        model_artifact,
        args.output,
    )
    manifest = {
        "protocol": "frozen_official_test_prediction_without_references",
        "protocol_version": model_artifact["protocol_version"],
        "configuration": model_artifact["configuration"],
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in {
                "data": args.data,
                "primary_predictions": args.primary_predictions,
                "secondary_predictions": args.secondary_predictions,
                "extractions": args.extractions,
                "model": args.model,
            }.items()
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


def command_uq_routing_evaluate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(
            f"Scored output already exists: {args.output}; use a new versioned path"
        )
    examples = read_prepared(args.data)
    summary = evaluate_frozen_uncertainty_routing_predictions(
        examples,
        read_jsonl_cache(args.predictions),
        read_evaluation_metadata(args.references),
        seed=args.seed,
        output_path=args.output,
    )
    manifest = {
        "protocol": summary["protocol"],
        "protocol_version": summary["protocol_version"],
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in {
                "data": args.data,
                "predictions": args.predictions,
                "references": args.references,
            }.items()
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


def command_merge_dev_results(args: argparse.Namespace) -> None:
    _print(
        merge_dev_results(
            args.candidate_experiments,
            args.direct_rag_summary,
            args.qacc_summary,
            args.output,
        )
    )


def command_evaluate(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    if any(example.split != "test" for example in examples):
        raise ValueError("evaluate expects a test protocol file")
    extractions = read_extraction_cache(args.extractions)
    evaluation_metadata = read_evaluation_metadata(args.references)
    references = {
        question_id: item["references"]
        for question_id, item in evaluation_metadata.items()
    }
    model = BayesianConditionalLogit.load(args.model)
    choice_sets, report = build_choice_sets(
        examples,
        extractions,
        require_training_label=False,
        answer_match_threshold=args.answer_match_threshold,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        references=references,
        evaluation_metadata=evaluation_metadata,
        dependency_correction=model.dependency_correction,
    )
    summary = evaluate_choice_sets(
        model,
        choice_sets,
        output_path=args.output,
        expected_question_ids=[example.question_id for example in examples],
        evaluation_metadata=evaluation_metadata,
    )
    _print({"build": report.__dict__, "metrics": summary})


def command_predict(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    if any(example.split != "test" for example in examples):
        raise ValueError("predict expects a test protocol input file")
    model = BayesianConditionalLogit.load(args.model)
    choice_sets, report = build_choice_sets(
        examples,
        read_extraction_cache(args.extractions),
        require_training_label=False,
        answer_match_threshold=args.answer_match_threshold,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        dependency_correction=model.dependency_correction,
    )
    frozen = write_candidate_predictions(
        choice_sets,
        expected_question_ids=[example.question_id for example in examples],
        probabilities=model.probabilities,
        abstention_threshold=model.abstention_threshold,
        method=args.method_name,
        output_path=args.output,
    )
    _print({"build": report.__dict__, "prediction": frozen})


def command_predict_baseline(args: argparse.Namespace) -> None:
    examples = read_prepared(args.data)
    if any(example.split != "test" for example in examples):
        raise ValueError("predict-baseline expects a test protocol input file")
    dependency_correction = args.method == "independent-source-majority"
    choice_sets, report = build_choice_sets(
        examples,
        read_extraction_cache(args.extractions),
        require_training_label=False,
        answer_match_threshold=args.answer_match_threshold,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        dependency_correction=dependency_correction,
    )

    support_index = FEATURE_NAMES.index("support_share")

    def support_probabilities(choice_set):
        values = choice_set.features[:, support_index]
        total = float(values.sum())
        return values / total if total > 0 else np.full(len(values), 1.0 / len(values))

    frozen = write_candidate_predictions(
        choice_sets,
        expected_question_ids=[example.question_id for example in examples],
        probabilities=support_probabilities,
        abstention_threshold=0.0,
        method=args.method,
        output_path=args.output,
    )
    _print({"build": report.__dict__, "prediction": frozen})


def command_score(args: argparse.Namespace) -> None:
    metadata = read_evaluation_metadata(args.references)
    summary = score_candidate_predictions(
        args.predictions,
        metadata,
        output_path=args.output,
    )
    _print({"output": str(args.output), "metrics": summary})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uacaa")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Audit the untouched official QACC file")
    audit.add_argument("--raw", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(function=command_audit)

    prepare = commands.add_parser("prepare", help="Create versioned raw/strict/weak protocol files")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--revision")
    prepare.set_defaults(function=command_prepare)

    extract = commands.add_parser("extract", help="Extract candidates once per question")
    extract.add_argument("--data", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--endpoint", default="https://api.openai.com/v1/chat/completions")
    extract.add_argument("--api-key-env", default="OPENAI_API_KEY")
    extract.add_argument("--timeout", type=float, default=120.0)
    extract.add_argument("--retries", type=int, default=3)
    extract.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "omit"),
        default="none",
        help="Use 'omit' for compatible endpoints that reject this OpenAI parameter",
    )
    extract.add_argument("--offset", type=int, default=0)
    extract.add_argument("--limit", type=int)
    extract.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard an existing cache instead of resuming it",
    )
    extract.add_argument(
        "--json-mode",
        action="store_true",
        help="Use legacy JSON mode for endpoints without Structured Outputs",
    )
    extract.set_defaults(function=command_extract)

    train = commands.add_parser("train", help="Fit and calibrate the candidate posterior")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--dev", type=Path, required=True)
    train.add_argument("--train-extractions", type=Path)
    train.add_argument("--dev-extractions", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--answer-match-threshold", type=float, default=0.8)
    train.add_argument("--near-duplicate-jaccard", type=float, default=0.88)
    train.add_argument("--no-dependency-correction", action="store_true")
    train.add_argument("--prior-scale", type=float, default=2.0)
    train.add_argument("--posterior-samples", type=int, default=1000)
    train.add_argument("--label-smoothing", type=float, default=0.05)
    train.add_argument("--minimum-coverage", type=float, default=0.7)
    train.add_argument(
        "--probability-method",
        choices=("posterior", "map"),
        default="posterior",
    )
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(function=command_train)

    unmapped = commands.add_parser(
        "analyze-unmapped",
        help="Audit train/dev gold answers missing from extracted candidate sets",
    )
    unmapped.add_argument("--data", type=Path, required=True)
    unmapped.add_argument("--extractions", type=Path, required=True)
    unmapped.add_argument("--output", type=Path, required=True)
    unmapped.add_argument("--answer-match-threshold", type=float, default=0.8)
    unmapped.set_defaults(function=command_analyze_unmapped)

    experiments = commands.add_parser(
        "dev-experiments",
        help="Run candidate baselines and ablations on train/dev only",
    )
    experiments.add_argument("--train", type=Path, required=True)
    experiments.add_argument("--dev", type=Path, required=True)
    experiments.add_argument("--train-extractions", type=Path, required=True)
    experiments.add_argument("--dev-extractions", type=Path, required=True)
    experiments.add_argument("--output", type=Path, required=True)
    experiments.add_argument("--answer-match-threshold", type=float, default=0.8)
    experiments.add_argument("--near-duplicate-jaccard", type=float, default=0.88)
    experiments.add_argument("--prior-scale", type=float, default=2.0)
    experiments.add_argument("--posterior-samples", type=int, default=1000)
    experiments.add_argument("--label-smoothing", type=float, default=0.05)
    experiments.add_argument("--minimum-coverage", type=float, default=0.7)
    experiments.add_argument("--seed", type=int, default=42)
    experiments.set_defaults(function=command_dev_experiments)

    answer_baseline = commands.add_parser(
        "answer-baseline",
        help="Run a one-call-per-question Direct-RAG or QACC prompting baseline",
    )
    answer_baseline.add_argument("--data", type=Path, required=True)
    answer_baseline.add_argument("--output", type=Path, required=True)
    answer_baseline.add_argument("--model", required=True)
    answer_baseline.add_argument(
        "--style",
        choices=("direct-rag", "qacc-explain-then-answer"),
        required=True,
    )
    answer_baseline.add_argument(
        "--endpoint", default="https://api.openai.com/v1/chat/completions"
    )
    answer_baseline.add_argument("--api-key-env", default="OPENAI_API_KEY")
    answer_baseline.add_argument("--timeout", type=float, default=120.0)
    answer_baseline.add_argument("--retries", type=int, default=3)
    answer_baseline.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "omit"),
        default="none",
    )
    answer_baseline.add_argument("--offset", type=int, default=0)
    answer_baseline.add_argument("--limit", type=int)
    answer_baseline.add_argument("--no-resume", action="store_true")
    answer_baseline.set_defaults(function=command_answer_baseline)

    evaluate_answer = commands.add_parser(
        "evaluate-answer-baseline",
        help="Evaluate a cached answer baseline on train/dev or frozen test references",
    )
    evaluate_answer.add_argument("--data", type=Path, required=True)
    evaluate_answer.add_argument("--predictions", type=Path, required=True)
    evaluate_answer.add_argument("--references", type=Path)
    evaluate_answer.add_argument("--output", type=Path, required=True)
    evaluate_answer.set_defaults(function=command_evaluate_answer_baseline)

    gca_run = commands.add_parser(
        "gca-run",
        help="Run joint generation with adaptive blind challenge/adjudication",
    )
    gca_run.add_argument("--data", type=Path, required=True)
    gca_run.add_argument("--output", type=Path, required=True)
    gca_run.add_argument("--model", required=True)
    gca_run.add_argument(
        "--mode",
        choices=("adaptive", "always", "never"),
        default="adaptive",
    )
    gca_run.add_argument(
        "--endpoint", default="https://api.openai.com/v1/chat/completions"
    )
    gca_run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    gca_run.add_argument("--timeout", type=float, default=120.0)
    gca_run.add_argument("--retries", type=int, default=3)
    gca_run.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "omit"),
        default="none",
    )
    gca_run.add_argument("--offset", type=int, default=0)
    gca_run.add_argument("--limit", type=int)
    gca_run.add_argument("--no-resume", action="store_true")
    gca_run.set_defaults(function=command_gca_run)

    gca_evaluate = commands.add_parser(
        "gca-evaluate",
        help="Compare joint-generation and adaptive-final answers",
    )
    gca_evaluate.add_argument("--data", type=Path, required=True)
    gca_evaluate.add_argument("--predictions", type=Path, required=True)
    gca_evaluate.add_argument("--references", type=Path)
    gca_evaluate.add_argument("--output", type=Path, required=True)
    gca_evaluate.add_argument("--offset", type=int, default=0)
    gca_evaluate.add_argument("--limit", type=int)
    gca_evaluate.set_defaults(function=command_gca_evaluate)

    gca_map_fit = commands.add_parser(
        "gca-map-fit",
        help="Fit the Dirichlet relation model used for conflict MAP inference",
    )
    gca_map_fit.add_argument("--train", type=Path, required=True)
    gca_map_fit.add_argument("--train-generations", type=Path, required=True)
    gca_map_fit.add_argument("--output", type=Path, required=True)
    gca_map_fit.add_argument("--alpha", type=float, default=1.0)
    gca_map_fit.add_argument("--near-duplicate-jaccard", type=float, default=0.88)
    gca_map_fit.add_argument("--no-dependency-correction", action="store_true")
    gca_map_fit.add_argument("--accept-probability", type=float, default=0.75)
    gca_map_fit.add_argument("--accept-margin", type=float, default=0.30)
    gca_map_fit.set_defaults(function=command_gca_map_fit)

    gca_map_resolve = commands.add_parser(
        "gca-map-resolve",
        help="Route clear, insufficient, and conflicting questions through MAP-GCA",
    )
    gca_map_resolve.add_argument("--data", type=Path, required=True)
    gca_map_resolve.add_argument("--generations", type=Path, required=True)
    gca_map_resolve.add_argument("--map-model", type=Path, required=True)
    gca_map_resolve.add_argument("--model", required=True)
    gca_map_resolve.add_argument("--output", type=Path, required=True)
    gca_map_resolve.add_argument(
        "--mode",
        choices=("map-guided", "map-only", "always-adjudicate"),
        default="map-guided",
    )
    gca_map_resolve.add_argument(
        "--endpoint", default="https://api.openai.com/v1/chat/completions"
    )
    gca_map_resolve.add_argument("--api-key-env", default="OPENAI_API_KEY")
    gca_map_resolve.add_argument("--timeout", type=float, default=120.0)
    gca_map_resolve.add_argument("--retries", type=int, default=3)
    gca_map_resolve.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "omit"),
        default="none",
    )
    gca_map_resolve.add_argument("--offset", type=int, default=0)
    gca_map_resolve.add_argument("--limit", type=int)
    gca_map_resolve.add_argument("--no-resume", action="store_true")
    gca_map_resolve.set_defaults(function=command_gca_map_resolve)

    gca_map_evaluate = commands.add_parser(
        "gca-map-evaluate",
        help="Evaluate initial generation versus the MAP-routed final answer",
    )
    gca_map_evaluate.add_argument("--data", type=Path, required=True)
    gca_map_evaluate.add_argument("--predictions", type=Path, required=True)
    gca_map_evaluate.add_argument("--references", type=Path)
    gca_map_evaluate.add_argument("--output", type=Path, required=True)
    gca_map_evaluate.add_argument("--offset", type=int, default=0)
    gca_map_evaluate.add_argument("--limit", type=int)
    gca_map_evaluate.set_defaults(function=command_gca_map_evaluate)

    two_stage_fit = commands.add_parser(
        "two-stage-fit",
        help="Fit a QACC-supervised conflict detector and evidence-aware candidate ranker",
    )
    two_stage_fit.add_argument("--train", type=Path, required=True)
    two_stage_fit.add_argument("--train-generations", type=Path, required=True)
    two_stage_fit.add_argument("--train-extractions", type=Path)
    two_stage_fit.add_argument(
        "--domain-metadata",
        type=Path,
        help="Optional frozen JSON with category, authority, and independence_group per domain",
    )
    two_stage_fit.add_argument("--output", type=Path, required=True)
    two_stage_fit.add_argument("--detector-prior-scale", type=float, default=1.0)
    two_stage_fit.add_argument("--ranker-prior-scale", type=float, default=1.0)
    two_stage_fit.add_argument("--posterior-samples", type=int, default=1000)
    two_stage_fit.add_argument("--label-smoothing", type=float, default=0.05)
    two_stage_fit.add_argument("--folds", type=int, default=5)
    two_stage_fit.add_argument("--seed", type=int, default=42)
    two_stage_fit.add_argument(
        "--accept-probability", type=float,
        help="Override the train-only OOF conservative switching threshold",
    )
    two_stage_fit.add_argument(
        "--accept-margin", type=float,
        help="Override the train-only OOF conservative margin threshold",
    )
    two_stage_fit.set_defaults(function=command_two_stage_fit)

    two_stage_resolve = commands.add_parser(
        "two-stage-resolve",
        help="Keep clear initial answers and rank candidates only when conflict is predicted",
    )
    two_stage_resolve.add_argument("--data", type=Path, required=True)
    two_stage_resolve.add_argument("--generations", type=Path, required=True)
    two_stage_resolve.add_argument("--extractions", type=Path)
    two_stage_resolve.add_argument("--two-stage-model", type=Path, required=True)
    two_stage_resolve.add_argument("--output", type=Path, required=True)
    two_stage_resolve.add_argument(
        "--mode", choices=("map", "conservative"), default="conservative"
    )
    two_stage_resolve.add_argument("--offset", type=int, default=0)
    two_stage_resolve.add_argument("--limit", type=int)
    two_stage_resolve.set_defaults(function=command_two_stage_resolve)

    two_stage_evaluate = commands.add_parser(
        "two-stage-evaluate",
        help="Evaluate conflict classification and final QA separately",
    )
    two_stage_evaluate.add_argument("--data", type=Path, required=True)
    two_stage_evaluate.add_argument("--predictions", type=Path, required=True)
    two_stage_evaluate.add_argument("--references", type=Path)
    two_stage_evaluate.add_argument("--output", type=Path, required=True)
    two_stage_evaluate.add_argument("--offset", type=int, default=0)
    two_stage_evaluate.add_argument("--limit", type=int)
    two_stage_evaluate.set_defaults(function=command_two_stage_evaluate)

    cross_generate = commands.add_parser(
        "cross-model-generate",
        help="Call an independent secondary model only on train-fitted risk cases",
    )
    cross_generate.add_argument("--data", type=Path, required=True)
    cross_generate.add_argument("--primary-generations", type=Path, required=True)
    cross_generate.add_argument("--extractions", type=Path)
    cross_generate.add_argument("--risk-model", type=Path, required=True)
    cross_generate.add_argument("--model", required=True)
    cross_generate.add_argument("--output", type=Path, required=True)
    cross_generate.add_argument(
        "--risk-mode", choices=("detector", "rules", "union"), default="union"
    )
    cross_generate.add_argument(
        "--endpoint", default="https://api.openai.com/v1/chat/completions"
    )
    cross_generate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    cross_generate.add_argument("--timeout", type=float, default=120.0)
    cross_generate.add_argument("--retries", type=int, default=3)
    cross_generate.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "omit"),
        default="none",
    )
    cross_generate.add_argument("--offset", type=int, default=0)
    cross_generate.add_argument("--limit", type=int)
    cross_generate.add_argument("--no-resume", action="store_true")
    cross_generate.set_defaults(function=command_cross_model_generate)

    cross_verify = commands.add_parser(
        "cross-model-verify",
        help="Use a candidate-locked verifier over source/time evidence cards",
    )
    cross_verify.add_argument("--data", type=Path, required=True)
    cross_verify.add_argument("--primary-generations", type=Path, required=True)
    cross_verify.add_argument("--secondary-generations", type=Path, required=True)
    cross_verify.add_argument("--extractions", type=Path)
    cross_verify.add_argument("--domain-metadata", type=Path)
    cross_verify.add_argument("--model", required=True)
    cross_verify.add_argument("--output", type=Path, required=True)
    cross_verify.add_argument(
        "--mode", choices=("select", "conservative"), default="conservative"
    )
    cross_verify.add_argument(
        "--endpoint", default="https://api.openai.com/v1/chat/completions"
    )
    cross_verify.add_argument("--api-key-env", default="OPENAI_API_KEY")
    cross_verify.add_argument("--timeout", type=float, default=120.0)
    cross_verify.add_argument("--retries", type=int, default=3)
    cross_verify.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "omit"),
        default="none",
    )
    cross_verify.add_argument("--offset", type=int, default=0)
    cross_verify.add_argument("--limit", type=int)
    cross_verify.add_argument("--no-resume", action="store_true")
    cross_verify.set_defaults(function=command_cross_model_verify)

    cross_evaluate = commands.add_parser(
        "cross-model-evaluate",
        help="Compare initial and candidate-locked cross-model answers",
    )
    cross_evaluate.add_argument("--data", type=Path, required=True)
    cross_evaluate.add_argument("--predictions", type=Path, required=True)
    cross_evaluate.add_argument("--references", type=Path)
    cross_evaluate.add_argument("--output", type=Path, required=True)
    cross_evaluate.add_argument("--offset", type=int, default=0)
    cross_evaluate.add_argument("--limit", type=int)
    cross_evaluate.set_defaults(function=command_cross_model_evaluate)

    disagreement_cv = commands.add_parser(
        "disagreement-cv",
        help=(
            "Run an out-of-fold Bayesian Luna/Mini disagreement ranker with "
            "split-conformal abstention"
        ),
    )
    disagreement_cv.add_argument("--data", type=Path, required=True)
    disagreement_cv.add_argument("--luna-predictions", type=Path, required=True)
    disagreement_cv.add_argument("--mini-predictions", type=Path, required=True)
    disagreement_cv.add_argument("--extractions", type=Path)
    disagreement_cv.add_argument("--output", type=Path, required=True)
    disagreement_cv.add_argument("--folds", type=int, default=5)
    disagreement_cv.add_argument("--calibration-fraction", type=float, default=0.25)
    disagreement_cv.add_argument("--alpha", type=float, default=0.20)
    disagreement_cv.add_argument("--prior-scale", type=float, default=0.75)
    disagreement_cv.add_argument("--posterior-samples", type=int, default=500)
    disagreement_cv.add_argument("--seed", type=int, default=42)
    disagreement_cv.set_defaults(function=command_disagreement_cv)

    uq_routing_cv = commands.add_parser(
        "uq-routing-cv",
        help=(
            "Run an out-of-fold Direct-RAG error-risk and conservative "
            "intervention routing pilot"
        ),
    )
    uq_routing_cv.add_argument("--data", type=Path, required=True)
    uq_routing_cv.add_argument("--luna-predictions", type=Path, required=True)
    uq_routing_cv.add_argument("--mini-predictions", type=Path, required=True)
    uq_routing_cv.add_argument("--extractions", type=Path)
    uq_routing_cv.add_argument("--output", type=Path, required=True)
    uq_routing_cv.add_argument("--folds", type=int, default=5)
    uq_routing_cv.add_argument("--calibration-fraction", type=float, default=0.25)
    uq_routing_cv.add_argument("--prior-scale", type=float, default=0.75)
    uq_routing_cv.add_argument("--intervention-quantile", type=float, default=0.75)
    uq_routing_cv.add_argument("--delta-margin", type=float, default=0.10)
    uq_routing_cv.add_argument("--selective-coverage", type=float, default=0.80)
    uq_routing_cv.add_argument(
        "--pre-feature-mode",
        choices=("full", "no-source-time", "answer-only"),
        default="full",
    )
    uq_routing_cv.add_argument("--seed", type=int, default=42)
    uq_routing_cv.set_defaults(function=command_uq_routing_cv)

    uq_train_dev = commands.add_parser(
        "uq-routing-train-dev",
        help="Fit UQ error models on train and calibrate/compare routing on dev",
    )
    uq_train_dev.add_argument("--train-data", type=Path, required=True)
    uq_train_dev.add_argument("--dev-data", type=Path, required=True)
    uq_train_dev.add_argument("--train-primary-predictions", type=Path, required=True)
    uq_train_dev.add_argument("--train-secondary-predictions", type=Path, required=True)
    uq_train_dev.add_argument("--dev-primary-predictions", type=Path, required=True)
    uq_train_dev.add_argument("--dev-secondary-predictions", type=Path, required=True)
    uq_train_dev.add_argument("--train-extractions", type=Path)
    uq_train_dev.add_argument("--dev-extractions", type=Path)
    uq_train_dev.add_argument("--primary-name", required=True)
    uq_train_dev.add_argument("--secondary-name", required=True)
    uq_train_dev.add_argument("--output", type=Path, required=True)
    uq_train_dev.add_argument("--prior-scale", type=float, default=0.75)
    uq_train_dev.add_argument("--intervention-quantile", type=float, default=0.75)
    uq_train_dev.add_argument("--delta-margin", type=float, default=0.10)
    uq_train_dev.add_argument("--selective-coverage", type=float, default=0.80)
    uq_train_dev.add_argument(
        "--pre-feature-mode",
        choices=("full", "no-source-time", "answer-only"),
        default="answer-only",
    )
    uq_train_dev.add_argument("--seed", type=int, default=42)
    uq_train_dev.set_defaults(function=command_uq_routing_train_dev)

    compare_uq = commands.add_parser(
        "compare-uq-routing",
        help="Compare frozen-format Luna-first and Mini-first dev summaries",
    )
    compare_uq.add_argument("--luna-first", type=Path, required=True)
    compare_uq.add_argument("--mini-first", type=Path, required=True)
    compare_uq.set_defaults(function=command_compare_uq_routing)

    uq_predict = commands.add_parser(
        "uq-routing-predict",
        help="Apply a frozen UQ router to test inputs without loading references",
    )
    uq_predict.add_argument("--data", type=Path, required=True)
    uq_predict.add_argument("--primary-predictions", type=Path, required=True)
    uq_predict.add_argument("--secondary-predictions", type=Path, required=True)
    uq_predict.add_argument("--extractions", type=Path, required=True)
    uq_predict.add_argument("--model", type=Path, required=True)
    uq_predict.add_argument("--output", type=Path, required=True)
    uq_predict.set_defaults(function=command_uq_routing_predict)

    uq_evaluate = commands.add_parser(
        "uq-routing-evaluate",
        help="Score an already-frozen UQ test prediction cache",
    )
    uq_evaluate.add_argument("--data", type=Path, required=True)
    uq_evaluate.add_argument("--predictions", type=Path, required=True)
    uq_evaluate.add_argument("--references", type=Path, required=True)
    uq_evaluate.add_argument("--output", type=Path, required=True)
    uq_evaluate.add_argument("--seed", type=int, default=42)
    uq_evaluate.set_defaults(function=command_uq_routing_evaluate)

    merge_results = commands.add_parser(
        "merge-dev-results",
        help="Merge candidate and prompting baselines into JSON and Markdown tables",
    )
    merge_results.add_argument("--candidate-experiments", type=Path, required=True)
    merge_results.add_argument("--direct-rag-summary", type=Path, required=True)
    merge_results.add_argument("--qacc-summary", type=Path, required=True)
    merge_results.add_argument("--output", type=Path, required=True)
    merge_results.set_defaults(function=command_merge_dev_results)

    predict = commands.add_parser(
        "predict",
        help="Freeze candidate predictions on test inputs without loading references",
    )
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--extractions", type=Path, required=True)
    predict.add_argument("--method-name", required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--answer-match-threshold", type=float, default=0.8)
    predict.add_argument("--near-duplicate-jaccard", type=float, default=0.88)
    predict.set_defaults(function=command_predict)

    predict_baseline = commands.add_parser(
        "predict-baseline",
        help="Freeze a candidate-support baseline without loading references",
    )
    predict_baseline.add_argument("--data", type=Path, required=True)
    predict_baseline.add_argument("--extractions", type=Path, required=True)
    predict_baseline.add_argument(
        "--method",
        choices=("raw-passage-majority", "independent-source-majority"),
        required=True,
    )
    predict_baseline.add_argument("--output", type=Path, required=True)
    predict_baseline.add_argument("--answer-match-threshold", type=float, default=0.8)
    predict_baseline.add_argument("--near-duplicate-jaccard", type=float, default=0.88)
    predict_baseline.set_defaults(function=command_predict_baseline)

    score = commands.add_parser(
        "score",
        help="Score an already frozen candidate prediction file against references",
    )
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--references", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(function=command_score)

    evaluate = commands.add_parser(
        "evaluate",
        help="Legacy combined prediction/scoring command; prefer predict then score",
    )
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--references", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--extractions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--answer-match-threshold", type=float, default=0.8)
    evaluate.add_argument("--near-duplicate-jaccard", type=float, default=0.88)
    evaluate.set_defaults(function=command_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
