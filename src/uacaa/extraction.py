from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from dataclasses import replace
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

from .normalization import is_no_answer
from .schemas import EvidenceObservation, PreparedExample


SYSTEM_PROMPT = """You extract candidate answers from web snippets for factual QA.
Treat every context independently even though they are shown together. Never compare contexts,
vote across them, suppress a minority answer, or decide which candidate is globally correct.
You MUST return exactly one assessment for every supplied context_index, including irrelevant
and unanswerable contexts. Never omit a context. Set answerable=false when that context alone
does not assert an answer; then use empty strings for candidate_answer and evidence_quote and
zero for all four scores. Set answerable=true when that context alone asserts a plausible answer.

The candidate_answer must be the shortest standalone answer span that directly fills the
question's requested slot. Preserve the source wording when possible. Return an entity, number,
date, short noun phrase, or necessary list--never a sentence, explanation, predicate, unit that
the question already supplies, or surrounding claim. Examples: for "How many episodes?" return
"38", not "38 episodes"; for "Who owns the building?" return "Vornado Realty Trust", not
"Vornado Realty Trust owns the building"; for "Who selects the nominee?" return "the
President", not "the president selects the nominee". Match the grammatical answer type: a
"who" question needs a person or organization, a "how many" question needs a number, and a
plural question may require a short list.

The evidence_quote must be a verbatim, contiguous quote copied from that same context. Return
JSON only with an `observations` array. Each assessment must contain context_index, answerable,
candidate_answer, evidence_quote, directness, entailment, answer_type_match, and temporal_fit.
Scores are numbers in [0,1]. Never use source reputation as proof of correctness."""

EXTRACTION_PROTOCOL_VERSION = "3.2"


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "context_index": {"type": "integer"},
                    "answerable": {"type": "boolean"},
                    "candidate_answer": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "directness": {"type": "number", "minimum": 0, "maximum": 1},
                    "entailment": {"type": "number", "minimum": 0, "maximum": 1},
                    "answer_type_match": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "temporal_fit": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "context_index",
                    "answerable",
                    "candidate_answer",
                    "evidence_quote",
                    "directness",
                    "entailment",
                    "answer_type_match",
                    "temporal_fit",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["observations"],
    "additionalProperties": False,
}


class CandidateExtractor(Protocol):
    def extract(self, example: PreparedExample) -> list[EvidenceObservation]: ...


def extraction_prompt(example: PreparedExample) -> str:
    rendered = "\n\n".join(
        f"[{context.index}] source={context.source or 'unknown'}\n{context.text}"
        for context in example.contexts
    )
    return f"Question: {example.question}\n\nContexts:\n{rendered}"


def parse_context_assessments(
    text: str,
    context_count: int,
    *,
    allow_missing: bool = False,
    quarantine_malformed_answerable: bool = False,
    quarantine_structural_errors: bool = False,
) -> tuple[list[EvidenceObservation], list[dict]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    payload = json.loads(cleaned)
    items = payload.get("observations", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Extractor response must contain an observations list")
    observations: list[EvidenceObservation] = []
    assessments: list[dict] = []
    seen: set[int] = set()
    duplicate_indices: set[int] = set()
    for item in items:
        index = int(item["context_index"])
        if index < 0 or index >= context_count:
            if quarantine_structural_errors:
                continue
            raise ValueError(f"Invalid or duplicate context_index: {index}")
        if index in seen:
            if not quarantine_structural_errors:
                raise ValueError(f"Invalid or duplicate context_index: {index}")
            duplicate_indices.add(index)
            assessments = [
                assessment
                for assessment in assessments
                if assessment["context_index"] != index
            ]
            observations = [
                observation
                for observation in observations
                if observation.context_index != index
            ]
            continue
        seen.add(index)
        answerable = item.get("answerable")
        if not isinstance(answerable, bool):
            if quarantine_structural_errors:
                assessments.append(
                    {
                        "context_index": index,
                        "answerable": False,
                        "candidate_answer": "",
                        "evidence_quote": "",
                        "directness": 0.0,
                        "entailment": 0.0,
                        "answer_type_match": 0.0,
                        "temporal_fit": 0.0,
                        "invalid_answerable_type": True,
                    }
                )
                continue
            raise ValueError(f"answerable must be boolean for context_index {index}")
        answer = str(item.get("candidate_answer", "")).strip()
        quote = str(item.get("evidence_quote", "")).strip()
        assessment = {
            "context_index": index,
            "answerable": answerable,
            "candidate_answer": answer,
            "evidence_quote": quote,
            "directness": float(item.get("directness", 0.0)),
            "entailment": float(item.get("entailment", 0.0)),
            "answer_type_match": float(item.get("answer_type_match", 0.0)),
            "temporal_fit": float(item.get("temporal_fit", 0.0)),
        }
        assessments.append(assessment)
        if not answerable:
            continue
        if is_no_answer(answer) or not quote:
            if quarantine_malformed_answerable:
                assessment["malformed_answerable"] = True
                continue
            raise ValueError(
                f"Answerable context_index {index} requires an answer and evidence quote"
            )
        observations.append(
            EvidenceObservation(
                context_index=index,
                candidate_answer=answer,
                evidence_quote=quote,
                directness=assessment["directness"],
                entailment=assessment["entailment"],
                answer_type_match=assessment["answer_type_match"],
                temporal_fit=assessment["temporal_fit"],
            )
        )
    missing = sorted(set(range(context_count)) - seen)
    if missing and not allow_missing:
        raise ValueError(f"Extractor omitted context_indices: {missing}")
    for index in missing:
        assessments.append(
            {
                "context_index": index,
                "answerable": False,
                "candidate_answer": "",
                "evidence_quote": "",
                "directness": 0.0,
                "entailment": 0.0,
                "answer_type_match": 0.0,
                "temporal_fit": 0.0,
                "omitted_by_model": True,
            }
        )
    for index in sorted(duplicate_indices):
        assessments.append(
            {
                "context_index": index,
                "answerable": False,
                "candidate_answer": "",
                "evidence_quote": "",
                "directness": 0.0,
                "entailment": 0.0,
                "answer_type_match": 0.0,
                "temporal_fit": 0.0,
                "duplicate_assessment": True,
            }
        )
    return (
        sorted(observations, key=lambda item: item.context_index),
        sorted(assessments, key=lambda item: item["context_index"]),
    )


def parse_extraction_response(text: str, context_count: int) -> list[EvidenceObservation]:
    observations, _ = parse_context_assessments(text, context_count)
    return observations


def evidence_quote_is_verbatim(quote: str, context: str) -> bool:
    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"\s+", " ", value).strip()

    normalized_quote = normalize(quote)
    return bool(normalized_quote) and normalized_quote in normalize(context)


def align_evidence_quote(quote: str, context: str) -> str | None:
    """Return the exact context span when only case/space/punctuation differs.

    Alignment requires the complete alphanumeric token sequence to occur
    contiguously and in order. It never performs semantic or fuzzy rewriting.
    """

    def token_spans(value: str) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        for match in re.finditer(r"[^\W_]+", value, flags=re.UNICODE):
            token = unicodedata.normalize("NFKC", match.group()).casefold()
            spans.append((token, match.start(), match.end()))
        return spans

    quote_tokens = [token for token, _, _ in token_spans(quote)]
    context_spans = token_spans(context)
    context_tokens = [token for token, _, _ in context_spans]
    if not quote_tokens or len(quote_tokens) > len(context_tokens):
        return None
    width = len(quote_tokens)
    for start in range(len(context_tokens) - width + 1):
        if context_tokens[start : start + width] == quote_tokens:
            return context[context_spans[start][1] : context_spans[start + width - 1][2]]
    return None


class OpenAICompatibleExtractor:
    def __init__(
        self,
        model: str,
        *,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        retries: int = 3,
        structured_output: bool = True,
        reasoning_effort: str | None = "none",
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.retries = retries
        self.structured_output = structured_output
        self.reasoning_effort = reasoning_effort
        self.last_assessments: list[dict] | None = None
        self.last_response_metadata: dict | None = None

    def extract(self, example: PreparedExample) -> list[EvidenceObservation]:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set")
        response_format = (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "qacc_candidate_extraction",
                    "strict": True,
                    "schema": EXTRACTION_SCHEMA,
                },
            }
            if self.structured_output
            else {"type": "json_object"}
        )
        payload = {
            "model": self.model,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": extraction_prompt(example)},
            ],
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        last_error: Exception | None = None
        self.last_assessments = None
        self.last_response_metadata = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                content = response_payload["choices"][0]["message"]["content"]
                observations, assessments = parse_context_assessments(
                    content,
                    len(example.contexts),
                    allow_missing=attempt + 1 >= self.retries,
                    quarantine_malformed_answerable=attempt + 1 >= self.retries,
                    quarantine_structural_errors=attempt + 1 >= self.retries,
                )
                aligned_observations: list[EvidenceObservation] = []
                invalid_quotes: list[int] = []
                assessment_by_index = {
                    assessment["context_index"]: assessment
                    for assessment in assessments
                }
                for assessment in assessments:
                    assessment["evidence_quote_raw"] = assessment["evidence_quote"]
                    assessment["accepted_for_candidate_aggregation"] = False
                    assessment["validation_status"] = (
                        "omitted_by_model"
                        if assessment.get("omitted_by_model")
                        else (
                            "malformed_answerable"
                            if assessment.get("malformed_answerable")
                            else (
                                "duplicate_assessment"
                                if assessment.get("duplicate_assessment")
                                else (
                                    "invalid_answerable_type"
                                    if assessment.get("invalid_answerable_type")
                                    else "not_answerable"
                                )
                            )
                        )
                    )
                for observation in observations:
                    context = example.contexts[observation.context_index].text
                    raw_quote = observation.evidence_quote
                    aligned_quote = align_evidence_quote(raw_quote, context)
                    assessment = assessment_by_index[observation.context_index]
                    if aligned_quote is None:
                        invalid_quotes.append(observation.context_index)
                        assessment["validation_status"] = "invalid_evidence_quote"
                        continue
                    status = (
                        "verbatim"
                        if evidence_quote_is_verbatim(raw_quote, context)
                        else "token_aligned"
                    )
                    assessment["evidence_quote"] = aligned_quote
                    assessment["accepted_for_candidate_aggregation"] = True
                    assessment["validation_status"] = status
                    aligned_observations.append(
                        replace(observation, evidence_quote=aligned_quote)
                    )
                if invalid_quotes and attempt + 1 < self.retries:
                    raise ValueError(
                        "Non-verbatim evidence_quote for context_indices: "
                        + str(invalid_quotes)
                    )
                self.last_assessments = assessments
                self.last_response_metadata = {
                    "response_id": response_payload.get("id"),
                    "response_model": response_payload.get("model"),
                    "system_fingerprint": response_payload.get("system_fingerprint"),
                    "usage": response_payload.get("usage"),
                    "validation_failures": invalid_quotes,
                    "omitted_context_indices": [
                        assessment["context_index"]
                        for assessment in assessments
                        if assessment.get("omitted_by_model")
                    ],
                    "malformed_answerable_context_indices": [
                        assessment["context_index"]
                        for assessment in assessments
                        if assessment.get("malformed_answerable")
                    ],
                    "duplicate_context_indices": [
                        assessment["context_index"]
                        for assessment in assessments
                        if assessment.get("duplicate_assessment")
                    ],
                    "invalid_answerable_type_context_indices": [
                        assessment["context_index"]
                        for assessment in assessments
                        if assessment.get("invalid_answerable_type")
                    ],
                }
                return aligned_observations
            except urllib.error.HTTPError as error:
                raw_error = error.read().decode("utf-8", errors="replace")
                try:
                    error_payload = json.loads(raw_error)
                    detail = error_payload.get("error", {}).get("message", raw_error)
                except json.JSONDecodeError:
                    detail = raw_error
                api_error = RuntimeError(
                    f"OpenAI API HTTP {error.code}: {detail or error.reason}"
                )
                # Invalid requests and authentication/permission failures do not
                # become valid when repeated. Preserve quota/rate-limit and server
                # errors as retryable because they may be transient.
                if error.code not in {408, 409, 429} and error.code < 500:
                    raise api_error from error
                last_error = api_error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                TypeError,
                AttributeError,
                ValueError,
            ) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Candidate extraction failed after {self.retries} attempts; "
            f"last error: {last_error}"
        ) from last_error


def annotated_observations(example: PreparedExample) -> list[EvidenceObservation]:
    """Use train/dev annotations for diagnostics only, never for test inference."""
    if example.split == "test":
        raise ValueError("Oracle QACC attribution is forbidden on the test split")
    if not example.supervision:
        return []
    observations: list[EvidenceObservation] = []
    for candidate in example.supervision.get("candidates", []):
        for index in candidate.get("context_indices", []):
            observations.append(
                EvidenceObservation(
                    context_index=int(index),
                    candidate_answer=str(candidate["answer"]),
                    directness=1.0,
                    entailment=1.0,
                    answer_type_match=1.0,
                    temporal_fit=0.5,
                )
            )
    return observations


def cache_key(example: PreparedExample, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "question": example.question,
            "contexts": [context.text for context in example.contexts],
            "sources": [context.source for context in example.contexts],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_extraction_cache(
    examples: Sequence[PreparedExample],
    extractor: CandidateExtractor,
    output_path: Path,
    *,
    resume: bool = True,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")

    completed: set[str] = set()
    seed = temporary if resume and temporary.exists() else output_path
    existing = ""
    if resume and seed.exists():
        existing = seed.read_text(encoding="utf-8")
        for line in existing.splitlines():
            if line.strip():
                completed.add(str(json.loads(line)["question_id"]))

    requested_ids = {example.question_id for example in examples}
    # Keep only records relevant to this invocation, so reusing an output path
    # with a different slice cannot silently contaminate the cache.
    if existing:
        retained = []
        for line in existing.splitlines():
            if line.strip() and str(json.loads(line)["question_id"]) in requested_ids:
                retained.append(line)
        existing = "\n".join(retained) + ("\n" if retained else "")
        completed &= requested_ids

    temporary.write_text(existing, encoding="utf-8")
    extracted = 0
    with temporary.open("a", encoding="utf-8") as handle:
        for example in examples:
            if example.question_id in completed:
                continue
            observations = extractor.extract(example)
            record = {
                "question_id": example.question_id,
                "observations": [observation.__dict__ for observation in observations],
            }
            assessments = getattr(extractor, "last_assessments", None)
            response_metadata = getattr(extractor, "last_response_metadata", None)
            if assessments is not None:
                record["context_assessments"] = assessments
            if response_metadata is not None:
                record["response_metadata"] = response_metadata
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed.add(example.question_id)
            extracted += 1
    temporary.replace(output_path)
    return {"requested": len(examples), "reused": len(examples) - extracted, "extracted": extracted}


def read_extraction_cache(path: Path) -> dict[str, list[EvidenceObservation]]:
    cache: dict[str, list[EvidenceObservation]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[str(record["question_id"])] = [
                EvidenceObservation(**item) for item in record.get("observations", [])
            ]
    return cache
