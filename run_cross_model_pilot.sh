#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?Export OPENAI_API_KEY in this terminal before running.}"
: "${SECONDARY_MODEL:?Set SECONDARY_MODEL to an exact independent generator model ID.}"
: "${VERIFIER_MODEL:?Set VERIFIER_MODEL to an exact verifier model ID.}"

LIMIT="${LIMIT:-20}"
SECONDARY_REASONING_EFFORT="${SECONDARY_REASONING_EFFORT:-omit}"
VERIFIER_REASONING_EFFORT="${VERIFIER_REASONING_EFFORT:-none}"

if [[ "$SECONDARY_MODEL" == "$VERIFIER_MODEL" ]]; then
  echo "Warning: generator and verifier use the same model; this weakens the model-diversity test." >&2
fi
if [[ "$SECONDARY_MODEL" == "gpt-6-luna" ]]; then
  echo "Warning: secondary generator matches the frozen primary model gpt-6-luna." >&2
fi
if [[ "$VERIFIER_MODEL" == "gpt-6-luna" ]]; then
  echo "Warning: verifier matches the frozen primary model gpt-6-luna." >&2
fi

# Use positional parameters instead of empty arrays. macOS ships Bash 3.2,
# where expanding an empty array under `set -u` raises an unbound-variable error.
set --
if [[ "${RESET_SECONDARY:-0}" == "1" ]]; then
  set -- "$@" --no-resume
fi
if [[ "$LIMIT" != "all" ]]; then
  set -- "$@" --limit "$LIMIT"
fi

uacaa cross-model-generate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --primary-generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --risk-model artifacts/two-stage-evidence-aware.v3.model.json \
  --risk-mode detector \
  --model "$SECONDARY_MODEL" \
  --reasoning-effort "$SECONDARY_REASONING_EFFORT" \
  "$@" \
  --output artifacts/dev.cross-model.v4.secondary.jsonl

set --
if [[ "$LIMIT" != "all" ]]; then
  set -- "$@" --limit "$LIMIT"
fi

uacaa cross-model-verify \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --primary-generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --secondary-generations artifacts/dev.cross-model.v4.secondary.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --model "$VERIFIER_MODEL" \
  --reasoning-effort "$VERIFIER_REASONING_EFFORT" \
  --mode conservative \
  "$@" \
  --output artifacts/dev.cross-model.v4.conservative.jsonl

uacaa cross-model-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.cross-model.v4.conservative.jsonl \
  "$@" \
  --output artifacts/dev.cross-model.v4.conservative.scored.jsonl
