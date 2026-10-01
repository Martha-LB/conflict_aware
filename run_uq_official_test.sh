#!/usr/bin/env bash
set -euo pipefail

if command -v uacaa >/dev/null 2>&1; then
  UACAA="$(command -v uacaa)"
elif [[ -x .venv/bin/uacaa ]]; then
  UACAA="$PWD/.venv/bin/uacaa"
else
  echo "uacaa is unavailable; activate .venv or run pip install -e '.[dev]'" >&2
  exit 1
fi

TEST_DATA="data/derived/qacc-v1/test.official.inputs.jsonl"
TEST_REFERENCES="data/derived/qacc-v1/test.official.references.jsonl"
TEST_EXTRACTIONS="artifacts/test.official.gpt-6-luna.v3.extractions.jsonl"
TEST_LUNA="artifacts/test.official.gpt-6-luna.direct-rag.v2.answers.jsonl"
TEST_MINI="artifacts/test.official.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl"
FROZEN_MODEL="artifacts/dev.uq-trainfit.luna-first.v1.model.json"
PREDICTIONS="artifacts/test.official.uq-luna-first.v1.predictions.jsonl"
SCORED="artifacts/test.official.uq-luna-first.v1.scored.jsonl"

if [[ "${SKIP_GENERATION:-0}" != "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set in this shell." >&2
    echo "Export it in the environment, then rerun; do not paste it into this script." >&2
    exit 1
  fi
  "$UACAA" answer-baseline \
    --data "$TEST_DATA" \
    --model gpt-4o-mini-2024-07-18 \
    --reasoning-effort omit \
    --style direct-rag \
    --output "$TEST_MINI"
fi

for required_file in \
  "$TEST_DATA" "$TEST_REFERENCES" "$TEST_EXTRACTIONS" \
  "$TEST_LUNA" "$TEST_MINI" "$FROZEN_MODEL"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Missing or empty required file: $required_file" >&2
    exit 1
  fi
done

if [[ -e "$PREDICTIONS" ]]; then
  if [[ ! -s "${PREDICTIONS}.manifest.json" ]]; then
    echo "Frozen predictions exist without a manifest: $PREDICTIONS" >&2
    exit 1
  fi
  echo "Reusing already frozen predictions: $PREDICTIONS"
else
  "$UACAA" uq-routing-predict \
    --data "$TEST_DATA" \
    --primary-predictions "$TEST_LUNA" \
    --secondary-predictions "$TEST_MINI" \
    --extractions "$TEST_EXTRACTIONS" \
    --model "$FROZEN_MODEL" \
    --output "$PREDICTIONS"
fi

if [[ -e "$SCORED" ]]; then
  echo "Official scoring already exists: ${SCORED%.jsonl}.summary.json"
else
  "$UACAA" uq-routing-evaluate \
    --data "$TEST_DATA" \
    --predictions "$PREDICTIONS" \
    --references "$TEST_REFERENCES" \
    --output "$SCORED"
fi
