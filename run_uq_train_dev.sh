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

TRAIN_DATA="data/derived/qacc-v1/train.strict.jsonl"
DEV_DATA="data/derived/qacc-v1/dev.strict.jsonl"
TRAIN_EXTRACTIONS="artifacts/train.gpt-6-luna.v3.extractions.jsonl"
DEV_EXTRACTIONS="artifacts/dev.gpt-6-luna.v3.extractions.jsonl"
TRAIN_LUNA="artifacts/train.gpt-6-luna.direct-rag.v1.answers.jsonl"
TRAIN_MINI="artifacts/train.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl"
DEV_LUNA="artifacts/dev.gpt-6-luna.direct-rag.v2.answers.jsonl"
DEV_MINI="artifacts/dev.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl"

if [[ "${SKIP_GENERATION:-0}" != "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set in this shell." >&2
    echo "Export it in the environment, then rerun; do not paste it into this script." >&2
    exit 1
  fi

  "$UACAA" answer-baseline \
    --data "$TRAIN_DATA" \
    --model gpt-6-luna \
    --reasoning-effort none \
    --style direct-rag \
    --output "$TRAIN_LUNA"

  "$UACAA" answer-baseline \
    --data "$TRAIN_DATA" \
    --model gpt-4o-mini-2024-07-18 \
    --reasoning-effort omit \
    --style direct-rag \
    --output "$TRAIN_MINI"
fi

for required_file in \
  "$TRAIN_DATA" "$DEV_DATA" "$TRAIN_EXTRACTIONS" "$DEV_EXTRACTIONS" \
  "$TRAIN_LUNA" "$TRAIN_MINI" "$DEV_LUNA" "$DEV_MINI"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Missing or empty required file: $required_file" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --train-data "$TRAIN_DATA"
  --dev-data "$DEV_DATA"
  --train-extractions "$TRAIN_EXTRACTIONS"
  --dev-extractions "$DEV_EXTRACTIONS"
  --prior-scale "${PRIOR_SCALE:-0.75}"
  --intervention-quantile "${INTERVENTION_QUANTILE:-0.75}"
  --delta-margin "${DELTA_MARGIN:-0.10}"
  --selective-coverage "${SELECTIVE_COVERAGE:-0.80}"
  --pre-feature-mode "${PRE_FEATURE_MODE:-answer-only}"
  --seed "${SEED:-42}"
)

"$UACAA" uq-routing-train-dev \
  "${COMMON_ARGS[@]}" \
  --train-primary-predictions "$TRAIN_LUNA" \
  --train-secondary-predictions "$TRAIN_MINI" \
  --dev-primary-predictions "$DEV_LUNA" \
  --dev-secondary-predictions "$DEV_MINI" \
  --primary-name gpt-6-luna \
  --secondary-name gpt-4o-mini-2024-07-18 \
  --output artifacts/dev.uq-trainfit.luna-first.v1.jsonl

"$UACAA" uq-routing-train-dev \
  "${COMMON_ARGS[@]}" \
  --train-primary-predictions "$TRAIN_MINI" \
  --train-secondary-predictions "$TRAIN_LUNA" \
  --dev-primary-predictions "$DEV_MINI" \
  --dev-secondary-predictions "$DEV_LUNA" \
  --primary-name gpt-4o-mini-2024-07-18 \
  --secondary-name gpt-6-luna \
  --output artifacts/dev.uq-trainfit.mini-first.v1.jsonl

"$UACAA" compare-uq-routing \
  --luna-first artifacts/dev.uq-trainfit.luna-first.v1.summary.json \
  --mini-first artifacts/dev.uq-trainfit.mini-first.v1.summary.json
