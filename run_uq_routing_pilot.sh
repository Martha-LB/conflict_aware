#!/usr/bin/env bash
set -euo pipefail

# Offline labelled-dev diagnostic. It reuses cached answers and extractions and
# never calls an LLM API. Do not report it as a frozen-test result.
uacaa uq-routing-cv \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --luna-predictions artifacts/dev.gpt-6-luna.direct-rag.v2.answers.jsonl \
  --mini-predictions artifacts/dev.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --folds "${FOLDS:-5}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --prior-scale "${PRIOR_SCALE:-0.75}" \
  --intervention-quantile "${INTERVENTION_QUANTILE:-0.75}" \
  --delta-margin "${DELTA_MARGIN:-0.10}" \
  --selective-coverage "${SELECTIVE_COVERAGE:-0.80}" \
  --pre-feature-mode "${PRE_FEATURE_MODE:-full}" \
  --seed "${SEED:-42}" \
  --output "${OUTPUT:-artifacts/dev.uq-routing.v1.oof.jsonl}"
