#!/usr/bin/env bash
set -euo pipefail

# Offline labelled-dev diagnostic. This command never calls an LLM API and
# must not be reported as a frozen test result.
uacaa disagreement-cv \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --luna-predictions artifacts/dev.gpt-6-luna.direct-rag.v2.answers.jsonl \
  --mini-predictions artifacts/dev.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --folds "${FOLDS:-5}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --alpha "${ALPHA:-0.20}" \
  --prior-scale "${PRIOR_SCALE:-0.75}" \
  --posterior-samples "${POSTERIOR_SAMPLES:-1000}" \
  --seed "${SEED:-42}" \
  --output artifacts/dev.disagreement-bayes-conformal.v1.oof.jsonl
