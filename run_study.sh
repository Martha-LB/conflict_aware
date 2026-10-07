#!/usr/bin/env bash
set -euo pipefail

# Run the complete final study. Existing API caches are reused.
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv. Follow the README setup first." >&2
  exit 2
fi
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Fixed models keep cache names and experiment settings consistent.
PRIMARY_MODEL="gpt-6-luna"
SECONDARY_MODEL="gpt-4o-mini-2024-07-18"
DATA_DIR="${DATA_DIR:-data/derived/qacc-v1.2}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"
QACC_RAW="${QACC_RAW:-data/raw/ConflictQA_Dataset.json}"
AMBIGQA_DIR="${AMBIGQA_DIR:-data/raw/ambigqa-light}"

mkdir -p "$(dirname "$QACC_RAW")" "$AMBIGQA_DIR" "$DATA_DIR" "$ARTIFACT_DIR"

# Download the two public datasets only when they are missing.
if [[ ! -f "$QACC_RAW" ]]; then
  curl -L --fail --show-error \
    "https://raw.githubusercontent.com/amazon-science/qa-with-conflicting-context/main/data/ConflictQA_Dataset.json" \
    -o "$QACC_RAW"
fi

if [[ ! -f "$AMBIGQA_DIR/train_light.json" || ! -f "$AMBIGQA_DIR/dev_light.json" ]]; then
  AMBIGQA_ZIP="$AMBIGQA_DIR/ambignq_light.zip"
  curl -L --fail --show-error \
    "https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip" \
    -o "$AMBIGQA_ZIP"
  unzip -o -q "$AMBIGQA_ZIP" train_light.json dev_light.json LICENSE \
    -d "$AMBIGQA_DIR"
fi

run_uacaa() {
  "$PYTHON" -m uacaa.cli "$@"
}

# Prepare leakage-safe train, dev, and test files.
run_uacaa prepare \
  --raw "$QACC_RAW" \
  --output-dir "$DATA_DIR" \
  --revision "amazon-science/main" \
  --ambigqa "$AMBIGQA_DIR/train_light.json" "$AMBIGQA_DIR/dev_light.json"

TRAIN="$DATA_DIR/train.strict.jsonl"
DEV="$DATA_DIR/dev.strict.jsonl"
TEST="$DATA_DIR/test.official.inputs.jsonl"
REFERENCES="$DATA_DIR/test.official.references.jsonl"

TRAIN_EXTRACTIONS="$ARTIFACT_DIR/train.gpt-6-luna.v3.extractions.jsonl"
DEV_EXTRACTIONS="$ARTIFACT_DIR/dev.gpt-6-luna.v3.extractions.jsonl"
TEST_EXTRACTIONS="$ARTIFACT_DIR/test.official.gpt-6-luna.v3.extractions.jsonl"

TRAIN_PRIMARY="$ARTIFACT_DIR/train.gpt-6-luna.direct-rag.v1.answers.jsonl"
DEV_PRIMARY="$ARTIFACT_DIR/dev.gpt-6-luna.direct-rag.v2.answers.jsonl"
TEST_PRIMARY="$ARTIFACT_DIR/test.official.gpt-6-luna.direct-rag.v2.answers.jsonl"
TRAIN_SECONDARY="$ARTIFACT_DIR/train.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl"
DEV_SECONDARY="$ARTIFACT_DIR/dev.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl"
TEST_SECONDARY="$ARTIFACT_DIR/test.official.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl"
TEST_EXPLAIN="$ARTIFACT_DIR/test.official.gpt-6-luna.qacc-prompt.v2.answers.jsonl"

# A new run needs an API key. Complete caches need no new model calls.
for cache in \
  "$TRAIN_EXTRACTIONS" "$DEV_EXTRACTIONS" "$TEST_EXTRACTIONS" \
  "$TRAIN_PRIMARY" "$DEV_PRIMARY" "$TEST_PRIMARY" \
  "$TRAIN_SECONDARY" "$DEV_SECONDARY" "$TEST_SECONDARY" "$TEST_EXPLAIN"; do
  if [[ ! -f "$cache" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY before generating missing model outputs." >&2
    exit 2
  fi
done

# Extract one answer from each context.
for item in \
  "$TRAIN|$TRAIN_EXTRACTIONS" \
  "$DEV|$DEV_EXTRACTIONS" \
  "$TEST|$TEST_EXTRACTIONS"; do
  data="${item%%|*}"
  output="${item#*|}"
  run_uacaa extract --data "$data" --model "$PRIMARY_MODEL" \
    --reasoning-effort none --output "$output"
done

run_baseline() {
  local data="$1" model="$2" effort="$3" style="$4" output="$5"
  run_uacaa answer-baseline --data "$data" --model "$model" \
    --reasoning-effort "$effort" --style "$style" --output "$output"
}

# Generate the two Direct-RAG answers used by ARSR.
run_baseline "$TRAIN" "$PRIMARY_MODEL" none direct-rag "$TRAIN_PRIMARY"
run_baseline "$DEV" "$PRIMARY_MODEL" none direct-rag "$DEV_PRIMARY"
run_baseline "$TEST" "$PRIMARY_MODEL" none direct-rag "$TEST_PRIMARY"
run_baseline "$TRAIN" "$SECONDARY_MODEL" omit direct-rag "$TRAIN_SECONDARY"
run_baseline "$DEV" "$SECONDARY_MODEL" omit direct-rag "$DEV_SECONDARY"
run_baseline "$TEST" "$SECONDARY_MODEL" omit direct-rag "$TEST_SECONDARY"
run_baseline "$TEST" "$PRIMARY_MODEL" none qacc-explain-then-answer "$TEST_EXPLAIN"

# Fit on train and dev, then freeze the model before test evaluation.
ARSR_DEV="$ARTIFACT_DIR/dev.uq-trainfit.luna-first.v1.jsonl"
run_uacaa arsr-train-dev \
  --train-data "$TRAIN" --dev-data "$DEV" \
  --train-primary-predictions "$TRAIN_PRIMARY" \
  --train-secondary-predictions "$TRAIN_SECONDARY" \
  --dev-primary-predictions "$DEV_PRIMARY" \
  --dev-secondary-predictions "$DEV_SECONDARY" \
  --train-extractions "$TRAIN_EXTRACTIONS" \
  --dev-extractions "$DEV_EXTRACTIONS" \
  --primary-name "$PRIMARY_MODEL" --secondary-name "$SECONDARY_MODEL" \
  --pre-feature-mode answer-only --output "$ARSR_DEV"

ARSR_MODEL="${ARSR_DEV%.jsonl}.model.json"
TEST_ARSR="$ARTIFACT_DIR/test.official.uq-luna-first.v1.predictions.jsonl"
if [[ ! -f "$TEST_ARSR" ]]; then
  run_uacaa arsr-predict \
    --data "$TEST" --primary-predictions "$TEST_PRIMARY" \
    --secondary-predictions "$TEST_SECONDARY" \
    --extractions "$TEST_EXTRACTIONS" --model "$ARSR_MODEL" \
    --output "$TEST_ARSR"
fi

TEST_SCORED="$ARTIFACT_DIR/test.official.uq-luna-first.v1.scored.jsonl"
if [[ ! -f "$TEST_SCORED" ]]; then
  run_uacaa arsr-evaluate --data "$TEST" --predictions "$TEST_ARSR" \
    --references "$REFERENCES" --output "$TEST_SCORED"
fi

# Build a reference-free candidate set for the diagnostic oracle.
CANDIDATES="$ARTIFACT_DIR/test.official.candidate-pool.v1.jsonl"
if [[ ! -f "$CANDIDATES" ]]; then
  run_uacaa candidate-pool \
    --data "$TEST" --primary-predictions "$TEST_PRIMARY" \
    --secondary-predictions "$TEST_SECONDARY" \
    --extractions "$TEST_EXTRACTIONS" --output "$CANDIDATES"
fi

# Create the final table used in the README.
FINAL="$ARTIFACT_DIR/test.official.identifiability.v1.json"
run_uacaa identifiability-report \
  --data "$TEST" --references "$REFERENCES" \
  --prediction direct_rag "$TEST_PRIMARY" answer \
  --prediction arsr "$TEST_ARSR" answer \
  --prediction qacc_explain "$TEST_EXPLAIN" answer \
  --prediction candidate_oracle "$CANDIDATES" oracle_candidate \
  --primary-name direct_rag --resolver-name arsr \
  --oracle-name candidate_oracle --output "$FINAL"

echo "Done. Final report: $FINAL"
echo "Readable table: ${FINAL%.json}.md"
