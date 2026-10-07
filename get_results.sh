#!/usr/bin/env bash
set -euo pipefail

# Run every experiment used in the README.
# Finished files are reused.

cd "$(dirname "$0")"
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"

# Stop early when the virtual environment is missing.
if [[ ! -x "$PYTHON" ]]; then
  echo "Create .venv and install the project first." >&2
  exit 2
fi

# Use the current source code.
export PYTHONPATH="$PWD/src"

# These small values are safer on a MacBook.
export TOKENIZERS_PARALLELISM=false
export LOKY_MAX_CPU_COUNT=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Fixed experiment settings.
MODEL="gpt-6-luna"
ORACLE_MODEL="gpt-4o-mini-2024-07-18"
DATA_DIR="data/derived/qacc-v1.2"
ARTIFACT_DIR="artifacts"
RAW_DATA="data/raw/ConflictQA_Dataset.json"
AMBIGQA_DIR="data/raw/ambigqa-light"

mkdir -p "data/raw" "$AMBIGQA_DIR" "$DATA_DIR" "$ARTIFACT_DIR"

# Run one project command.
run() {
  "$PYTHON" -m conflict_maxent.cli "$@"
}

# Download one file only when it is missing.
download() {
  local url="$1"
  local output="$2"
  if [[ ! -f "$output" ]]; then
    curl -L --fail --show-error "$url" -o "$output"
  fi
}

# Download QACC and the public answer aliases.
download \
  "https://raw.githubusercontent.com/amazon-science/qa-with-conflicting-context/main/data/ConflictQA_Dataset.json" \
  "$RAW_DATA"
download \
  "https://raw.githubusercontent.com/shmsw25/AmbigQA/master/data/train_light.json" \
  "$AMBIGQA_DIR/train_light.json"
download \
  "https://raw.githubusercontent.com/shmsw25/AmbigQA/master/data/dev_light.json" \
  "$AMBIGQA_DIR/dev_light.json"

# Build the leakage-safe QACC files.
run prepare \
  --raw "$RAW_DATA" \
  --output-dir "$DATA_DIR" \
  --revision "amazon-science/main" \
  --ambigqa "$AMBIGQA_DIR/train_light.json" "$AMBIGQA_DIR/dev_light.json"

# Main input and reference files.
TEST="$DATA_DIR/test.official.inputs.jsonl"
REFERENCES="$DATA_DIR/test.official.references.jsonl"

# Model outputs.
EXTRACTIONS="$ARTIFACT_DIR/test.official.gpt-6-luna.v3.extractions.jsonl"
DIRECT="$ARTIFACT_DIR/test.official.gpt-6-luna.direct-rag.v2.answers.jsonl"
EXPLAIN="$ARTIFACT_DIR/test.official.gpt-6-luna.qacc-prompt.v2.answers.jsonl"
SECONDARY="$ARTIFACT_DIR/test.official.gpt-4o-mini-2024-07-18.direct-rag.v1.answers.jsonl"

# Passage weights and weighted answers.
WEIGHTS="$ARTIFACT_DIR/test.official.maxent.weights.jsonl"
UNIFORM="$ARTIFACT_DIR/test.official.uniform-weighted.v1.answers.jsonl"
SUPPORT="$ARTIFACT_DIR/test.official.independent-support.answers.jsonl"
MAXENT="$ARTIFACT_DIR/test.official.maxent.answers.jsonl"

# Ask for a key only when a model output is missing.
for file in "$EXTRACTIONS" "$DIRECT" "$EXPLAIN" "$SECONDARY" \
  "$UNIFORM" "$SUPPORT" "$MAXENT"; do
  if [[ ! -f "$file" && -z "${OPENAI_API_KEY:-}" ]]; then
    read -r -s -p "Paste OPENAI_API_KEY: " OPENAI_API_KEY
    echo
    export OPENAI_API_KEY
    break
  fi
done

# Extract one possible answer from every context.
run extract --data "$TEST" --output "$EXTRACTIONS" \
  --model "$MODEL" --reasoning-effort none

# Run the two main LLM baselines.
run answer-baseline --data "$TEST" --output "$DIRECT" \
  --model "$MODEL" --reasoning-effort none --style direct-rag
run answer-baseline --data "$TEST" --output "$EXPLAIN" \
  --model "$MODEL" --reasoning-effort none --style qacc-explain-then-answer

# This second model adds more candidates for the oracle.
run answer-baseline --data "$TEST" --output "$SECONDARY" \
  --model "$ORACLE_MODEL" --reasoning-effort omit --style direct-rag

# Compute Uniform, Independent Support, and Soft MaxEnt weights.
run maxent-build --data "$TEST" --extractions "$EXTRACTIONS" \
  --output "$WEIGHTS" --device cpu --batch-size 8 \
  --uniform-mix 0.1 --cluster-floor 0.05 --slack-penalty 5.0

# Generate one weighted answer file when it is missing.
weighted_answer() {
  local condition="$1"
  local output="$2"
  if [[ ! -f "$output" ]]; then
    run answer-weighted --data "$TEST" --weights "$WEIGHTS" \
      --condition "$condition" --model "$MODEL" \
      --reasoning-effort none --output "$output"
  else
    echo "Reuse $output"
  fi
}

# Only the weights change across these three conditions.
weighted_answer uniform "$UNIFORM"
weighted_answer support-voting "$SUPPORT"
weighted_answer soft-maxent "$MAXENT"

# Build the candidate set before reading the references.
CANDIDATES="$ARTIFACT_DIR/test.official.candidate-pool.v1.jsonl"
if [[ ! -f "$CANDIDATES" ]]; then
  run candidate-pool --data "$TEST" \
    --primary-predictions "$DIRECT" \
    --secondary-predictions "$SECONDARY" \
    --extractions "$EXTRACTIONS" \
    --output "$CANDIDATES"
fi

# References enter only in this final scoring step.
FINAL="$ARTIFACT_DIR/test.official.results.json"
run identifiability-report \
  --data "$TEST" \
  --references "$REFERENCES" \
  --prediction direct_rag "$DIRECT" answer \
  --prediction uniform_weighted "$UNIFORM" answer \
  --prediction independent_support "$SUPPORT" answer \
  --prediction soft_maxent "$MAXENT" answer \
  --prediction qacc_explain "$EXPLAIN" answer \
  --prediction candidate_oracle "$CANDIDATES" oracle_candidate \
  --primary-name direct_rag \
  --resolver-name soft_maxent \
  --oracle-name candidate_oracle \
  --output "$FINAL"

echo "Done: $FINAL"
echo "Table: ${FINAL%.json}.md"
