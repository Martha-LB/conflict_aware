#!/usr/bin/env bash
set -euo pipefail

# Rebuild the final comparison from frozen caches. No API calls are made.
export PATH="$PWD/.venv/bin:$PATH"

DATA="${DATA:-data/derived/qacc-v1.2/test.official.inputs.jsonl}"
REFERENCES="${REFERENCES:-data/derived/qacc-v1.2/test.official.references.jsonl}"
DIRECT="${DIRECT:-artifacts/test.official.gpt-6-luna.direct-rag.v2.answers.jsonl}"
ARSR="${ARSR:-artifacts/test.official.uq-luna-first.v1.predictions.jsonl}"
QACC_EXPLAIN="${QACC_EXPLAIN:-artifacts/test.official.gpt-6-luna.qacc-prompt.v2.answers.jsonl}"
ORACLE="${ORACLE:-artifacts/test.official.candidate-pool.v1.jsonl}"
OUTPUT="${OUTPUT:-artifacts/test.official.identifiability.v1.json}"

for required in "$DATA" "$REFERENCES" "$DIRECT" "$ARSR" "$QACC_EXPLAIN" "$ORACLE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

uacaa identifiability-report \
  --data "$DATA" \
  --references "$REFERENCES" \
  --prediction direct_rag "$DIRECT" answer \
  --prediction arsr "$ARSR" answer \
  --prediction qacc_explain "$QACC_EXPLAIN" answer \
  --prediction candidate_oracle "$ORACLE" oracle_candidate \
  --primary-name direct_rag \
  --resolver-name arsr \
  --oracle-name candidate_oracle \
  --output "$OUTPUT"

echo "JSON report: $OUTPUT"
echo "Readable table: ${OUTPUT%.json}.md"
echo "Per-example rows: ${OUTPUT%.json}.rows.jsonl"
