#!/usr/bin/env bash
set -euo pipefail

# QACC's released train/dev rows omit the multi-reference AmbigQA aliases used
# by the paper's evaluator.  Fetch the small official AmbigNQ light archive and
# verify the checksum published with the upstream dataset.
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="${ROOT_DIR}/data/raw/ambigqa-light"
ARCHIVE="${DEST_DIR}/ambignq_light.zip"
EXPECTED_SHA256="3f5dada69dec05cef1533a64945cd7bafde1aa94b0cdd6fa9a22f881206220db"

mkdir -p "${DEST_DIR}"
curl -L --fail --show-error \
  "https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip" \
  -o "${ARCHIVE}"

ACTUAL_SHA256="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "AmbigQA checksum mismatch: ${ACTUAL_SHA256}" >&2
  exit 1
fi

unzip -o -q "${ARCHIVE}" train_light.json dev_light.json LICENSE -d "${DEST_DIR}"

echo "AmbigQA references ready in ${DEST_DIR}"
