#!/usr/bin/env bash
set -Eeuo pipefail

# Keep a Mac awake for the complete run. The environment guard prevents recursion.
if command -v caffeinate >/dev/null 2>&1 && [[ "${GCA_CAFFEINATED:-0}" != "1" ]]; then
  exec env GCA_CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="gpt-6-luna"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
UACAA="$SCRIPT_DIR/.venv/bin/uacaa"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
DEV_DATA="data/derived/qacc-v1/dev.strict.jsonl"
TRAIN_DATA="data/derived/qacc-v1/train.strict.jsonl"
TRAIN_GENERATIONS="artifacts/train.gpt-6-luna.joint-generation.v1.jsonl"
DEV_GENERATIONS="artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gca-overnight-$(date '+%Y%m%d-%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'status=$?; printf "\n[overnight] failed at line %s with status %s\n" "$LINENO" "$status"; printf "[overnight] log: %s\n" "$LOG_FILE"; exit "$status"' ERR

if [[ ! -x "$UACAA" || ! -x "$PYTHON" ]]; then
  echo "Missing .venv. Run: python -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before starting the overnight run."
  exit 1
fi
for required in "$TRAIN_DATA" "$DEV_DATA" \
  artifacts/dev.gpt-6-luna.direct-rag.v2.answers.jsonl \
  artifacts/dev.gpt-6-luna.qacc-prompt.v2.answers.jsonl; do
  if [[ ! -f "$required" ]]; then
    echo "Required file is missing: $required"
    exit 1
  fi
done

run_with_retry() {
  local label="$1"
  shift
  local attempt=1
  local maximum=6
  local delay=15
  while true; do
    echo
    echo "[$(date '+%F %T')] $label (attempt $attempt/$maximum)"
    if "$@"; then
      return 0
    fi
    if (( attempt >= maximum )); then
      echo "Giving up after $maximum attempts: $label"
      return 1
    fi
    echo "Command failed; resumable cache retained. Retrying in ${delay}s."
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay + 15))
    if (( delay > 60 )); then
      delay=60
    fi
  done
}

echo "[overnight] project: $SCRIPT_DIR"
echo "[overnight] model: $MODEL"
echo "[overnight] reasoning effort: $REASONING_EFFORT"
echo "[overnight] log: $LOG_FILE"
echo "[overnight] official test is intentionally excluded"

echo
echo "[1/13] Unit and integration tests"
"$PYTHON" -m pytest -q

echo
echo "[2/13] One-call joint generations for all train questions"
run_with_retry "train joint generation" \
  "$UACAA" gca-run \
    --data "$TRAIN_DATA" \
    --model "$MODEL" \
    --reasoning-effort "$REASONING_EFFORT" \
    --mode never \
    --output "$TRAIN_GENERATIONS"

echo
echo "[3/13] Full dev v1 adaptive run (resumes the existing 20-question cache)"
run_with_retry "dev adaptive GCA v1" \
  "$UACAA" gca-run \
    --data "$DEV_DATA" \
    --model "$MODEL" \
    --reasoning-effort "$REASONING_EFFORT" \
    --mode adaptive \
    --output "$DEV_GENERATIONS"

echo
echo "[4/13] Fit Dirichlet relation model with dependency correction"
"$UACAA" gca-map-fit \
  --train "$TRAIN_DATA" \
  --train-generations "$TRAIN_GENERATIONS" \
  --output artifacts/gca-dirichlet-map.dep.v2.model.json

echo
echo "[5/13] Fit no-dependency-correction ablation"
"$UACAA" gca-map-fit \
  --train "$TRAIN_DATA" \
  --train-generations "$TRAIN_GENERATIONS" \
  --no-dependency-correction \
  --output artifacts/gca-dirichlet-map.no-dep.v2.model.json

echo
echo "[6/13] MAP-only, dependency corrected (no additional LLM calls)"
"$UACAA" gca-map-resolve \
  --data "$DEV_DATA" \
  --generations "$DEV_GENERATIONS" \
  --map-model artifacts/gca-dirichlet-map.dep.v2.model.json \
  --model "$MODEL" \
  --mode map-only \
  --output artifacts/dev.gpt-6-luna.gca-map-only.dep.v2.jsonl

echo
echo "[7/13] MAP-only, no dependency correction (no additional LLM calls)"
"$UACAA" gca-map-resolve \
  --data "$DEV_DATA" \
  --generations "$DEV_GENERATIONS" \
  --map-model artifacts/gca-dirichlet-map.no-dep.v2.model.json \
  --model "$MODEL" \
  --mode map-only \
  --output artifacts/dev.gpt-6-luna.gca-map-only.no-dep.v2.jsonl

echo
echo "[8/13] MAP-guided adaptive adjudication"
run_with_retry "MAP-guided adjudication" \
  "$UACAA" gca-map-resolve \
    --data "$DEV_DATA" \
    --generations "$DEV_GENERATIONS" \
    --map-model artifacts/gca-dirichlet-map.dep.v2.model.json \
    --model "$MODEL" \
    --reasoning-effort "$REASONING_EFFORT" \
    --mode map-guided \
    --output artifacts/dev.gpt-6-luna.gca-map-guided.dep.v2.jsonl

echo
echo "[9/13] Always adjudicate every detected conflict"
run_with_retry "always-adjudicate ablation" \
  "$UACAA" gca-map-resolve \
    --data "$DEV_DATA" \
    --generations "$DEV_GENERATIONS" \
    --map-model artifacts/gca-dirichlet-map.dep.v2.model.json \
    --model "$MODEL" \
    --reasoning-effort "$REASONING_EFFORT" \
    --mode always-adjudicate \
    --output artifacts/dev.gpt-6-luna.gca-always-adjudicate.dep.v2.jsonl

echo
echo "[10/13] Score one-call and v1 adaptive outputs"
"$UACAA" gca-evaluate \
  --data "$DEV_DATA" \
  --predictions "$DEV_GENERATIONS" \
  --output artifacts/dev.gpt-6-luna.adaptive-gca.v1.scored.jsonl

echo
echo "[11/13] Score all MAP routing experiments"
for name in \
  gca-map-only.dep.v2 \
  gca-map-only.no-dep.v2 \
  gca-map-guided.dep.v2 \
  gca-always-adjudicate.dep.v2; do
  "$UACAA" gca-map-evaluate \
    --data "$DEV_DATA" \
    --predictions "artifacts/dev.gpt-6-luna.${name}.jsonl" \
    --output "artifacts/dev.gpt-6-luna.${name}.scored.jsonl"
done

echo
echo "[12/13] Re-score the frozen one-call prompting baselines"
"$UACAA" evaluate-answer-baseline \
  --data "$DEV_DATA" \
  --predictions artifacts/dev.gpt-6-luna.direct-rag.v2.answers.jsonl \
  --output artifacts/dev.gpt-6-luna.direct-rag.v2.scored.jsonl
"$UACAA" evaluate-answer-baseline \
  --data "$DEV_DATA" \
  --predictions artifacts/dev.gpt-6-luna.qacc-prompt.v2.answers.jsonl \
  --output artifacts/dev.gpt-6-luna.qacc-prompt.v2.scored.jsonl

echo
echo "[13/13] Merge comparison tables"
"$PYTHON" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path


ARTIFACTS = Path("artifacts")


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def answer_row(name: str, summary: dict, role: str) -> dict:
    return {
        "method": name,
        "role": role,
        "examples": summary.get("examples"),
        "answer_exact": summary.get("answer_exact"),
        "answer_token_f1": summary.get("answer_token_f1"),
        "coverage": summary.get("coverage"),
        "selective_exact": summary.get("selective_exact"),
        "conflict_rate": None,
        "helped": None,
        "hurt": None,
        "mean_llm_calls": 1.0,
        "mean_tokens": None,
    }


def gca_row(name: str, summary: dict, role: str) -> dict:
    final = summary["adaptive_final"]
    routing = summary.get("routing", {})
    trigger = summary.get("trigger", {})
    return {
        "method": name,
        "role": role,
        "examples": final.get("examples"),
        "answer_exact": final.get("answer_exact"),
        "answer_token_f1": final.get("answer_token_f1"),
        "coverage": final.get("coverage"),
        "selective_exact": final.get("selective_exact"),
        "conflict_rate": routing.get("conflict_rate", trigger.get("rate")),
        "helped": routing.get("helped", trigger.get("helped")),
        "hurt": routing.get("hurt", trigger.get("hurt")),
        "mean_llm_calls": trigger.get("mean_llm_calls_per_question"),
        "mean_tokens": summary.get("usage", {}).get("mean_total_tokens_per_question"),
    }


direct = load("dev.gpt-6-luna.direct-rag.v2.scored.summary.json")
qacc = load("dev.gpt-6-luna.qacc-prompt.v2.scored.summary.json")
v1 = load("dev.gpt-6-luna.adaptive-gca.v1.scored.summary.json")
map_dep = load("dev.gpt-6-luna.gca-map-only.dep.v2.scored.summary.json")
map_no_dep = load("dev.gpt-6-luna.gca-map-only.no-dep.v2.scored.summary.json")
map_guided = load("dev.gpt-6-luna.gca-map-guided.dep.v2.scored.summary.json")
always = load("dev.gpt-6-luna.gca-always-adjudicate.dep.v2.scored.summary.json")

rows = [
    answer_row("direct_rag", direct, "one_call_baseline"),
    answer_row("qacc_explain_then_answer", qacc, "one_call_baseline"),
    answer_row("joint_generation", v1["initial_generation"], "one_call_baseline"),
    gca_row("v1_llm_adaptive", v1, "llm_only_adjudication"),
    gca_row("v2_map_only_dependency", map_dep, "bayesian_ablation"),
    gca_row("v2_map_only_no_dependency", map_no_dep, "bayesian_ablation"),
    gca_row("v2_map_guided", map_guided, "selected_v2"),
    gca_row("v2_always_adjudicate", always, "cost_ablation"),
]

old_path = ARTIFACTS / "dev-results.gpt-6-luna.v3.json"
if old_path.exists():
    old = json.loads(old_path.read_text(encoding="utf-8"))["methods"]
    for method in ("raw_passage_majority", "selected_map_no_dependency"):
        metrics = old[method]["metrics"]
        rows.append(
            {
                "method": f"old_{method}",
                "role": old[method]["role"],
                "examples": metrics.get("examples"),
                "answer_exact": metrics.get("answer_exact"),
                "answer_token_f1": metrics.get("answer_token_f1"),
                "coverage": metrics.get("coverage"),
                "selective_exact": metrics.get("selective_answer_exact"),
                "conflict_rate": None,
                "helped": None,
                "hurt": None,
                "mean_llm_calls": None,
                "mean_tokens": None,
            }
        )

payload = {
    "scope": "QACC strict dev; post-hoc development only; official test excluded",
    "rows": rows,
    "fit_dependency": load("gca-dirichlet-map.dep.v2.model.json"),
    "fit_no_dependency": load("gca-dirichlet-map.no-dep.v2.model.json"),
}
json_path = ARTIFACTS / "overnight-gca-v2-results.json"
json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def fmt(value):
    return "—" if value is None else f"{float(value):.3f}"


table = [
    "# Overnight GCA v2 dev comparison",
    "",
    "Post-hoc QACC strict-dev development results. The official test is excluded.",
    "",
    "| Method | Role | EM | Token F1 | Coverage | Selective EM | Conflict/trigger rate | Helped | Hurt | Mean calls | Mean tokens |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    table.append(
        "| "
        + " | ".join(
            [
                row["method"],
                row["role"],
                fmt(row["answer_exact"]),
                fmt(row["answer_token_f1"]),
                fmt(row["coverage"]),
                fmt(row["selective_exact"]),
                fmt(row["conflict_rate"]),
                "—" if row["helped"] is None else str(row["helped"]),
                "—" if row["hurt"] is None else str(row["hurt"]),
                fmt(row["mean_llm_calls"]),
                fmt(row["mean_tokens"]),
            ]
        )
        + " |"
    )
table.extend(
    [
        "",
        "MAP thresholds are fixed pilot defaults (posterior >= 0.75 and margin >= 0.30), not calibrated correctness probabilities.",
        "",
    ]
)
markdown_path = ARTIFACTS / "overnight-gca-v2-results.md"
markdown_path.write_text("\n".join(table), encoding="utf-8")
print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "methods": len(rows)}, indent=2))
PY

echo
echo "[overnight] complete"
echo "[overnight] results: artifacts/overnight-gca-v2-results.md"
echo "[overnight] machine-readable: artifacts/overnight-gca-v2-results.json"
echo "[overnight] log: $LOG_FILE"
