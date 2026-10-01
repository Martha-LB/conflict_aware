# Adaptive cross-model candidate-locked verification (v4)

This post-hoc pipeline tests a qualitatively different hypothesis from MAP:
independent model diversity and explicit verification may resolve conflicts that
small supervised rankers cannot. It follows evidence on QACC that cross-model
verification can outperform single-model and same-model verification:
<https://aclanthology.org/2025.ranlp-1.116/>.

## Method

1. Reuse the frozen `gpt-6-luna` joint-generation answer as C1.
2. Use the train-fitted risk detector (or deterministic rules) only as a cost
   gate. It never chooses the final answer.
3. On triggered questions, call a different model without revealing C1. The
   secondary model emits a primary answer, alternatives, and verbatim evidence.
4. Merge equivalent answers into a locked candidate set.
5. Build evidence cards containing source domain/category, frozen optional
   authority metadata, independent-source groups, dates, question-time match,
   and extraction entailment/directness/type/temporal assessments.
6. A verifier selects an existing candidate ID, `KEEP_INITIAL`, or `UNKNOWN`.
   It cannot write an answer string.
7. Conservative mode switches only for high-confidence, sufficient decisions
   whose decisive context actually supports the selected candidate and whose
   explicit question-time constraint is not violated.

The secondary and verifier caches are separate and resumable. API keys are read
from environment variables and are never written to files.

## Model choice

Use genuinely different strong model families when available. A cheap model is
acceptable for the first 20-question engineering check, but it is not an equal
substitute for a strong diverse generator or verifier. Freeze exact model IDs.

The current Codex process cannot see a key exported in another Terminal window,
so run API commands in the same activated Terminal environment used for earlier
experiments.

## Twenty-question smoke test

Choose exact model IDs available to the account:

```bash
export SECONDARY_MODEL='<different-generator-model-id>'
export VERIFIER_MODEL='<different-verifier-model-id>'
```

The pilot script defaults to `SECONDARY_REASONING_EFFORT=omit`, which is
required by older models such as `gpt-4o-mini-2024-07-18`, and
`VERIFIER_REASONING_EFFORT=none` for `gpt-6-luna`. Override either environment
variable only when the selected endpoint supports the requested parameter.

Generate only on train-fitted detector risk cases:

```bash
uacaa cross-model-generate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --primary-generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --risk-model artifacts/two-stage-evidence-aware.v3.model.json \
  --risk-mode detector \
  --model "$SECONDARY_MODEL" \
  --reasoning-effort none \
  --limit 20 \
  --output artifacts/dev.cross-model.v4.secondary.jsonl
```

Run candidate-locked verification:

```bash
uacaa cross-model-verify \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --primary-generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --secondary-generations artifacts/dev.cross-model.v4.secondary.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --model "$VERIFIER_MODEL" \
  --reasoning-effort none \
  --mode conservative \
  --limit 20 \
  --output artifacts/dev.cross-model.v4.conservative.jsonl

uacaa cross-model-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.cross-model.v4.conservative.jsonl \
  --limit 20 \
  --output artifacts/dev.cross-model.v4.conservative.scored.jsonl
```

Inspect `helped`, `hurt`, `changed`, exact match, token F1, and the evidence
cards before scaling. The key safety check is that `final_answer` is always
either an exact locked candidate, the unchanged initial answer, or `unknown`.

## Resume the full dev run

Use the same output paths and exact model IDs, but omit `--limit`. Completed
question IDs are reused:

```bash
uacaa cross-model-generate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --primary-generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --risk-model artifacts/two-stage-evidence-aware.v3.model.json \
  --risk-mode detector \
  --model "$SECONDARY_MODEL" \
  --reasoning-effort none \
  --output artifacts/dev.cross-model.v4.secondary.jsonl

uacaa cross-model-verify \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --primary-generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --secondary-generations artifacts/dev.cross-model.v4.secondary.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --model "$VERIFIER_MODEL" \
  --reasoning-effort none \
  --mode conservative \
  --output artifacts/dev.cross-model.v4.conservative.jsonl

uacaa cross-model-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.cross-model.v4.conservative.jsonl \
  --output artifacts/dev.cross-model.v4.conservative.scored.jsonl
```

For the forced-selection ablation, write a separate cache with `--mode select`.
Never resume the conservative output path with a different mode; the manifest
will reject the provenance mismatch.

On the existing dev cache, detector-only routing triggers 117/397 questions.
If every triggered question requires both secondary generation and
verification, the expected total is approximately 1.59 LLM calls per question
including the already-completed primary generation. `union` routing would
trigger 163/397 and raise that upper bound to approximately 1.82 calls.
