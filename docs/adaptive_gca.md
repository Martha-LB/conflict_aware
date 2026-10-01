# Adaptive Generate–Challenge–Adjudicate

## Status and evaluation boundary

This is a post-hoc alternative motivated by the completed candidate-aggregation
experiment. It was designed after the QACC official-test references had been
opened. All design, prompt, and trigger decisions must therefore be made on
train/dev. The existing official test can be used only for explicitly labelled
exploratory analysis; a new held-out dataset or split is required for a fresh
confirmatory result.

Version 1.0 is a feasibility experiment. It measures whether adjudication
improves the first answer often enough to justify a later confidence-calibration
stage. It does not yet claim a calibrated probability of correctness.

## Pipeline

1. A joint generator reads the question and every context in one request.
2. It returns a minimal primary answer, up to three supported alternatives,
   explicit contradictions, and contiguous evidence quotations.
3. Quotations are deterministically aligned back to the original contexts.
4. A second request is triggered by any of the following auditable conditions:
   no primary answer, insufficient evidence, an unsupported primary answer, an
   explicit contradiction, or a distinct alternative with valid evidence.
5. Candidate order is deterministically randomized. The adjudicator is not
   told which answer was initially preferred and does not receive source names.
6. The adjudicator sees all context text, attempts to falsify each hypothesis,
   and may select a listed candidate, return a different answer, or abstain.
7. Evaluation reports the initial and final answer side by side, including how
   often adjudication helped or hurt, citation validity, trigger rate, calls per
   question, and token usage.

This method does not use `g2`, `g4`, `g6`, document credibility, convex
projection, candidate-level logistic regression, or source-category priors.

## Smoke test on 20 dev questions

Use a new cache name. Do not reuse the candidate-extraction cache.

```bash
uacaa gca-run \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --model gpt-6-luna \
  --reasoning-effort none \
  --mode adaptive \
  --limit 20 \
  --output artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl

uacaa gca-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --limit 20 \
  --output artifacts/dev.gpt-6-luna.adaptive-gca.v1.scored.jsonl
```

Inspect the resulting summary before paying for the full run. Continue only if
quotation validity is high and the second call does not systematically replace
correct initial answers with wrong answers.

## Resume the full dev run

The cache and manifest are resumable. Running the same command without
`--limit` reuses the first 20 examples.

```bash
uacaa gca-run \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --model gpt-6-luna \
  --reasoning-effort none \
  --mode adaptive \
  --output artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl

uacaa gca-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --output artifacts/dev.gpt-6-luna.adaptive-gca.v1.scored.jsonl
```

The decisive feasibility quantities are:

- `initial_generation.answer_exact`: the one-call joint generator;
- `adaptive_final.answer_exact`: the final adaptive method;
- `trigger.helped` and `trigger.hurt`: paired answer changes;
- `trigger.mean_llm_calls_per_question`: inference cost;
- `citation_validation.valid_rate`: grounding quality.

Only if the full-dev adaptive result has a positive paired benefit should a
calibration model and fixed abstention policy be added. That calibration must
be fitted on a separate calibration partition or by out-of-fold prediction,
not on the same rows used to report its performance.
