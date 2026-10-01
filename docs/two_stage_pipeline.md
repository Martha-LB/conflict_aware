# Two-stage conflict detection and evidence-aware ranking

This post-hoc development pipeline separates two tasks that earlier GCA
variants conflated:

1. predict the QACC conflict label;
2. rank candidate answers only when the detector predicts conflict.

It reuses the frozen joint-generation and per-context extraction caches. No
additional LLM request is made.

## Leakage protocol

- Conflict labels come from the number of distinct QACC answer annotations and
  are read from `supervision` on train/dev only.
- The fitted detector is the only source of the conflict decision at inference.
- Gold answers are used only to fit the train candidate ranker and to evaluate
  dev after predictions have been written.
- A domain metadata file, when supplied, is serialized into the model. Live web
  lookups are deliberately excluded because they would make the run unstable.
- Conflict and answer metrics are reported separately.

## Features

The conflict detector combines joint-generation disagreement with independent
per-context extraction disagreement. Its features cover candidate counts,
support entropy, answer distance, numeric/date disagreement, temporal-fit
dispersion, source-category diversity, and frozen authority dispersion.

The candidate ranker uses support and independent-domain counts, extraction
directness/entailment/type/temporal scores, deterministic question–evidence
time alignment, temporal conflict, source-category shares, domain diversity,
and optional frozen external authority metadata.

The optional metadata JSON is keyed by normalized domain:

```json
{
  "example.gov": {
    "category": "official",
    "authority": 0.9,
    "independence_group": "example-agency"
  }
}
```

`authority` must be in `[0, 1]`. This is a task-conditional feature, not a claim
that a domain has universal credibility. Missing domains receive neutral
authority. Do not construct this file using dev/test answers.

## Commands

Fit on train only. The detector threshold and conservative answer-switch
thresholds are selected with five-fold out-of-fold train predictions:

```bash
uacaa two-stage-fit \
  --train data/derived/qacc-v1/train.strict.jsonl \
  --train-generations artifacts/train.gpt-6-luna.joint-generation.v1.jsonl \
  --train-extractions artifacts/train.gpt-6-luna.v3.extractions.jsonl \
  --output artifacts/two-stage-evidence-aware.v3.model.json
```

Freeze dev predictions, then evaluate them:

```bash
uacaa two-stage-resolve \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --extractions artifacts/dev.gpt-6-luna.v3.extractions.jsonl \
  --two-stage-model artifacts/two-stage-evidence-aware.v3.model.json \
  --mode conservative \
  --output artifacts/dev.gpt-6-luna.two-stage-conservative.v3.jsonl

uacaa two-stage-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.gpt-6-luna.two-stage-conservative.v3.jsonl \
  --output artifacts/dev.gpt-6-luna.two-stage-conservative.v3.scored.jsonl
```

Use `--mode map` as the forced-ranking ablation. It always accepts the top
ranked candidate on a predicted-conflict question. The default conservative
mode retains the initial answer unless train-only OOF thresholds accept a
switch.

## First frozen dev result

With no external domain metadata supplied:

| Method | Conflict P | Conflict R | Conflict F1 | Final EM | Final token F1 |
|---|---:|---:|---:|---:|---:|
| Initial joint generation | — | — | — | 0.6398 | 0.7642 |
| Two-stage conservative | 0.2222 | 0.5417 | 0.3152 | 0.6348 | 0.7618 |
| Two-stage forced MAP | 0.2222 | 0.5417 | 0.3152 | 0.6322 | 0.7606 |

The conservative pipeline changed 11 answers, helped 0, and hurt 2. The forced
MAP variant changed 13 answers, helped 0, and hurt 3. Therefore this feature
set does not improve dev performance. The detector generalizes poorly from 53
positive train labels to 48 positive dev labels, and the ranker has only 100
usable multi-candidate training questions. These results are development
evidence against selecting this version for a new official-test run.

