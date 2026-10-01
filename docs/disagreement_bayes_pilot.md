# Bayesian disagreement arbitration pilot

## Status and scope

This is a labelled-development diagnostic, not a frozen test result. It reuses
the 397 cached Direct-RAG answers from GPT-6 Luna and GPT-4o mini and the cached
Luna context extractions. It makes no API calls. Every reported prediction is
out of fold: the label for a held-out question is not used to fit or calibrate
the fold model that predicts it.

The available train split does not yet have both Direct-RAG caches. A formal
train/dev/test experiment would require generating both model outputs on train,
fitting only on train, selecting all thresholds on dev, and then freezing test
predictions before scoring.

## Method

Answers are first merged only under strict normalized equality or a narrow
whitelist of harmless temporal wrappers such as `every October`. Numeric and
non-wrapper content must remain identical. Semantically agreeing answers bypass
the ranker and retain the shorter minimal span.

Disagreements form a three-way conditional choice set:

1. the Luna answer;
2. the GPT-4o mini answer;
3. `unknown`, representing none of the supplied candidates.

A Gaussian-prior conditional logit is fitted with a Laplace posterior. The 15
candidate features include model identity, an interaction with extracted
conflict strength, support and independent-domain counts, entailment,
directness, answer-type and temporal fit, coarse source categories, exact
context mentions, unsupported-token rate, and answer length. Individual domain
reliabilities are deliberately excluded because QACC has too little repeated
domain supervision.

Within each outer fold, a hash-stable subset of the remaining questions is held
out for temperature and split-conformal calibration. The MAP answer evaluates
the ranker itself. The conformal answer is emitted only when its prediction set
contains exactly one non-`unknown` candidate; all other sets abstain.

## Reproduction

```bash
./run_disagreement_bayes_pilot.sh
```

The output is `artifacts/dev.disagreement-bayes-conformal.v1.oof.jsonl` with a
matching `.summary.json`.

## Frozen pilot result (seed 42)

| Method | EM | token F1 | coverage | selective EM |
|---|---:|---:|---:|---:|
| Luna Direct-RAG | 0.6474 | 0.7677 | 0.9647 | 0.6710 |
| GPT-4o mini Direct-RAG | 0.6474 | 0.7686 | 0.9798 | 0.6607 |
| OOF Bayesian MAP | 0.6574 | 0.7745 | 0.9521 | 0.6905 |
| OOF conformal singleton | 0.6297 | 0.7192 | 0.8690 | 0.7246 |

There were 68 strict disagreements. Exact candidate recall was 0.5882 and
relaxed candidate recall was 0.6912. On the disagreements, MAP produced eight
helps and nine hurts relative to Luna, for a net loss of one. The overall MAP
gain therefore came from the
narrow minimal-span normalization on semantically agreeing answers, not from
Bayesian conflict arbitration.

Seed sensitivity also shows limited data stability. With seeds 1, 7, 21, 42,
and 99, MAP EM was 0.6448, 0.6524, 0.6474, 0.6574, and 0.6650. The method can
fall below Luna despite shrinkage. The conformal branch behaves as intended as
a risk/coverage trade-off, but it does not improve ordinary EM.

## Interpretation

This pilot does not validate Bayesian disagreement arbitration as a conflict
resolver. It does validate two narrower findings:

- conservative answer-span canonicalization can remove unsupported wrappers;
- conformal singleton filtering raises accuracy among answered questions at a
  substantial coverage cost.

The unresolved bottlenecks are candidate recall and the very small number of
informative disagreements. No posterior or MAP rule can recover a correct
answer that neither model generated.
