# Uncertainty-aware routing pilot

## Scope

This is an offline labelled-development diagnostic. It reuses the 397 cached
Luna and GPT-4o-mini Direct-RAG answers plus the cached Luna context
extractions. It makes no API calls and is not a frozen test result.

Every risk score and routing decision is out of fold. In each outer fold, the
remaining questions are divided into a fitting subset and a disjoint
temperature/threshold calibration subset using stable hashes.

## Method

The primary model is frozen Luna Direct-RAG. A pre-routing regularized logistic
model estimates its error probability without using any Mini-derived feature.
Only questions in the calibrated high-risk tail trigger a simulated secondary
Mini call. After that trigger, two full-feature logistic models estimate the
relative error probabilities of the Luna and Mini answers. The 26 available
features are label-free and cover:

- evidence answer-cluster entropy, vote margin, and numeric conflict;
- Luna/Mini answer disagreement;
- answer support, entailment, directness, answer type, and temporal fit;
- independent domains, same-source/near-duplicate rate, and coarse source type;
- year dispersion and temporal cues;
- exact context mention and unsupported-answer-token rates.

The selective branch accepts Luna only below a fold-calibrated pre-routing risk
quantile. The intervention branch switches to Mini only if all three conditions
hold:

1. Luna and Mini disagree;
2. Luna is in the calibrated high-risk tail;
3. predicted Mini error is at least `delta_margin` below predicted Luna error.

The risk-only route drops the third condition and is a diagnostic showing
whether conservative expected-gain gating prevents harmful interventions.

## Reproduction

```bash
./run_uq_routing_pilot.sh
```

Primary outputs:

- `artifacts/dev.uq-routing.v1.oof.jsonl`
- `artifacts/dev.uq-routing.v1.oof.summary.json`

The summary reports AUROC, average precision, Brier score, ECE, AURC,
selective accuracy at fixed coverage levels, final EM/F1, paired McNemar
tests, and paired bootstrap intervals.

For feature ablations, set `PRE_FEATURE_MODE=no-source-time` or
`PRE_FEATURE_MODE=answer-only`. The former removes all domain/dependence and
temporal features from the pre-routing risk estimator; the latter retains only
question size and answer-surface support features. The post-trigger comparison
model remains unchanged so that the ablation isolates the first-stage router.

## Exploratory dev result

For seed 42 with the full feature set, the pre-routing error detector reached
AUROC 0.7124 and average precision 0.6829. At an approximately 80% coverage
point, selective exact accuracy was 0.7579, compared with 0.6474 when every
question was answered. The adaptive pipeline triggered the secondary model on
103/397 questions, corresponding to 1.259 simulated LLM calls per question.

The conservative route reached EM 0.6675 and token F1 0.7891, versus Luna
Direct-RAG EM 0.6474 and token F1 0.7677. It made 28 answer switches: 12 helped
strict EM and 4 hurt it. The paired bootstrap interval for EM improvement was
[0.0000, 0.0403], and the two-sided exact McNemar p-value was 0.0768. This is a
positive dev signal, not confirmatory evidence.

Across seeds 1, 7, 21, 42, and 99, full-feature risk AUROC was 0.7278 +/-
0.0167 and routed EM was 0.6741 +/- 0.0049. Mean simulated calls per question
were 1.247. Conflict-only routed EM averaged 0.5750, above Luna's 0.5000 but
below Mini's 0.6042; non-conflict routed EM averaged 0.6877 versus Luna's
0.6676. The pipeline therefore does not yet establish superior conflict
resolution.

The first-stage feature ablation is important. Mean results over the same five
seeds were:

| Pre-routing features | Error AUROC | Routed EM | Selective EM at 80% |
|---|---:|---:|---:|
| Full | 0.7278 | 0.6741 | 0.7547 |
| No source/time | 0.7357 | 0.6746 | 0.7566 |
| Answer-only | 0.7341 | 0.6796 | 0.7478 |

Source and temporal features did not improve this dev pilot. The overall EM
gain is driven mainly by routing unsupported, abstained, or badly scoped Luna
answers to Mini, rather than by resolving QACC's annotated conflict subset.
Because the same dev set has now informed method design, a formal claim needs a
fresh protocol: fit on train, choose all feature sets and thresholds on dev,
then evaluate once on the frozen official test.
