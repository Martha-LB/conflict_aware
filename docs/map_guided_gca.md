# Bayesian MAP–Guided GCA v2

## Decision pipeline

The first joint-generation call returns a primary answer, supported alternatives,
verbatim evidence, explicit contradictions, and evidence sufficiency. Validated
citations induce a candidate–context relation matrix:

- a cited context is `support` for its candidate;
- the same context is `contradict` for a competing, non-equivalent candidate;
- an uncited context is `irrelevant`.

Inference then has three distinct routes:

1. one supported answer and sufficient evidence: return the initial answer;
2. no supported conflict but insufficient evidence: abstain;
3. two or more supported, non-equivalent answers: run the Bayesian relation
   model and use MAP-guided conflict resolution.

No QACC `has_conflict` or gold-answer field enters inference.

## Bayesian relation model

For relation label (r\in\{support, contradict, irrelevant\}), the model learns
separate posterior-predictive distributions for correct and incorrect
candidates. Both receive symmetric Dirichlet priors. For candidate (a), each
independent evidence group contributes:

\[
\log BF_{ga}=\log P(R_{ga}\mid a\text{ correct},D)
-\log P(R_{ga}\mid a\text{ incorrect},D).
\]

The candidate posterior is the softmax of the summed log Bayes factors plus a
uniform candidate prior. Copied contexts and contexts from the same domain are
one evidence group by default, so repeated snippets cannot multiply posterior
mass without limit.

The fixed v2 pilot thresholds accept MAP locally only when its posterior is at
least 0.75 and its margin over the second candidate is at least 0.30. These
thresholds are predeclared defaults, not calibrated probabilities. If either
condition fails, a second LLM receives the MAP ranking, relation-specific log
Bayes factors, decisive quotations, counterevidence, and the original contexts
without source labels. It is explicitly allowed to overturn MAP or abstain.

## Pilot protocol

First build a one-call joint-generation cache for a train subset. `--mode never`
prevents any v1 adjudication call. Start with 100 questions to test whether the
relation likelihoods are identifiable before paying for the full train split.

```bash
uacaa gca-run \
  --data data/derived/qacc-v1/train.strict.jsonl \
  --model gpt-6-luna \
  --reasoning-effort none \
  --mode never \
  --limit 100 \
  --output artifacts/train.gpt-6-luna.joint-generation.v1.jsonl
```

Fit the Dirichlet relation model. Only labelled multi-candidate generations are
used; single-candidate questions do not identify the incorrect-candidate
relation distribution.

```bash
uacaa gca-map-fit \
  --train data/derived/qacc-v1/train.strict.jsonl \
  --train-generations artifacts/train.gpt-6-luna.joint-generation.v1.jsonl \
  --output artifacts/gca-dirichlet-map.v2.model.json
```

Before paying for additional adjudication calls, evaluate MAP-only on the same
20 dev generations already produced by the v1 smoke test:

```bash
uacaa gca-map-resolve \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --map-model artifacts/gca-dirichlet-map.v2.model.json \
  --model gpt-6-luna \
  --mode map-only \
  --limit 20 \
  --output artifacts/dev.gpt-6-luna.gca-map-only.v2.jsonl

uacaa gca-map-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.gpt-6-luna.gca-map-only.v2.jsonl \
  --limit 20 \
  --output artifacts/dev.gpt-6-luna.gca-map-only.v2.scored.jsonl
```

Only if MAP-only is at least neutral should the MAP-guided resolver make new
LLM calls:

```bash
uacaa gca-map-resolve \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --generations artifacts/dev.gpt-6-luna.adaptive-gca.v1.jsonl \
  --map-model artifacts/gca-dirichlet-map.v2.model.json \
  --model gpt-6-luna \
  --reasoning-effort none \
  --mode map-guided \
  --limit 20 \
  --output artifacts/dev.gpt-6-luna.gca-map-guided.v2.jsonl

uacaa gca-map-evaluate \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.gpt-6-luna.gca-map-guided.v2.jsonl \
  --limit 20 \
  --output artifacts/dev.gpt-6-luna.gca-map-guided.v2.scored.jsonl
```

Required ablations are the initial one-call answer, v1 LLM-only adjudication,
MAP-only, MAP-guided adjudication, and always-adjudicate. Report route counts,
helped/hurt pairs, conflict-only initial/final EM, average calls, and token usage.

The official QACC test was opened before v2 was designed. It is not a fresh
confirmatory test for this method.
