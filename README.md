# Uncertainty-Aware Candidate Answer Aggregation for QACC

This repository implements a leakage-safe, reproducible pipeline for factual QA
with conflicting web contexts. It replaces document-level convex credibility
projection with a posterior over candidate answers.

The raw QACC file is never edited. The pipeline records its SHA-256 digest,
creates explicit cohorts, extracts one candidate answer per context, and fits a
regularized conditional-choice model with calibrated abstention. A Laplace
Bayesian posterior and source-dependence correction remain explicit ablations;
neither was selected after train/dev comparison. There are no hard moment
constraints and therefore no empty feasible set.

## Repository layout

```text
configs/qacc.json       frozen default hyperparameters
docs/data_protocol.md   field roles, missingness, cohorts, leakage rules
docs/method.md          mathematical model and feature definitions
docs/unmapped_error_analysis.md  frozen train/dev mapping audit
src/uacaa/qacc.py       official-file parser and audit
src/uacaa/protocol.py   immutable raw/strict/weak protocol preparation
src/uacaa/extraction.py one batched LLM extraction call per question
src/uacaa/clustering.py answer alias clustering
src/uacaa/dependency.py source/copy dependence correction
src/uacaa/features.py   candidate-level evidence features
src/uacaa/bayes.py      Gaussian-prior conditional logit + Laplace posterior
src/uacaa/evaluation.py EM/F1, candidate recall, Brier, coverage
src/uacaa/adaptive_gca.py adaptive joint generation and blind adjudication
src/uacaa/map_gca.py    Dirichlet relation posterior and MAP-guided routing
src/uacaa/two_stage.py  supervised conflict detector + evidence-aware ranker
src/uacaa/cross_model.py adaptive diverse generation + locked verification
src/uacaa/disagreement.py OOF Bayesian disagreement ranking + conformal abstention
src/uacaa/uncertainty_routing.py OOF error-risk estimation + conservative routing
src/uacaa/cli.py        reproducible command-line pipeline
tests/                  unit and integration tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Reproducible pipeline

The offline uncertainty-aware routing pilot is documented in
`docs/uncertainty_routing_pilot.md` and can be run with
`./run_uq_routing_pilot.sh`. It reuses cached development predictions and does
not call an LLM API.

The subsequent train-fit/dev-calibration protocol, including symmetric
Luna-first and Mini-first comparison, is documented in
`docs/formal_uq_train_dev_protocol.md` and run by `./run_uq_train_dev.sh`.

After freezing the selected Luna-first configuration, the leakage-separated
official prediction and evaluation protocol is documented in
`docs/official_uq_test_protocol.md` and run by `./run_uq_official_test.sh`.

Set the path to the untouched official release:

```bash
QACC_RAW='../conflict_projection/data/qa-with-conflicting-context/data/ConflictQA_Dataset.json'
```

### 1. Audit and prepare cohorts

Use an upstream Git commit SHA for a reported experiment:

```bash
uacaa audit --raw "$QACC_RAW" --output artifacts/qacc-audit.json

uacaa prepare \
  --raw "$QACC_RAW" \
  --revision '<official-git-commit-sha>' \
  --output-dir data/derived/qacc-v1
```

Outputs include:

- `manifest.json`: raw digest, revision, protocol version, and leakage policy;
- `audit.json`: field coverage, anomalies, duplicates, and split counts;
- `train/dev.weak.jsonl`: all rows with explicit quality flags;
- `train/dev.strict.jsonl`: deterministic exclusions for supervised fitting;
- `test.official.inputs.jsonl` and `.references.jsonl`: physically separated
  official evaluation inputs and gold references;
- `test.deduplicated_sensitivity.inputs.jsonl` and `.references.jsonl`:
  secondary sensitivity analysis only.

### 2. Extract candidates

One request contains the question and all of its contexts. The V3 extractor
must emit exactly one explicit assessment for every context, including an
`answerable=false` record for abstentions. It must judge each context
independently and must not choose the global answer. The cache retains all
context assessments, positive candidate observations, response usage, model
identity, and system fingerprint. Candidate evidence is accepted verbatim or
deterministically aligned when its complete token sequence differs only in
case, whitespace, or punctuation. Other mismatches are retried, then isolated
and recorded per context rather than terminating the full run.

```bash
read -s "OPENAI_API_KEY?OpenAI API key: "
export OPENAI_API_KEY

# First validate the prompt and schema on 20 examples.
uacaa extract \
  --data data/derived/qacc-v1/train.strict.jsonl \
  --model '<frozen-model-id>' \
  --limit 20 \
  --output artifacts/train.extractions.jsonl

# Then resume the same cache; the first 20 requests are not repeated.
uacaa extract \
  --data data/derived/qacc-v1/train.strict.jsonl \
  --model '<frozen-model-id>' \
  --output artifacts/train.extractions.jsonl

uacaa extract \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --model '<same-frozen-model-id>' \
  --output artifacts/dev.extractions.jsonl
```

Extraction uses a strict JSON schema by default, writes a model/input/prompt
manifest beside each cache, and resumes completed question IDs after an
interruption. Use `--json-mode` only for an OpenAI-compatible endpoint that does
not implement Structured Outputs, or `--no-resume` to deliberately replace a
cache.

Run the same frozen extractor on `test.official.inputs.jsonl` only after all choices
have been fixed. API keys are read from an environment variable and are never
stored in the repository.

### 3. Fit, calibrate, and freeze the posterior

Before fitting, generate a deterministic error-analysis ledger for gold answers
that are absent from the extracted candidate set. This command is restricted to
train/dev files and writes JSONL details, a JSON summary, and a CSV with empty
manual-review columns. Automated categories are triage labels, not permission
to change the matcher without manual review.

```bash
uacaa analyze-unmapped \
  --data data/derived/qacc-v1/train.strict.jsonl \
  --extractions artifacts/train.extractions.jsonl \
  --output artifacts/train.unmapped.jsonl
```

```bash
uacaa train \
  --train data/derived/qacc-v1/train.strict.jsonl \
  --dev data/derived/qacc-v1/dev.strict.jsonl \
  --train-extractions artifacts/train.extractions.jsonl \
  --dev-extractions artifacts/dev.extractions.jsonl \
  --output artifacts/uacaa-model.json
```

Omitting the extraction caches intentionally uses QACC train/dev attribution
as an oracle diagnostic upper bound. The CLI prints a warning; such a run is
not a deployable result.

### 4. Freeze baselines and ablations on dev

The following command reuses the same extraction caches for raw-passage
majority, independent-source majority, point-estimate conditional logit, the
full model, and the predeclared feature/calibration ablations. Its metrics are
development diagnostics tuned on dev, not final test estimates. Direct-RAG and
QACC explain-then-answer require separate LLM caches and are reported as
pending until those calls have been completed.

```bash
uacaa dev-experiments \
  --train data/derived/qacc-v1/train.strict.jsonl \
  --dev data/derived/qacc-v1/dev.strict.jsonl \
  --train-extractions artifacts/train.extractions.jsonl \
  --dev-extractions artifacts/dev.extractions.jsonl \
  --output artifacts/dev-experiments.json
```

The two prompting baselines each use one additional request per question. Run
them on dev before freezing the test protocol:

```bash
uacaa answer-baseline \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --model '<same-frozen-model-id>' \
  --style direct-rag \
  --output artifacts/dev.direct-rag.answers.jsonl

uacaa answer-baseline \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --model '<same-frozen-model-id>' \
  --style qacc-explain-then-answer \
  --output artifacts/dev.qacc-prompt.answers.jsonl

uacaa evaluate-answer-baseline \
  --data data/derived/qacc-v1/dev.strict.jsonl \
  --predictions artifacts/dev.direct-rag.answers.jsonl \
  --output artifacts/dev.direct-rag.scored.jsonl
```

### 5. Freeze test predictions, then score once

The prediction and scoring stages are separate commands. Run every selected
method and baseline first; verify that each prediction file is complete; only
then expose the evaluator to the references file. Do not use the legacy
combined `evaluate` command for a reported experiment.

```bash
uacaa extract \
  --data data/derived/qacc-v1/test.official.inputs.jsonl \
  --model '<same-frozen-model-id>' \
  --output artifacts/test.extractions.jsonl

uacaa predict \
  --data data/derived/qacc-v1/test.official.inputs.jsonl \
  --model artifacts/uacaa-model.json \
  --extractions artifacts/test.extractions.jsonl \
  --method-name selected_map_no_dependency \
  --output artifacts/test.predictions.jsonl

# Freeze the predeclared support baseline before scoring as well.
uacaa predict-baseline \
  --data data/derived/qacc-v1/test.official.inputs.jsonl \
  --extractions artifacts/test.extractions.jsonl \
  --method raw-passage-majority \
  --output artifacts/test.raw-majority.predictions.jsonl

# This is the first command in the candidate pipeline that may read references.
uacaa score \
  --predictions artifacts/test.predictions.jsonl \
  --references data/derived/qacc-v1/test.official.references.jsonl \
  --output artifacts/test.scored.jsonl
```

No test candidate answer, context attribution, `correctAnswer`, reason,
explanation, or `ambigqa_answer` is available to extraction or scoring. Test
references and the conflict-subgroup flag enter only inside the evaluator after
prediction. Candidate-extraction failures remain in the evaluation denominator
as incorrect abstentions rather than being silently dropped.

The completed official experiment and its frozen prediction-file hashes are
reported in `docs/official_test_results.md`.

## Post-hoc alternative: Adaptive Generate–Challenge–Adjudicate

The candidate ranker did not outperform the prompting baselines. The repository
therefore includes a separate post-hoc research direction that preserves the
full contexts: one joint generation call, followed by a blind second call only
when deterministic evidence checks detect conflict or insufficiency. Because
the official test was already opened before this method was designed, develop
and compare it on train/dev only. Do not present a later run on the same test as
a fresh confirmatory result. See `docs/adaptive_gca.md` for the protocol.

The Bayesian MAP-guided v2 is documented in `docs/map_guided_gca.md`. It uses
the validated joint-generation cache, fits relation likelihoods on labelled
train questions, and invokes MAP or a MAP-guided adjudicator only for detected
conflicts. Evidence insufficiency is a separate abstention branch.

The later two-stage diagnostic is documented in `docs/two_stage_pipeline.md`.
It trains conflict detection directly against QACC train labels, adds explicit
domain/time features to a separate candidate ranker, and reports conflict F1
independently from final answer EM/F1. It is fully offline when the generation
and extraction caches already exist.

The v4 cross-model experiment is documented in `docs/cross_model_v4.md`. It
uses the train-fitted detector only as a cost gate, calls an independent model
on triggered questions, and lets a separate verifier select only from locked
candidate IDs using source/time evidence cards. It never permits free-form
answer rewriting.

The offline Bayesian disagreement pilot is documented in
`docs/disagreement_bayes_pilot.md`. It cross-fits a low-dimensional conditional
choice model over cached Luna/Mini Direct-RAG disagreements and adds a
split-conformal singleton rule. It is explicitly a labelled-dev diagnostic;
its small MAP gain is attributable to answer-span canonicalization rather than
successful conflict arbitration.

## Required comparisons

Report the same frozen questions and extractor for:

1. direct RAG;
2. raw passage majority;
3. independent-source majority;
4. non-Bayesian conditional logistic ranking;
5. this Laplace-posterior model;
6. QACC explain-and-answer prompting.

Report overall and conflict-only answer EM/F1, candidate recall, multiclass
Brier score, coverage/selective accuracy, and the official-versus-deduplicated
test sensitivity result. Never select a cleaning rule or confidence threshold
using test performance.

## Important interpretation

The output is a posterior over the candidates extracted for one question. It
is not an intrinsic probability that a whole document or domain is credible.
If the correct answer was not extracted, the ranker cannot recover it; candidate
recall is therefore a first-class metric.
