# Answer-Risk–Guided Selective Revision for Conflicting-Context QA

This project studies one question:

> When retrieved contexts conflict and source reliability is unknown, can the correct
> answer be identified from the available input?

The experiments use QACC. The proposed method is **Answer-Risk–Guided Selective
Revision (ARSR)**. Earlier convex projection and MAP experiments are not part of the
final method.

## Experiment design

The final comparison contains four systems.

| System | Role |
|---|---|
| Luna Direct-RAG | Main baseline |
| ARSR | Proposed method |
| QACC explain-then-answer | QACC paper reference baseline |
| Candidate oracle | Non-deployable diagnostic upper bound |

All systems are evaluated with the public QACC rule: normalized Exact Match (EM), token
F1, and the maximum score over the official references.

### Data

| Split | Examples | Use |
|---|---:|---|
| Train | 313 | Fit the ARSR risk models |
| Dev | 397 | Calibrate risk and freeze decision thresholds |
| Official test | 813 | Final evaluation |

The official test set contains 207 QACC conflict questions and 606 non-conflict
questions. QACC's `has_conflict` field defines the two subsets.

Test inputs and references are stored separately. Test references are used only during
evaluation.

## Proposed method: ARSR

ARSR uses the following decision process:

1. Luna produces a Direct-RAG answer.
2. A regularized and calibrated logistic model estimates the probability that this
   answer is wrong.
3. If the predicted risk exceeds a threshold fixed on dev, ARSR requests a second
   Direct-RAG answer from GPT-4o Mini.
4. ARSR estimates the error risk of both answers using answer disagreement, context
   support, evidence quality, source, and time features.
5. It switches only when the answers differ and the predicted risk improvement is at
   least `0.1`. Otherwise it keeps the Luna answer.

The value `0.1` is a **decision margin**, not a statistical p-value.

The method therefore tests two different abilities:

- detecting that the current answer may be wrong;
- identifying a better answer from the same available evidence.

The code and frozen artifacts retain the earlier internal name `UQ routing`.

## Official test results

| System | Overall EM | Overall F1 | Conflict EM | Non-conflict EM |
|---|---:|---:|---:|---:|
| Luna Direct-RAG | 0.5855 | 0.7039 | 0.4879 | 0.6188 |
| **ARSR** | **0.5892** | **0.7061** | **0.4928** | **0.6221** |
| QACC explain-then-answer | 0.6212 | 0.7487 | 0.5217 | 0.6551 |
| Candidate oracle | 0.7023 | 0.8115 | 0.6087 | 0.7343 |

ARSR improves overall EM by `0.0037` and conflict EM by `0.0048` over Direct-RAG.
This improvement is not statistically significant:

- paired bootstrap 95% interval: `[-0.0074, 0.0148]`;
- two-sided McNemar test: `p = 0.6636`;
- 12 answer changes help and 9 hurt.

ARSR makes an average of `1.369` model calls per question.

### Risk prediction

The answer-risk model is more successful than the final revision policy:

| Metric | Result |
|---|---:|
| Error-detection AUROC | 0.7771 |
| Error-detection average precision | 0.7612 |
| Brier score | 0.1984 |
| Expected calibration error | 0.1180 |

The system can often detect that an answer is risky, but it cannot reliably determine
which alternative is correct. The candidate oracle is much higher than ARSR, which also
shows that candidate availability does not guarantee candidate identifiability.

## Main conclusion

The experiment supports **limited empirical identifiability** on QACC:

> Conflict and answer risk are partly observable, but the available contexts and metadata
> often do not provide enough evidence to select the correct conflicting answer.

This is an empirical result for QACC and the tested systems. It is not a proof that every
possible conflict-resolution method must fail.

The QACC explain-then-answer baseline is stronger than ARSR. It may use Luna's internal
knowledge as well as the supplied contexts. Because no no-context control was run, this
experiment cannot separate those two information sources.

## Reproduce the report

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Regenerate the final report from frozen predictions:

```bash
./run_identifiability_study.sh
```

This command makes no API calls. It writes:

```text
artifacts/test.official.identifiability.v1.json
artifacts/test.official.identifiability.v1.md
artifacts/test.official.identifiability.v1.rows.jsonl
```

The prepared QACC files are:

```text
data/derived/qacc-v1.2/train.strict.jsonl
data/derived/qacc-v1.2/dev.strict.jsonl
data/derived/qacc-v1.2/test.official.inputs.jsonl
data/derived/qacc-v1.2/test.official.references.jsonl
```

## Important files

```text
docs/data_protocol.md                  data fields and leakage rules
src/uacaa/answer_baselines.py          Direct-RAG and QACC prompts
src/uacaa/uncertainty_routing.py       ARSR risk models and decision rule
src/uacaa/identifiability.py           unified QACC evaluation
run_identifiability_study.sh           final frozen comparison
```

## Limitations

- QACC uses a broad definition of conflict, not only strict contradiction.
- The official test set was inspected during development, so it is not an untouched
  one-shot holdout.
- Some QACC answers and AmbigQA references differ. The reported results follow the
  official QACC reference evaluator.
- The no-context control was not run.
