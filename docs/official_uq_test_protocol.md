# Frozen official UQ test protocol

## Frozen selection

The method was selected before official-test UQ scoring:

- primary: `gpt-6-luna` Direct-RAG;
- secondary: `gpt-4o-mini-2024-07-18` Direct-RAG;
- pre-routing features: `answer-only`;
- intervention quantile: `0.75`;
- expected-gain margin: `0.10`;
- prior scale: `0.75`;
- model: `artifacts/dev.uq-trainfit.luna-first.v1.model.json`.

No option may be changed in response to official-test results.

## Leakage barrier

`uacaa uq-routing-predict` accepts test inputs, cached answers, extractions, and
the frozen model. It has no references argument and writes the prediction file
with exclusive creation. A SHA-256 manifest is written immediately.

Only after that command completes does `uacaa uq-routing-evaluate` load the
physically separate reference file. Scored rows and their summary are written
to new files and cannot overwrite an earlier official result.

## Run

The Luna test answers and extractions already exist. The first run generates
the missing 813 Mini Direct-RAG answers, freezes UQ predictions, then scores:

```bash
./run_uq_official_test.sh
```

The Mini answer cache is resumable. If interrupted, rerun the same command. If
generation has completed, offline execution is available with:

```bash
SKIP_GENERATION=1 ./run_uq_official_test.sh
```

Primary outputs:

- `artifacts/test.official.uq-luna-first.v1.predictions.jsonl`
- `artifacts/test.official.uq-luna-first.v1.predictions.jsonl.manifest.json`
- `artifacts/test.official.uq-luna-first.v1.scored.jsonl`
- `artifacts/test.official.uq-luna-first.v1.scored.summary.json`
