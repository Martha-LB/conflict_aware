# Formal UQ train-to-dev protocol

## Purpose

The earlier UQ results are out-of-fold development diagnostics. This protocol
moves model fitting to QACC train and reserves dev for calibration and method
selection. Official test inputs and references are not touched at this stage.

## Run

Activate the project environment and ensure `OPENAI_API_KEY` is already
exported in that shell. The key is read from the environment and is never
written to the script or an artifact.

```bash
./run_uq_train_dev.sh
```

The first run generates 313 Luna and 313 GPT-4o-mini train Direct-RAG answers.
Both caches are resumable. Later offline reruns can skip API construction:

```bash
SKIP_GENERATION=1 ./run_uq_train_dev.sh
```

The script then fits on train and evaluates two symmetric development routes
under the same secondary-call budget:

1. Luna primary, Mini secondary;
2. Mini primary, Luna secondary.

The default first-stage feature set is `answer-only`, selected because the
earlier dev-only ablation found no benefit from source/time features. The
post-secondary comparison model still receives the full evidence feature set.

Each direction writes development rows, a `.summary.json`, and a serialised
`.model.json`. These are calibration and selection artifacts, not test results.
After choosing one direction and freezing every option, a separate command
will apply the saved model to official test exactly once.
