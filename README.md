# Answer-Risk–Guided Selective Revision for Conflicting-Context QA

## Background and research question

Retrieval-augmented generation (RAG) gives a language model several retrieved
contexts before it answers a question. These contexts may support different
answers. QACC provides source URLs and occasional time information, but only
limited metadata for judging source reliability.

This project builds on Liu et al. (2025), [*Open Domain Question Answering with
Conflicting Contexts*](https://aclanthology.org/2025.findings-naacl.99/). The paper
introduces the human-annotated QACC dataset and studies how language models answer
questions with conflicting web contexts. It also shows that asking a model to
explain before answering can improve performance.

The research question is:

> When retrieved contexts conflict and only limited source metadata is available,
> can the correct answer be identified from the available input?

The study uses the QACC author annotation as the conflict label. In the prepared
data, the original `secondAnswerExist` field is stored as `has_conflict`. The split
contains 313 train, 397 dev, and 813 official test examples. The test set has 207
conflict and 606 non-conflict examples. Test inputs and references are stored in
separate files to prevent answer leakage.

## Method

The main baseline is **Direct-RAG** with GPT-6 Luna. It receives the question and
all contexts and returns one short answer. The proposed method is
**Answer-Risk–Guided Selective Revision (ARSR)**:

1. Luna produces the primary Direct-RAG answer.
2. A regularized logistic model estimates the probability that this answer is
   wrong.
3. If the risk is above a threshold fixed on dev, GPT-4o Mini produces a second
   Direct-RAG answer.
4. ARSR compares the estimated error risk of both answers. It uses answer
   disagreement, context support, extracted evidence quality, source-domain, and
   time features.
5. ARSR switches only if the answers differ and the predicted improvement is at
   least `0.1`. Otherwise, it keeps the first answer.

The risk models are fitted on train. Calibration, thresholds, and the decision
margin are fixed on dev. The frozen method then predicts the official test set
without access to test references. References are loaded only in the final
evaluation step.

The comparison also includes the QACC **explain-then-answer** prompt and a
**candidate oracle**. The oracle uses the references to choose the best available
candidate, so it is only a diagnostic upper bound. All results use the public QACC
normalized Exact Match (EM) and token F1 rules, taking the best score over all
official references.

## Main results and takeaways

| System | Overall EM | Overall F1 | Conflict EM | Non-conflict EM |
|---|---:|---:|---:|---:|
| Luna Direct-RAG | 0.5855 | 0.7039 | 0.4879 | 0.6188 |
| ARSR | 0.5892 | 0.7061 | 0.4928 | 0.6221 |
| QACC explain-then-answer | 0.6212 | 0.7487 | 0.5217 | 0.6551 |
| Candidate oracle | 0.7023 | 0.8115 | 0.6087 | 0.7343 |

ARSR improves overall EM by `0.0037` and conflict EM by `0.0048` over Direct-RAG.
The gain is small and is not statistically significant: the paired bootstrap 95%
interval is `[-0.0074, 0.0148]`, and the two-sided McNemar test gives `p = 0.6636`.
ARSR changes 21 answers: 12 changes help and 9 hurt. It uses an average of `1.369`
model calls per question.

The risk model can detect difficult answers reasonably well (`AUROC = 0.7771`),
but the revision step rarely identifies a better answer. The large gap between
ARSR and the candidate oracle shows the main problem: a correct candidate may be
present, but the input may not show which candidate is correct. The stronger
explain-then-answer result may also use Luna's internal knowledge; this experiment
does not separate internal knowledge from context-based evidence.

The main conclusion is **limited empirical identifiability**: conflict and answer
risk are partly observable, but the available QACC contexts and metadata often do
not contain enough information for reliable conflict resolution. This conclusion
applies to QACC and the tested systems; it is not a proof that all possible methods
must fail.

## Running the code

Use Python 3.11 or newer. The shell script downloads the public data, prepares the
splits, runs all model calls, trains ARSR, evaluates the official test set, and
creates the final table. Existing caches are reused.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pytest
export PYTHONPATH="$PWD/src"
export OPENAI_API_KEY="your-key"

pytest -q
./run_study.sh
```

The full first run makes many API calls and may take several hours. On macOS, use
`caffeinate -i ./run_study.sh` to prevent sleep. Generated files
are stored in `data/derived/` and `artifacts/`; they are not committed to Git.
