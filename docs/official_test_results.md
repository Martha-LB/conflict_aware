# Frozen QACC official-test results

## Protocol

All seven methods produced predictions for the same 813 official test
questions before `test.official.references.jsonl` was loaded by either scoring
command. The five candidate-method prediction files contain no reference,
correctness, or score fields. Each has 813 unique question IDs. The two
free-form prompting caches likewise contain 813 unique predictions and have
complete provenance manifests.

The selected candidate model was fixed from train/dev as MAP conditional logit
without dependence correction. The Laplace-posterior model and dependence
correction remain ablations even if an isolated test metric is numerically
higher; test results are not used to change method selection.

## Overall results

| Method | Role | EM | Token F1 | Candidate recall | Brier ↓ | Coverage | Selective EM |
|---|---|---:|---:|---:|---:|---:|---:|
| QACC explain-then-answer | prompting baseline | **0.625** | **0.756** | — | — | 0.986 | 0.633 |
| Direct RAG | prompting baseline | 0.588 | 0.714 | — | — | 0.946 | 0.622 |
| Raw-passage majority | candidate baseline | 0.515 | 0.648 | 0.705 | 0.197 | 0.950 | 0.543 |
| Independent-source majority | candidate baseline | 0.514 | 0.646 | 0.705 | 0.195 | 0.950 | 0.541 |
| Laplace posterior + dependence correction | Bayesian ablation | 0.493 | 0.579 | 0.705 | 0.188 | 0.752 | 0.656 |
| MAP, no dependence correction | **selected candidate method** | 0.492 | 0.578 | 0.705 | 0.188 | 0.747 | **0.659** |
| MAP + dependence correction | non-Bayesian ablation | 0.492 | 0.577 | 0.705 | **0.187** | 0.749 | 0.657 |

## Conflict-only results

The conflict subset contains 207 questions.

| Method | EM | Token F1 | Candidate recall | Brier ↓ | Coverage | Selective EM |
|---|---:|---:|---:|---:|---:|---:|
| QACC explain-then-answer | **0.517** | **0.676** | — | — | 0.990 | 0.522 |
| Direct RAG | 0.483 | 0.640 | — | — | 0.961 | 0.503 |
| Raw-passage majority | 0.411 | 0.563 | 0.628 | 0.252 | 0.971 | 0.423 |
| Independent-source majority | 0.411 | 0.557 | 0.628 | 0.253 | 0.971 | 0.423 |
| Laplace posterior + dependence correction | 0.362 | 0.440 | 0.628 | 0.249 | 0.643 | 0.564 |
| MAP, no dependence correction | 0.362 | 0.446 | 0.628 | 0.247 | 0.643 | 0.564 |
| MAP + dependence correction | 0.362 | 0.439 | 0.628 | **0.247** | 0.638 | **0.568** |

## Interpretation

The main empirical result is negative for learned candidate aggregation. The
selected MAP method does not beat raw-passage majority in end-to-end EM or F1.
It achieves higher exact accuracy conditional on answering, but only by
reducing coverage from 0.950 to 0.747. Selective EM must therefore not be read
as an overall improvement.

Candidate recall is only 0.705 overall and 0.628 on conflicting questions, so
the extraction-and-clustering stage excludes a valid reference for roughly 30%
of all questions and 37% of conflicting questions. A ranker constrained to the
candidate set cannot repair those cases. The nearly identical Bayesian, MAP,
and dependence-ablation results further indicate that posterior averaging and
the current copy/source correction are not the limiting factors.

The strongest tested approach is the one-call QACC explain-then-answer prompt.
Its advantage over Direct RAG is 3.69 EM points and 4.19 token-F1 points
overall. This supports a narrower conclusion: with the tested model, allowing
the LLM to synthesize conflicting evidence jointly is more effective than the
current extract-then-rank feature pipeline.

## Frozen candidate-prediction hashes

| Prediction file | SHA-256 |
|---|---|
| `test.official.selected-map-no-dependency.predictions.jsonl` | `53663b45deaa1b92ecc9ebba51c5fc629f6e180a486397ff849f3bdb39c19488` |
| `test.official.bayesian-dependency-ablation.predictions.jsonl` | `8c8d47c818b523c83d7361cc439b75908aa0e14cfb36286cc84b3a3e61a32a58` |
| `test.official.map-dependency-baseline.predictions.jsonl` | `998201fb9f1623789a11717e5d0fcdd90cbd36ee39306fda07a292f969c18756` |
| `test.official.raw-passage-majority.predictions.jsonl` | `5306f9a5e30ea9b8866377ac8a1fa2a81f39f605cfc677091613b1bf0120ca45` |
| `test.official.independent-source-majority.predictions.jsonl` | `7feac4677749fefe5df6e6bd978dc6246f0dbc07ec02223f1fa1ccd8834bb5a7` |
