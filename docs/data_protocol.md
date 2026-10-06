# QACC data-use protocol

## Field roles

Inference is allowed to read only:

- `question`;
- `contexts`;
- `sources`.

Train/dev supervision may additionally read:

- `first/second/thirdAnswer`;
- `first/second/thirdContext`;
- `correctAnswer`;
- `reasons`;
- `explanation`.

Test-only evaluation may read `ambigqa_answer`, but only after a prediction has
been frozen. Prepared test inputs and references are stored in physically
separate files. Annotated candidates and context assignments are absent from
the prepared test inputs.

The released QACC train/dev rows omit the upstream `ambigqa_answer` aliases
that its public evaluator uses for test. Protocol v1.2 restores those aliases
from the official AmbigQA light `train_light.json` and `dev_light.json` files
by an exact normalized-question join. All 394 raw train and 410 raw dev rows
must match; the manifest records file hashes and split coverage. These aliases
are evaluation-only and are never placed in an LLM prompt.

Reported answer metrics reproduce QACC's released evaluator: lowercase,
remove ASCII punctuation and the articles `a/an/the`, normalize whitespace,
then compute exact string equality and word-overlap F1. For multiple references
the maximum score is used. QACC's annotator-selected `correctAnswer` remains
the supervision target and is also reported separately when needed; it is not
silently merged into the official AmbigQA reference set.

## Missingness

The official release contains non-standard JSON `NaN` tokens. They are parsed
as missing values. A missing second or third answer/context is structural when
the corresponding existence flag says that answer is absent. An unassigned
context is not a negative label: it remains unknown.

## Quality flags

- `truncated_fourth_answer`: a fourth answer is declared but absent from schema;
- `candidate_without_context`: an answer has an empty attribution list;
- `no_answer_candidate`: an annotation is `N/A` or equivalent;
- `correct_answer_unmapped`: selected answer does not map to a candidate;
- `duplicate_candidate_answer`: multiple ordinal slots normalize identically;
- `cross_candidate_context_overlap`: one context is assigned to multiple answers.

The strict train/dev cohort excludes the first five flags. Cross-candidate
overlap is retained and modeled as multi-label evidence because it may reflect
aliases or refinements rather than an error. The weak cohort keeps every row and
all flags for missing-data or positive-unlabelled experiments.

## Test duplicates

`test.official` preserves every released row. The deduplicated test file is a
predeclared sensitivity cohort and must not replace the official primary result.

## Provenance

Every prepared dataset records the raw absolute path, SHA-256 digest, upstream
revision, protocol version, exclusion rules, and generated filenames. Reported
experiments must archive `manifest.json`, model JSON, extraction caches, and the
exact command arguments.
