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
