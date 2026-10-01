# Train/dev unmapped-candidate error analysis

This audit uses only QACC train/dev supervision and the frozen
`gpt-6-luna.v3` extraction caches. Official test references are not read.
`correctAnswer` is treated as noisy weak supervision, not as an infallible
answer key.

## Reproducible outputs

For each split, `uacaa analyze-unmapped` writes:

- JSONL details with the question, weak gold answer, extracted candidates,
  QACC-attributed contexts, lexical diagnostics, and evidence text;
- a machine-readable JSON summary;
- a CSV ledger with empty `manual_resolution` and `manual_notes` columns.

The automated categories are triage labels. They do not alter supervision or
authorize a match by themselves.

## Results after conservative normalization fixes

| Split | Examples | Gold candidate mapped | No candidates | Candidate set present but gold unmapped |
|---|---:|---:|---:|---:|
| Train | 313 | 239 | 12 | 62 |
| Dev | 397 | 343 | 7 | 47 |

The initial mapping counts were 236/313 for train and 336/397 for dev. Manual
inspection supported only three general-purpose corrections:

1. minor orthographic variation with exact numeric agreement, covering plural
   suffixes, common month abbreviations, and a one-character name typo;
2. a minimal numeric candidate paired with a count-unit annotation, such as
   `10` versus `10 episodes` or `2` versus `two seats`;
3. removal of a trailing all-zero decimal in a quantity, such as
   `$160,000.00` versus `$160,000`.

These rules recovered 3 train and 7 dev examples. Every newly mapped example
was inspected. Genuine numeric conflicts remain non-matches.

## Remaining review inventory

| Review queue | Train | Dev | Interpretation |
|---|---:|---:|---|
| Extraction recall | 37 | 23 | No candidate, or no answer extracted from the QACC-attributed context |
| Alias or answer granularity | 6 | 17 | Potential semantic alias, incomplete name/date, or answer-boundary difference |
| Label or extractor disagreement | 24 | 13 | The attributed context was answered but its candidate is lexically disjoint from `correctAnswer` |
| Numeric or temporal conflict | 7 | 1 | Competing dates, quantities, unit conversions, or malformed numeric annotations |

The queues are not ground-truth error types. Examples show that the same queue
can contain an extractor miss, a time-dependent answer, an annotation with the
wrong answer type, or an incorrect `correctAnswer`. Consequently, the pipeline
does not use per-example overrides and does not broaden matching to force these
rows into training.

## Frozen decision

- Train/dev rows with no mapped gold candidate are excluded from conditional
  choice fitting because their target class is absent.
- They remain visible in candidate-recall accounting.
- No manual alias table is used.
- Test candidate failures remain in the official evaluation denominator as
  incorrect abstentions.
- Exact match remains normalized string equality; tolerant matching is used
  only for candidate-set recall/mapping, never to inflate answer EM.

