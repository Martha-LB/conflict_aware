# Method

## Prediction target

For question \(q\), batched extraction produces observations from individual
contexts. Alias-aware clustering yields candidates
\(A_q=\{a_{q1},\ldots,a_{qK}\}\). The target is the correct candidate \(Z_q\),
not document credibility.

One LLM request contains all contexts for a question, but the V3 output schema
requires exactly one assessment for every context index. Each assessment has an
explicit `answerable` decision. Answerable contexts return a minimal candidate
span and a contiguous evidence quote; abstentions remain archived with empty
answer/evidence fields. The parser rejects missing or duplicate indices. Quotes
are either verified verbatim or deterministically aligned to an exact context
span when the complete token sequence differs only in punctuation, case, or
whitespace. Other mismatches are retried and then quarantined per context with
an audit flag instead of terminating the full run. This preserves minority
claims while keeping the cost at one request per question.

## Support and dependency ablation

Contexts supporting the same candidate are grouped when they share a source
domain or have token Jaccard similarity above the frozen threshold. Candidate
support can therefore count independent evidence groups rather than raw
snippets. This correction was evaluated as a predeclared ablation. On dev it
did not improve choice accuracy or Brier score, so the frozen selected model
uses raw passage support. The corrected version remains reported, not silently
discarded.

## Features

Each candidate has:

1. log support count;
2. support share;
3. mean directness;
4. maximum and mean answer-evidence entailment;
5. answer-type compatibility;
6. temporal compatibility;
7. source-category shares;
8. raw-to-independent redundancy ratio.

Query relevance is a retrieval gate rather than a truth signal. Novelty is used
only for dependence correction. Entropy is not used as a credibility feature.

## Regularized conditional choice

For standardized candidate feature vector \(x_{qa}\):

\[
s_{qa}=\beta^\top x_{qa},\qquad
P(Z_q=a\mid X_q,\beta)=
\frac{\exp(s_{qa})}{\sum_b\exp(s_{qb})}.
\]

Coefficients use a zero-centred Gaussian penalty/prior:

\[
\beta\sim\mathcal N(0,\sigma^2I).
\]

Training obtains the MAP estimate. Temperature scaling and the abstention
threshold are fitted on labelled, multi-candidate dev questions. This
point-estimate model is the frozen selected candidate.

The Bayesian ablation additionally uses the inverse Hessian of the negative log
posterior as a Laplace covariance and averages softmax probabilities over
deterministic Monte-Carlo coefficient samples. Its temperature is calibrated
against those posterior-predictive probabilities, rather than MAP logits. The
Bayesian variant did not improve dev accuracy or Brier score and is therefore
not presented as the selected primary method.

Small fixed label smoothing makes the fit less brittle to QACC's noisy single-
annotator labels. The abstention threshold is selected on labelled,
multi-candidate dev questions, subject
to a predeclared minimum coverage. Single-candidate softmax probabilities are
identically one and therefore contain no information for calibrating selective
prediction; they remain in end-to-end answer evaluation and candidate-recall
accounting. Evaluation reports a separate multi-candidate slice so that these
trivial conditional probabilities cannot hide ranking uncertainty.

Contradictory signals lower or split conditional probability mass. They do not
need to satisfy simultaneous moment constraints, so the method has no
constraint-feasibility failure. Probabilities remain conditional on the
extracted candidate set and must not be interpreted as universal source or
document credibility.
