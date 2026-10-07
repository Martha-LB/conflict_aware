# Soft MaxEnt for Conflicting-Context QA

## Research question

Retrieval-augmented generation (RAG) gives a language model several web
contexts. These contexts sometimes give different answers. The central question
of this project is:

> When contexts conflict and source information is limited, can the correct
> answer be identified from the available input?

The project builds on Liu et al. (2025), [*Open Domain Question Answering with
Conflicting Contexts*](https://aclanthology.org/2025.findings-naacl.99/). This
paper provides the human-annotated QACC dataset and an explain-then-answer
method. QACC includes text, web links, and some time information. However,
complete author, date, and reliability information is not available for most
contexts.

The experiment uses 813 official test questions: 207 conflict and 606
non-conflict questions. Test inputs and references are kept in different files.
References are used only after all answers are frozen. Scores use the public
QACC normalized Exact Match (EM) and token F1 evaluator.

## Method

The main baseline is **Direct-RAG** with GPT-6 Luna. It reads all contexts and
returns a short answer. The experiment then separates three passage-weighting
conditions. They use the same model, prompt, contexts, context order, and output
format. Only the numeric weights change.

1. **Uniform** gives every context the same weight.
2. **Independent support** gives more weight to answers supported by different
   web domains. This is a simple voting baseline.
3. **Soft MaxEnt** starts from a uniform prior. It uses four feature families
   restored from the original project notebook:
   - \(g_2\): NLI contradiction with other contexts;
   - \(h_j\): viewpoint clusters;
   - \(g_6\): question relevance minus repeated information;
   - \(g_4\): a coarse reliability tier from the web domain.

Soft MaxEnt finds weights \(q\) close to the uniform prior while trying to meet
the feature constraints:

\[
\min_{q,\xi}\;D_{KL}(q\|p)+C\sum_j\xi_j.
\]

The slack variables \(\xi\) keep the optimization valid when several conditions
cannot be satisfied together. Here \(C=5\). Every original context is one
evidence unit. Publication-time feature \(g_3\) is not used because QACC time
metadata is too incomplete. No gold answer or conflict label enters weighting.

The study also reports the QACC explain-then-answer prompt and a candidate
oracle. The oracle uses references to select the best available candidate. It is
an upper bound, not a usable QA method.

## Main results

The fixed official-test results are below.

| System | Overall EM | Overall F1 | Conflict EM | Non-conflict EM |
|---|---:|---:|---:|---:|
| Luna Direct-RAG | 0.5855 | 0.7039 | 0.4879 | 0.6188 |
| Uniform weighted prompt | 0.5966 | 0.7237 | 0.4783 | 0.6370 |
| Independent support | 0.6052 | 0.7240 | 0.5169 | 0.6353 |
| Soft MaxEnt | 0.6052 | 0.7224 | 0.5024 | 0.6403 |
| QACC explain-then-answer | 0.6212 | 0.7487 | 0.5217 | 0.6551 |
| Candidate oracle | 0.7023 | 0.8115 | 0.6087 | 0.7343 |

Soft MaxEnt improves overall EM from 0.5855 to 0.6052. Uniform weighting already
reaches 0.5966, so part of the gain may come from the clearer weighted prompt.
The four MaxEnt features add only 0.0086 EM over this prompt control. Independent
support reaches the same overall EM as Soft MaxEnt and has higher conflict EM.
Explain-then-answer is the strongest usable method. The much higher candidate
oracle shows that a good answer often exists but is difficult to identify from
the available evidence. The main result is therefore limited empirical
identifiability on QACC: conflict is visible, but the available evidence often
does not clearly show which answer is correct.

## Run

Use Python 3.11 or newer. The first run downloads QACC, AmbigQA reference aliases,
and two local feature models. It reuses finished API caches.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install ".[test]"

export OPENAI_API_KEY="your-key"
python -m pytest -q
caffeinate -i ./get_results.sh
~~~
