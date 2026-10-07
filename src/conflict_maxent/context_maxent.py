"""Notebook-aligned context features for uniform-prior soft MaxEnt."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .source_utils import normalize_domain
from .schemas import EvidenceObservation, PreparedExample
from .soft_maxent import soft_information_projection, support_voting_weights


CONTEXT_MAXENT_PROTOCOL_VERSION = "2.0"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


TRUSTED_DOMAINS = {
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "britannica.com",
    "nationalgeographic.com",
    "ncbi.nlm.nih.gov",
    "nytimes.com",
    "reuters.com",
    "smithsonianmag.com",
    "theguardian.com",
    "wikipedia.org",
}
UGC_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "medium.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "tiktok.com",
    "tumblr.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}
EDUCATION_DOMAINS = {
    "byjus.com",
    "chegg.com",
    "cliffsnotes.com",
    "khanacademy.org",
    "quizlet.com",
    "sparknotes.com",
    "toppr.com",
}
VERTICAL_DOMAINS = {
    "espn.com",
    "ew.com",
    "genius.com",
    "hollywoodreporter.com",
    "imdb.com",
    "rottentomatoes.com",
    "screenrant.com",
    "secondhandsongs.com",
    "songfacts.com",
}
CMS_HOSTS = ("wordpress.com", "blogspot.com", "wixsite.com", "weebly.com", "squarespace.com")
SUSPECT_TLDS = (".info", ".biz", ".xyz", ".click", ".top")
GOV_EDU_PATTERN = re.compile(r"\.gov(?:\.[a-z]{2})?$|\.edu$|\.ac\.[a-z]{2}$")


@dataclass
class SourceUnit:
    """One original QACC context used as one evidence unit."""

    context_positions: list[int]
    representative: str
    cluster_id: int = -1


class ContextFeatureRuntime(Protocol):
    """Small model interface used by the real runtime and test doubles."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def contradiction_probabilities(
        self, pairs: Sequence[tuple[str, str]]
    ) -> np.ndarray: ...


class SentenceTransformersRuntime:
    """Load the two local models used in the original notebook."""

    def __init__(
        self,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedding_model_name = embedding_model
        self.nli_model_name = nli_model
        self.batch_size = batch_size
        model_kwargs = {} if device is None else {"device": device}
        self.embedding_model = SentenceTransformer(embedding_model, **model_kwargs)
        self.nli_model = CrossEncoder(nli_model, **model_kwargs)
        labels = getattr(self.nli_model.model.config, "id2label", {})
        contradiction = [
            int(index)
            for index, label in labels.items()
            if "contrad" in str(label).lower()
        ]
        # This notebook model normally stores contradiction in column zero.
        self.contradiction_index = contradiction[0] if contradiction else 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized sentence embeddings."""

        return np.asarray(
            self.embedding_model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )

    def contradiction_probabilities(
        self, pairs: Sequence[tuple[str, str]]
    ) -> np.ndarray:
        """Return the NLI contradiction probability for every ordered pair."""

        if not pairs:
            return np.zeros(0)
        probabilities = np.asarray(
            self.nli_model.predict(
                list(pairs),
                batch_size=self.batch_size,
                apply_softmax=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )
        return probabilities[:, self.contradiction_index]


def _domain_matches(domain: str, root: str) -> bool:
    """Match one registered domain and all of its subdomains."""

    return domain == root or domain.endswith(f".{root}")


def domain_reliability_score(source: str | None) -> float:
    """Compute notebook feature g4 from the URL domain only."""

    domain = normalize_domain(source)
    if not domain or domain == "unknown":
        return 0.4
    if any(_domain_matches(domain, root) for root in TRUSTED_DOMAINS):
        return 1.0
    if GOV_EDU_PATTERN.search(domain) or "pubmed" in domain:
        return 0.9
    if any(_domain_matches(domain, root) for root in UGC_DOMAINS):
        return 0.2
    if any(_domain_matches(domain, root) for root in CMS_HOSTS):
        return 0.25
    if domain.endswith(SUSPECT_TLDS):
        return 0.35
    if any(_domain_matches(domain, root) for root in EDUCATION_DOMAINS):
        return 0.6
    if any(_domain_matches(domain, root) for root in VERTICAL_DOMAINS):
        return 0.7
    return 0.4


def _unit_embeddings(units: Sequence[SourceUnit], embeddings: np.ndarray) -> np.ndarray:
    """Normalize the embedding of each original context."""

    values = []
    for unit in units:
        value = embeddings[unit.context_positions].mean(axis=0)
        norm = float(np.linalg.norm(value))
        values.append(value / norm if norm else value)
    return np.stack(values)


def _cluster_units(units: list[SourceUnit], embeddings: np.ndarray) -> int:
    """Assign the viewpoint clusters used by h_j."""

    if len(units) == 1:
        units[0].cluster_id = 0
        return 1
    from sklearn.cluster import KMeans

    count = min(len(units), max(2, len(units) // 2))
    labels = KMeans(n_clusters=count, random_state=42, n_init=10).fit_predict(embeddings)
    for unit, label in zip(units, labels):
        unit.cluster_id = int(label)
    return count


def _g2(units: Sequence[SourceUnit], runtime: ContextFeatureRuntime) -> np.ndarray:
    """Mean symmetric NLI contradiction with all other units."""

    count = len(units)
    if count == 1:
        return np.zeros(1)
    texts = [item.representative for item in units]
    pairs = [(texts[i], texts[j]) for i in range(count) for j in range(count) if i != j]
    probabilities = runtime.contradiction_probabilities(pairs)
    matrix = np.zeros((count, count), dtype=float)
    cursor = 0
    for left in range(count):
        for right in range(count):
            if left != right:
                matrix[left, right] = probabilities[cursor]
                cursor += 1
    matrix = 0.5 * (matrix + matrix.T)
    return matrix.sum(axis=1) / (count - 1)


def _g6(query: np.ndarray, unit_embeddings: np.ndarray) -> np.ndarray:
    """MMR information gain: query relevance minus maximum redundancy."""

    norm = float(np.linalg.norm(query))
    normalized_query = query / norm if norm else query
    relevance = unit_embeddings @ normalized_query
    if len(unit_embeddings) == 1:
        return relevance
    similarity = unit_embeddings @ unit_embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    return relevance - similarity.max(axis=1)


def _constraint_target(values: np.ndarray, sign: float, initial: float) -> float:
    """Keep the notebook's twenty-percent margin from an extreme target."""

    span = float(values.max() - values.min())
    if sign > 0:
        return min(initial, float(values.max()) - 0.2 * span)
    return max(initial, float(values.min()) + 0.2 * span)


def build_context_maxent_record(
    example: PreparedExample,
    observations: Sequence[EvidenceObservation],
    runtime: ContextFeatureRuntime,
    *,
    uniform_mix: float = 0.1,
    cluster_floor: float = 0.05,
    slack_penalty: float = 5.0,
    max_iter: int = 200,
    tolerance: float = 1e-6,
) -> dict:
    """Build the three separated conditions for one QACC question."""

    if not example.contexts:
        raise ValueError(f"Example has no contexts: {example.question_id}")
    support, support_features = support_voting_weights(
        example, observations, uniform_mix=uniform_mix
    )
    uniform = np.full(len(example.contexts), 1.0 / len(example.contexts))

    texts = [context.text for context in example.contexts]
    encoded = runtime.encode([example.question, *texts])
    query_embedding, context_embeddings = encoded[0], encoded[1:]
    units = [SourceUnit([position], text) for position, text in enumerate(texts)]
    unit_embeddings = _unit_embeddings(units, context_embeddings)
    cluster_count = _cluster_units(units, unit_embeddings)

    # Every original context starts with the same prior mass.
    unit_prior = np.full(len(units), 1.0 / len(units))
    g2 = _g2(units, runtime)
    g6 = _g6(query_embedding, unit_embeddings)
    g4 = np.asarray(
        [
            float(
                np.mean(
                    [
                        domain_reliability_score(example.contexts[position].source)
                        for position in unit.context_positions
                    ]
                )
            )
            for unit in units
        ]
    )

    columns: list[np.ndarray] = []
    targets: list[float] = []
    signs: list[float] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray, sign: float, target: float) -> None:
        if np.ptp(values) < 1e-12:
            return
        columns.append(values[:, None])
        targets.append(_constraint_target(values, sign, target))
        signs.append(sign)
        names.append(name)

    g2_target = (
        float(g2.max())
        if float(g2.max()) < 0.05
        else float((g2.min() + g2.mean()) / 2)
    )
    add("g2_contradiction", g2, -1.0, g2_target)
    add("g6_information_gain", g6, 1.0, float(g6.mean()))

    if cluster_count > 1:
        membership = np.zeros((len(units), cluster_count), dtype=float)
        for index, unit in enumerate(units):
            membership[index, unit.cluster_id] = 1.0
        floor = min(cluster_floor, 0.5 / cluster_count)
        for cluster in range(cluster_count):
            columns.append(membership[:, cluster : cluster + 1])
            targets.append(floor)
            signs.append(1.0)
            names.append(f"h_{cluster}")

    add("g4_source_reliability", g4, 1.0, float(g4.mean()))
    matrix = np.hstack(columns) if columns else np.empty((len(units), 0))
    projection = soft_information_projection(
        unit_prior,
        matrix,
        targets,
        signs=signs,
        slack_penalty=slack_penalty,
        max_iter=max_iter,
        tolerance=tolerance,
    )

    # Return the weights in the original context order.
    context_weights = np.zeros(len(example.contexts), dtype=float)
    unit_by_position: dict[int, int] = {}
    for unit_index, unit in enumerate(units):
        share = projection.distribution[unit_index]
        for position in unit.context_positions:
            context_weights[position] = share
            unit_by_position[position] = unit_index

    passage_features = []
    for position, base in enumerate(support_features):
        unit_index = unit_by_position[position]
        unit = units[unit_index]
        passage_features.append(
            {
                **base,
                "unit_index": unit_index,
                "viewpoint_cluster": unit.cluster_id,
                "g2_contradiction": float(g2[unit_index]),
                "g6_information_gain": float(g6[unit_index]),
                "g4_source_reliability": float(g4[unit_index]),
            }
        )

    return {
        "question_id": example.question_id,
        "uniform_weights": uniform.tolist(),
        "support_voting_weights": support.tolist(),
        "maxent_weights": context_weights.tolist(),
        "passage_features": passage_features,
        "projection": {
            **{
                key: value
                for key, value in asdict(projection).items()
                if key not in {"distribution", "lambdas"}
            },
            "prior": "uniform_document_mass",
            "unit_count": len(units),
            "constraints": names,
            "targets": targets,
            "signs": signs,
            "lambdas": projection.lambdas.tolist(),
        },
    }


def write_context_maxent_cache(
    examples: Sequence[PreparedExample],
    extractions: dict[str, Sequence[EvidenceObservation]],
    runtime: ContextFeatureRuntime,
    output_path: Path,
    *,
    resume: bool = True,
    progress_every: int = 10,
    **settings: float | int,
) -> dict:
    """Build expensive local features with safe question-level resume."""

    missing = [item.question_id for item in examples if item.question_id not in extractions]
    if missing:
        raise ValueError(
            "Missing extraction records for question IDs: " + ", ".join(missing[:5])
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    seed_path = temporary if resume and temporary.exists() else output_path
    records: dict[str, dict] = {}
    if resume and seed_path.exists():
        for line in seed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[str(record["question_id"])] = record
    requested = {item.question_id for item in examples}
    records = {key: value for key, value in records.items() if key in requested}
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records.values()
        ),
        encoding="utf-8",
    )
    extracted = 0
    with temporary.open("a", encoding="utf-8") as handle:
        for position, example in enumerate(examples, start=1):
            if example.question_id in records:
                continue
            record = build_context_maxent_record(
                example,
                extractions[example.question_id],
                runtime,
                **settings,
            )
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            records[example.question_id] = record
            extracted += 1
            if progress_every and position % progress_every == 0:
                print(f"maxent features: {position}/{len(examples)}")
    temporary.replace(output_path)
    diagnostics = [record["projection"] for record in records.values()]
    return {
        "requested": len(examples),
        "reused": len(examples) - extracted,
        "extracted": extracted,
        "converged": sum(bool(item["converged"]) for item in diagnostics),
        "mean_max_violation": float(
            np.mean([item["max_violation"] for item in diagnostics])
        )
        if diagnostics
        else 0.0,
        "output": str(output_path),
    }


__all__ = [
    "CONTEXT_MAXENT_PROTOCOL_VERSION",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_NLI_MODEL",
    "ContextFeatureRuntime",
    "SentenceTransformersRuntime",
    "build_context_maxent_record",
    "domain_reliability_score",
    "write_context_maxent_cache",
]
