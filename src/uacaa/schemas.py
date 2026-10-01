from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Context:
    index: int
    text: str
    source: str | None = None


@dataclass(frozen=True)
class EvidenceObservation:
    context_index: int
    candidate_answer: str
    evidence_quote: str = ""
    directness: float = 0.0
    entailment: float = 0.0
    answer_type_match: float = 0.5
    temporal_fit: float = 0.5

    def __post_init__(self) -> None:
        for name in ("directness", "entailment", "answer_type_match", "temporal_fit"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass
class Candidate:
    answer: str
    aliases: list[str] = field(default_factory=list)
    observations: list[EvidenceObservation] = field(default_factory=list)


@dataclass
class PreparedExample:
    question_id: str
    split: str
    question: str
    contexts: list[Context]
    supervision: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreparedExample":
        return cls(
            question_id=str(value["question_id"]),
            split=str(value["split"]),
            question=str(value["question"]),
            contexts=[Context(**item) for item in value.get("contexts", [])],
            supervision=value.get("supervision"),
            evaluation=value.get("evaluation"),
            quality_flags=list(value.get("quality_flags", [])),
        )


@dataclass(frozen=True)
class ChoiceSet:
    question_id: str
    answers: tuple[str, ...]
    features: np.ndarray
    correct_index: int | None = None
    references: tuple[str, ...] = ()
    has_conflict: bool | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.features, dtype=float)
        object.__setattr__(self, "features", matrix)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.answers):
            raise ValueError("features must have one row per candidate answer")
        if self.correct_index is not None and not 0 <= self.correct_index < len(self.answers):
            raise ValueError("correct_index is outside the candidate set")


@dataclass(frozen=True)
class Prediction:
    question_id: str
    answer: str | None
    probabilities: tuple[float, ...]
    confidence: float
    abstained: bool
