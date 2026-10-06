"""Normalize answers and compute QACC scores."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Sequence


_ARTICLES = {"a", "an", "the"}
_NO_ANSWER = {"", "n a", "na", "none", "no answer", "not available", "unknown"}
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
_COUNT_UNITS = {"episode", "episodes", "seat", "seats", "time", "times"}


def normalize_answer(value: object) -> str:
    """Loose normalization for internal matching."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # A trailing all-zero decimal does not change a quantity (for example,
    # "$160,000.00" versus "$160,000"). Preserve all non-zero decimals.
    text = re.sub(r"(?<=\d)\.0+(?!\d)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)


def is_no_answer(value: object) -> bool:
    """Recognize the abstention strings used by the prompts."""

    return normalize_answer(value) in _NO_ANSWER


def token_f1(left: object, right: object) -> float:
    """Compute multiset token F1 for internal answer matching."""

    left_tokens = normalize_answer(left).split()
    right_tokens = normalize_answer(right).split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2.0 * precision * recall / (precision + recall)


def exact_match(left: object, right: object) -> bool:
    """Compare two answers after loose internal normalization."""

    left_key = normalize_answer(left)
    right_key = normalize_answer(right)
    return bool(left_key) and left_key == right_key


def qacc_normalize_text(value: object) -> str:
    """Reproduce the normalization in QACC's released evaluator.

    This is intentionally separate from :func:`normalize_answer`: the latter
    is a more permissive internal canonicalizer used for candidate clustering,
    whereas reported QACC EM/F1 must remain comparable with the paper.
    """

    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def qacc_exact_match(left: object, right: object) -> bool:
    """QACC normalized exact match for one prediction/reference pair."""

    return qacc_normalize_text(left) == qacc_normalize_text(right)


def qacc_token_f1(left: object, right: object) -> float:
    """Reproduce QACC's released word-overlap F1 implementation."""

    left_tokens = qacc_normalize_text(left).split()
    right_tokens = qacc_normalize_text(right).split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    # QACC's public evaluator uses set overlap rather than multiset overlap.
    common = set(left_tokens) & set(right_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(left_tokens)
    recall = len(common) / len(right_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answers_match(left: object, right: object, *, threshold: float = 0.8) -> bool:
    """Match answer aliases inside the pipeline."""
    left_key = normalize_answer(left)
    right_key = normalize_answer(right)
    if not left_key or not right_key:
        return False
    # Exact normalized text is always the safest alias match.
    if left_key == right_key:
        return True
    # Numeric order is semantic in ratios, ranges, rankings, and many dates.
    # A bag/set comparison incorrectly aliases inverse relations such as 3:2
    # and 2:3 even though QACC's released evaluator distinguishes them.
    left_numbers = tuple(re.findall(r"\d+", left_key))
    right_numbers = tuple(re.findall(r"\d+", right_key))
    if (left_numbers or right_numbers) and left_numbers != right_numbers:
        return False
    return token_f1(left_key, right_key) >= threshold


def target_matches_candidate(
    target: object,
    candidate: object,
    *,
    threshold: float = 0.8,
) -> bool:
    """Map a gold/reference answer to a possibly more explicit candidate.

    This asymmetric matcher is deliberately restricted to candidate-set mapping.
    Final answer EM/F1 remains strict, so verbose predictions do not receive an
    inflated end metric.
    """
    if answers_match(target, candidate, threshold=threshold):
        return True
    target_tokens = normalize_answer(target).split()
    candidate_tokens = normalize_answer(candidate).split()
    if not target_tokens or not candidate_tokens:
        return False
    # Count questions often annotate the unit even though the extractor
    # correctly returns only the minimal numeric slot filler.
    if len(target_tokens) == 2 and len(candidate_tokens) == 1:
        target_number = _NUMBER_WORDS.get(target_tokens[0], target_tokens[0])
        candidate_number = _NUMBER_WORDS.get(candidate_tokens[0], candidate_tokens[0])
        if (
            target_number.isdigit()
            and target_number == candidate_number
            and target_tokens[1] in _COUNT_UNITS
        ):
            return True
    target_numbers = tuple(re.findall(r"\d+", " ".join(target_tokens)))
    candidate_numbers = tuple(re.findall(r"\d+", " ".join(candidate_tokens)))
    if target_numbers or candidate_numbers:
        if target_numbers != candidate_numbers:
            return False
    # Candidate-set mapping tolerates minor orthographic variation (plural
    # suffixes, common month abbreviations, and single-character name typos).
    # This does not relax final answer EM/F1.
    if SequenceMatcher(
        None,
        " ".join(target_tokens),
        " ".join(candidate_tokens),
    ).ratio() >= 0.94:
        return True
    target_counts = Counter(target_tokens)
    candidate_counts = Counter(candidate_tokens)
    return all(candidate_counts[token] >= count for token, count in target_counts.items())


def best_answer_index(
    target: object,
    candidates: Sequence[object],
    *,
    threshold: float = 0.8,
) -> int | None:
    """Return the closest candidate, or None when no candidate is close."""

    scores = [token_f1(target, candidate) for candidate in candidates]
    exact = [index for index, candidate in enumerate(candidates) if answers_match(target, candidate)]
    if exact:
        return max(exact, key=scores.__getitem__)
    if not scores:
        return None
    index = int(max(range(len(scores)), key=scores.__getitem__))
    return index if scores[index] >= threshold else None
# These wrapper words may be removed without changing a short answer.
_SAFE_SURFACE_MODIFIERS = {
    "during", "each", "every", "in", "month", "months", "of", "on", "throughout",
}


def safe_surface_equivalent(left: object, right: object) -> bool:
    """Match harmless wrappers while preserving numbers and names."""
    if exact_match(left, right):
        return True
    left_key = normalize_answer(left)
    right_key = normalize_answer(right)
    if not left_key or not right_key:
        return False
    if set(re.findall(r"\d+", left_key)) != set(re.findall(r"\d+", right_key)):
        return False
    left_core = [
        token for token in left_key.split() if token not in _SAFE_SURFACE_MODIFIERS
    ]
    right_core = [
        token for token in right_key.split() if token not in _SAFE_SURFACE_MODIFIERS
    ]
    return bool(left_core) and left_core == right_core
