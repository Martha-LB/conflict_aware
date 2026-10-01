from __future__ import annotations

import re
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
    return normalize_answer(value) in _NO_ANSWER


def token_f1(left: object, right: object) -> float:
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
    left_key = normalize_answer(left)
    right_key = normalize_answer(right)
    return bool(left_key) and left_key == right_key


def answers_match(left: object, right: object, *, threshold: float = 0.8) -> bool:
    left_key = normalize_answer(left)
    right_key = normalize_answer(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_numbers = set(re.findall(r"\d+", left_key))
    right_numbers = set(re.findall(r"\d+", right_key))
    if left_numbers and right_numbers and left_numbers != right_numbers:
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
    target_numbers = set(re.findall(r"\d+", " ".join(target_tokens)))
    candidate_numbers = set(re.findall(r"\d+", " ".join(candidate_tokens)))
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
    scores = [token_f1(target, candidate) for candidate in candidates]
    exact = [index for index, candidate in enumerate(candidates) if answers_match(target, candidate)]
    if exact:
        return max(exact, key=scores.__getitem__)
    if not scores:
        return None
    index = int(max(range(len(scores)), key=scores.__getitem__))
    return index if scores[index] >= threshold else None
