from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence
from urllib.parse import urlparse

from .schemas import Candidate, Context


def normalize_domain(value: str | None) -> str:
    if not value:
        return "unknown"
    candidate = value if "://" in value else f"https://{value}"
    host = (urlparse(candidate).hostname or value).lower().rstrip(".")
    return host.removeprefix("www.")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def evidence_dependency_groups(
    candidate: Candidate,
    contexts: Sequence[Context],
    *,
    near_duplicate_jaccard: float = 0.88,
) -> list[list[int]]:
    indices = list(dict.fromkeys(item.context_index for item in candidate.observations))
    if not indices:
        return []
    parent = {index: index for index in indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_index = {context.index: context for context in contexts}
    for left_pos, left in enumerate(indices):
        for right in indices[left_pos + 1 :]:
            left_context, right_context = by_index[left], by_index[right]
            same_domain = normalize_domain(left_context.source) == normalize_domain(right_context.source)
            copied = token_jaccard(left_context.text, right_context.text) >= near_duplicate_jaccard
            if same_domain or copied:
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        groups[find(index)].append(index)
    return list(groups.values())


def source_category(source: str | None) -> str:
    domain = normalize_domain(source)
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return "official"
    if domain in {"wikipedia.org", "britannica.com"}:
        return "reference"
    if domain in {"reddit.com", "quora.com", "twitter.com", "x.com", "youtube.com"}:
        return "social"
    if any(token in domain for token in ("news", "times", "guardian", "reuters", "bbc")):
        return "news"
    return "other"
