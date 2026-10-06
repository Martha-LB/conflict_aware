from __future__ import annotations

import re
from urllib.parse import urlparse


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
