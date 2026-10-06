"""Simple source and text helpers used by ARSR."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def normalize_domain(value: str | None) -> str:
    """Reduce a source URL to a comparable host name."""

    if not value:
        return "unknown"
    candidate = value if "://" in value else f"https://{value}"
    host = (urlparse(candidate).hostname or value).lower().rstrip(".")
    return host.removeprefix("www.")


def _tokens(text: str) -> set[str]:
    """Create a small lowercase token set for overlap features."""

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_jaccard(left: str, right: str) -> float:
    """Return word overlap from zero to one."""
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def source_category(source: str | None) -> str:
    """Put a domain in one simple source group."""
    domain = normalize_domain(source)
    # Categories are broad signals, not claims that one source is correct.
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return "official"
    if domain in {"wikipedia.org", "britannica.com"}:
        return "reference"
    if domain in {"reddit.com", "quora.com", "twitter.com", "x.com", "youtube.com"}:
        return "social"
    if any(token in domain for token in ("news", "times", "guardian", "reuters", "bbc")):
        return "news"
    return "other"
