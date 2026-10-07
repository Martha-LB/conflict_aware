"""Simple source and text helpers used by passage weighting."""

from __future__ import annotations

from urllib.parse import urlparse


def normalize_domain(value: str | None) -> str:
    """Reduce a source URL to a comparable host name."""

    if not value:
        return "unknown"
    candidate = value if "://" in value else f"https://{value}"
    host = (urlparse(candidate).hostname or value).lower().rstrip(".")
    return host.removeprefix("www.")
