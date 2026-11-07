"""Cited sentences of a report, shared by the app's fact-check and the evals' free and model-graded checks."""

import re

from src.utils.sources import CITE_RE, cited_numbers, strip_sources_section

MIN_CLAIM_CHARS = 30


def sentences(text: str) -> list[str]:
    text = re.sub(r"^#{1,6}.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    return re.split(r"(?<=[.!?])\s+(?=[\"“(A-ZÇĞİÖŞÜ0-9])", text)


def extract_claims(report: str, max_claims: int = 24) -> list[dict]:
    """Sentences that carry a citation, with the markers removed. Over the cap, an evenly spaced
    sample is kept so the choice is repeatable."""
    claims = []
    for sentence in sentences(strip_sources_section(report)):
        cites = sorted(cited_numbers(sentence))
        text = re.sub(r"\s+([,.;:!?])", r"\1", CITE_RE.sub("", sentence))
        text = re.sub(r"\s+", " ", text).strip()
        if cites and len(text) >= MIN_CLAIM_CHARS:
            claims.append({"text": text, "cites": cites})
    if len(claims) > max_claims:
        claims = [claims[int(i * len(claims) / max_claims)] for i in range(max_claims)]
    for index, claim in enumerate(claims, 1):
        claim["id"] = index
    return claims
