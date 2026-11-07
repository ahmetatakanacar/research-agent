"""Figures in a report against its sources. Free and code-only: it needs no model."""

import re

from src.utils.claims import extract_claims

_NUMBER = re.compile(r"(\d[\d.,]*\d|\d)(\s?%)?")
_GROUPED = re.compile(r"\d{1,3}(?:[.,]\d{3})+")


def number_forms(raw: str) -> set[str]:
    """The ways a number may be written in another text: as is, with a decimal comma read as a point, and,
    when it looks like thousands grouping (1,234 or 1.234), without the separators."""
    raw = raw.strip(".,")
    forms = {raw, raw.replace(",", ".")}
    if _GROUPED.fullmatch(raw):
        forms.add(re.sub(r"[.,]", "", raw))
    return forms


def significant_numbers(text: str) -> list[str]:
    """Numbers worth checking: bare integers up to 10 are left out (counts such as "two studies"), unless
    they are percentages."""
    found = []
    for match in _NUMBER.finditer(text):
        raw, percent = match.group(1).strip(".,"), match.group(2)
        if not raw or (not percent and raw.isdigit() and int(raw) <= 10):
            continue
        found.append(raw)
    return found


def all_number_forms(text: str) -> set[str]:
    forms: set[str] = set()
    for match in _NUMBER.finditer(text):
        forms |= number_forms(match.group(1))
    return forms


def number_grounding(report: str, texts: dict[int, dict], topic: str = "") -> dict:
    """Every figure in a cited sentence should appear in at least one source that sentence cites. This is a
    free, code-only check of the figures. A figure the user's own question contains is left out: a report
    that says "no source confirms 1987" is quoting the question, not a source. A figure the report derives
    itself (a sum, a difference) is listed too, so read the rows before treating one as invented.

    An ungrounded figure that another source of the report does contain (`elsewhere`) points to a citation on
    the wrong source; one that no source contains comes from the model itself."""
    cache = {n: all_number_forms(str(text.get("content") or "")) for n, text in texts.items()}
    asked = all_number_forms(topic)
    rows, total = [], 0
    for claim in extract_claims(report, max_claims=10_000):
        available = set().union(*(cache.get(n, set()) for n in claim["cites"]))
        for raw in significant_numbers(claim["text"]):
            forms = number_forms(raw)
            if forms & asked:
                continue
            total += 1
            if not forms & available:
                elsewhere = sorted(n for n, known in cache.items() if forms & known)
                rows.append({"number": raw, "claim": claim["text"][:200], "cites": claim["cites"],
                             "elsewhere": elsewhere})
    return {
        "numbers": total, "ungrounded": len(rows), "misplaced": sum(1 for row in rows if row["elsewhere"]),
        "grounding": round(1 - len(rows) / total, 3) if total else None, "rows": rows,
    }
