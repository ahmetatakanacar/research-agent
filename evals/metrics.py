"""Deterministic report checks. They cost nothing and need no model."""

import re
from collections import Counter

from src.utils import ranking
from src.utils.sources import CITE_RE, cited_numbers, strip_sources_section

MAX_GROUP_SIZE = 3
MAX_TITLE_LENGTH = ranking.MAX_TITLE_LENGTH + 1

_RULE = re.compile(r"\n[ \t]*-{3,}[ \t]*\n")
_GROUP = re.compile(r"(?:\[\d{1,3}\])+")
_ADJACENT = re.compile(r"\][ \t]+\[\d")
_AFTER_PUNCTUATION = re.compile(r"[.!?][ \t]*\[\d{1,3}\]")


def _norm(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


def _site(host: str) -> str:
    return ".".join(host.split(".")[-2:])


def split_report(report: str) -> tuple[str, str, str, bool]:
    """Return (introduction, body, conclusion, structured). `structured` is False when the
    three parts could not be told apart."""
    main = strip_sources_section(report)
    parts = [part.strip() for part in _RULE.split(main)]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], True
    return main, main, main, False


def evaluate_report(report: str, sources: list[dict]) -> dict:
    introduction, body, conclusion, structured = split_report(report)
    main = strip_sources_section(report)

    numbers = {source["n"] for source in sources}
    cited = cited_numbers(main)
    groups = _GROUP.findall(main)
    group_numbers = [[int(n) for n in re.findall(r"\d+", group)] for group in groups]

    urls = [str(source["url"]) for source in sources]
    titles = [str(source.get("title") or "") for source in sources]
    hosts = [ranking.host_of(url) for url in urls]
    total = len(sources)

    def share(count: int) -> float:
        return round(count / total, 3) if total else 0.0

    authority = sum(1 for url in urls if ranking.quality_bonus(url) > 0)
    marketing = sum(
        1 for url, title in zip(urls, titles, strict=True) if ranking._marketing_penalty({"url": url, "title": title}) > 0
    )

    return {
        "structured": structured,
        "words": len(body.split()),
        "subheadings": len(re.findall(r"^###\s", body, re.M)),
        "sources": total,
        "dangling_citations": len(cited - numbers),
        "uncited_sources": len(numbers - cited),
        "max_group_size": max((len(g) for g in group_numbers), default=0),
        "oversized_groups": sum(1 for g in group_numbers if len(g) > MAX_GROUP_SIZE),
        "unsorted_groups": sum(1 for g in group_numbers if g != sorted(g)),
        "adjacent_groups": len(_ADJACENT.findall(main)),
        "citations_after_punctuation": len(_AFTER_PUNCTUATION.findall(main)),
        "introduction_cited": bool(CITE_RE.search(introduction)),
        "body_cited": bool(CITE_RE.search(body)),
        "conclusion_cited": bool(CITE_RE.search(conclusion)),
        "duplicate_urls": total - len(set(urls)),
        "duplicate_titles": total - len({(_norm(t), _site(h)) for t, h in zip(titles, hosts, strict=True)}),
        "blocked_sources": sum(1 for url in urls if ranking.is_blocked(url)),
        "marketing_sources": marketing,
        "authority_share": share(authority),
        "max_host_share": share(max(Counter(hosts).values())) if hosts else 0.0,
        "cut_off_titles": sum(1 for title in titles if title.rstrip().endswith(("…", "..."))),
        "long_titles": sum(1 for title in titles if len(title) > MAX_TITLE_LENGTH),
    }


HARD_CHECKS = {
    "three_part_structure": lambda m: m["structured"],
    "no_dangling_citations": lambda m: m["dangling_citations"] == 0,
    "no_uncited_sources": lambda m: m["uncited_sources"] == 0,
    "groups_at_most_three": lambda m: m["oversized_groups"] == 0,
    "groups_sorted": lambda m: m["unsorted_groups"] == 0,
    "no_adjacent_groups": lambda m: m["adjacent_groups"] == 0,
    "citations_before_punctuation": lambda m: m["citations_after_punctuation"] == 0,
    "introduction_cited": lambda m: m["introduction_cited"],
    "body_cited": lambda m: m["body_cited"],
    "conclusion_cited": lambda m: m["conclusion_cited"],
    "no_duplicate_sources": lambda m: m["duplicate_urls"] == 0 and m["duplicate_titles"] == 0,
    "no_blocked_sources": lambda m: m["blocked_sources"] == 0,
    "titles_complete": lambda m: m["cut_off_titles"] == 0 and m["long_titles"] == 0,
}


def run_checks(metrics: dict) -> dict[str, bool]:
    return {name: bool(check(metrics)) for name, check in HARD_CHECKS.items()}
