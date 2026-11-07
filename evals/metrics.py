"""Deterministic report checks. They cost nothing and need no model."""

import re
from collections import Counter

from src.utils import ranking
from src.utils.claims import sentences
from src.utils.sources import CITE_RE, cited_numbers, count_misplaced_citations, strip_sources_section

MAX_GROUP_SIZE = 3
MAX_TITLE_LENGTH = ranking.MAX_TITLE_LENGTH + 1
MAX_WRONG_LANGUAGE = 0.05

_RULE = re.compile(r"\n[ \t]*-{3,}[ \t]*\n")
_GROUP = re.compile(r"(?:\[\d{1,3}\])+")
_ADJACENT = re.compile(r"\][ \t]+\[\d")
_WORD = re.compile(r"[^\W\d_]+")

# Words that mark a language and are not words in the other one. Used only to spot sentences that
# are clearly in the wrong language (for example an English quotation inside a Türkçe report).
STOPWORDS = {
    "en": frozenset(
        "the of and that with which from this are was were been have has their they these than into also such "
        "will would could while between because although however".split()),
    "tr": frozenset(
        "ve bir bu için ile olarak daha gibi olan ancak veya çok kadar sonra ise değil ki şu bunun bunlar "
        "olduğu ayrıca göre hem fakat çünkü üzerinde arasında yani".split()),
}


def _norm(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


def _site(host: str) -> str:
    return ".".join(host.split(".")[-2:])


def language_mix(report: str, language: str | None) -> dict:
    """Sentences that are clearly written in the other one of English and Türkçe than the report's language."""
    own = STOPWORDS.get(language or "")
    other = STOPWORDS["tr" if language == "en" else "en"]
    body = [s for s in sentences(strip_sources_section(report)) if len(s.split()) >= 4]
    wrong = []
    if own:
        for sentence in body:
            words = _WORD.findall(CITE_RE.sub("", sentence).lower())
            own_hits, other_hits = sum(w in own for w in words), sum(w in other for w in words)
            if other_hits >= 3 and other_hits >= 2 * own_hits:
                wrong.append(sentence[:160])
    return {"sentences": len(body), "wrong": len(wrong), "examples": wrong[:5]}


def split_report(report: str) -> tuple[str, str, str, bool]:
    """Return (introduction, body, conclusion, structured). `structured` is False when the
    three parts could not be told apart."""
    main = strip_sources_section(report)
    parts = [part.strip() for part in _RULE.split(main)]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], True
    return main, main, main, False


def evaluate_report(report: str, sources: list[dict], language: str | None = None) -> dict:
    introduction, body, conclusion, structured = split_report(report)
    main = strip_sources_section(report)
    mix = language_mix(report, language)

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
        "citations_after_punctuation": count_misplaced_citations(main),
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
        "wrong_language_sentences": mix["wrong"],
        "wrong_language_share": round(mix["wrong"] / mix["sentences"], 3) if mix["sentences"] else 0.0,
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
    "language_consistent": lambda m: m["wrong_language_share"] <= MAX_WRONG_LANGUAGE,
}


def run_checks(metrics: dict) -> dict[str, bool]:
    return {name: bool(check(metrics)) for name, check in HARD_CHECKS.items()}
