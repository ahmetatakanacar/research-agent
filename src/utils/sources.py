import re
from typing import Any, Callable, Iterable

from src.utils.objects import SourceDoc
from src.utils.ranking import page_key

MAX_DOC_CHARS = 4000

CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
_SOURCES_HEADING_RE = re.compile(
    r"^#{1,6}[ \t]*(?:sources|references|bibliography|kaynaklar|kaynakça|quellen|quellenverzeichnis"
    r"|références|fuentes|referencias)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def unique_documents(context: Iterable | None) -> list[SourceDoc]:
    """One document per page: the same URL, or two URLs of one page (see `ranking.page_key`), keeps the first."""
    seen: dict[str, SourceDoc] = {}
    for item in context or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        title = str(item.get("title") or "").strip() or url
        key = page_key(url, title)
        if not url or not content or key in seen:
            continue
        seen[key] = {"url": url, "title": title, "content": content, "kind": str(item.get("kind") or "web")}
    return list(seen.values())


def format_numbered(docs: list[SourceDoc], max_chars: int = MAX_DOC_CHARS) -> str:
    blocks = [f"[{i}] {doc['title']} — {doc['url']}\n{doc['content'][:max_chars]}" for i, doc in enumerate(docs, 1)]
    return "\n\n---\n\n".join(blocks)


_CITE_WITH_SPACE = re.compile(r"([ \t]?)\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")


def rewrite_citations(text: str, mapper: Callable[[int], int | None]) -> str:
    def replace(match: re.Match) -> str:
        space = match.group(1)
        numbers: list[int] = []
        for raw in re.split(r"[,;]", match.group(2)):
            mapped = mapper(int(raw))
            if mapped is not None and mapped not in numbers:
                numbers.append(mapped)
        if not numbers:
            following = match.string[match.end():match.end() + 1]
            return space if following.isalnum() else ""
        return space + "".join(f"[{n}]" for n in numbers)

    return _CITE_WITH_SPACE.sub(replace, text)


def cited_numbers(text: str) -> set[int]:
    found: set[int] = set()
    for match in CITE_RE.finditer(text):
        found.update(int(part) for part in re.split(r"[,;]", match.group(1)))
    return found


def merge_sections(sections: list) -> tuple[list[str], list[dict]]:
    registry: dict[str, int] = {}
    sources: list[dict] = []
    merged: list[str] = []

    for section in sections:
        if isinstance(section, dict):
            text = str(section.get("text") or "")
            local = section.get("sources") or []
        else:
            text, local = str(section), []

        def mapper(k: int, local=local) -> int | None:
            if not 1 <= k <= len(local):
                return None
            doc = local[k - 1]
            key = page_key(doc["url"], doc.get("title") or "")
            number = registry.get(key)
            if number is None:
                number = len(sources) + 1
                registry[key] = number
                sources.append({
                    "n": number,
                    "url": doc["url"],
                    "title": doc.get("title") or doc["url"],
                    "content": str(doc.get("content") or "")[:MAX_DOC_CHARS],
                    "kind": doc.get("kind", "web"),
                })
            return number

        merged.append(rewrite_citations(text, mapper))

    return merged, sources


def finalize_citations(parts: list[str], sources: list[dict]) -> tuple[list[str], list[dict]]:
    by_number = {source["n"]: source for source in sources}
    order: dict[int, int] = {}
    used: list[dict] = []

    def mapper(n: int) -> int | None:
        source = by_number.get(n)
        if source is None:
            return None
        if n not in order:
            order[n] = len(used) + 1
            used.append({
                "n": order[n],
                "url": source["url"],
                "title": source.get("title") or source["url"],
                "kind": source.get("kind", "web"),
            })
        return order[n]

    return [rewrite_citations(part, mapper) for part in parts], used


_CITE_RUN = r"\[\d{1,3}\](?:[ \t]*\[\d{1,3}\])*"
_AFTER_SENTENCE_END = re.compile(rf"(\S+?)([.!?])[ \t]*({_CITE_RUN})(?=\s|$)")
_ADJACENT_GROUPS = re.compile(r"\[\d{1,3}\](?:[ \t]*\[\d{1,3}\])+")
_ABBREVIATIONS = frozenset({"e.g", "i.e", "etc", "vs", "cf", "al", "approx", "no", "fig"})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _movable(word: str) -> bool:
    """False for a word that ends in a full stop of its own or is an abbreviation ("etc. [3]")."""
    return not (word.endswith(".") or word.lower().rstrip(".") in _ABBREVIATIONS)


def count_misplaced_citations(text: str) -> int:
    """Citations that sit after a sentence-ending mark and that `tidy_citations` would move before it. A citation
    that starts the next sentence and is followed by a comma ("... explains it. [8], pointing to ...") is not one."""
    return sum(1 for match in _AFTER_SENTENCE_END.finditer(text) if _movable(match.group(1)))


def _statement_before(text: str, end: int) -> str:
    """The statement a citation group at `end` belongs to: the text back to the previous sentence end."""
    return _SENTENCE_END.split(text[max(0, end - 1000):end])[-1]


def tidy_citations(
    text: str, cap: int | None = None, priority: Callable[[int, str], Any] | None = None
) -> str:
    """Move citation groups before the sentence's punctuation and merge neighbouring groups. Over `cap`, the
    group keeps the sources that `priority(n, statement)` ranks highest, the lower numbers first on a tie."""
    def move_before_punctuation(match: re.Match) -> str:
        word, mark, cites = match.groups()
        if not _movable(word):
            return match.group(0)
        return f"{word} {cites}{mark}"

    def merge(match: re.Match) -> str:
        numbers = sorted({int(n) for n in re.findall(r"\d+", match.group(0))})
        if cap and len(numbers) > cap:
            statement = _statement_before(match.string, match.start())
            rank = priority or (lambda _n, _statement: 0)
            # sorted() is stable with reverse=True too, so equal ranks keep the lower numbers first
            numbers = sorted(sorted(numbers, key=lambda n: rank(n, statement), reverse=True)[:cap])
        return "".join(f"[{n}]" for n in numbers)

    return _ADJACENT_GROUPS.sub(merge, _AFTER_SENTENCE_END.sub(move_before_punctuation, text))


def strip_sources_section(text: str) -> str:
    match = _SOURCES_HEADING_RE.search(text)
    return text[: match.start()].rstrip() if match else text


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).replace("[", "(").replace("]", ")").strip()


def source_line(source: dict) -> str:
    title = _clean_title(source.get("title") or "") or source["url"]
    url = source["url"]
    if url.startswith(("http://", "https://")):
        safe = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
        return f"[{title}]({safe})"
    return title


def sources_markdown(sources: list[dict], heading: str) -> str:
    if not sources:
        return ""
    lines = [f"## {heading}", ""]
    lines += [f"{source['n']}. {source_line(source)}" for source in sources]
    return "\n".join(lines)
