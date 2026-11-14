"""Search tools. Every tool returns a list of SourceDoc dicts and never raises: a failing source must not
take the whole interview down. The one exception is a Tavily failure that every later search repeats, used up
credits or a rejected key (`SearchUnavailable`), because then the user has to be told."""

import logging
import os
import re
import threading
import time
from functools import lru_cache
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from src.utils.languages import LANGUAGES
from src.utils.objects import SourceDoc
from src.utils.ranking import BLOCKED_DOMAINS, clean_title, rank_results

load_dotenv()

logger = logging.getLogger(__name__)

_user_email = os.getenv("WIKIPEDIA_USER_EMAIL", "").strip()
USER_AGENT = f"research-agent/0.2 ({_user_email})" if _user_email else "research-agent/0.2"
if not _user_email:
    logger.warning(
        "WIKIPEDIA_USER_EMAIL is not set: Wikipedia rate-limits anonymous clients much sooner, "
        "so Wikipedia searches may come back empty. Put your contact e-mail in .env and restart."
    )

WIKIPEDIA_MIN_INTERVAL = 0.25 if _user_email else 0.75
REQUEST_TIMEOUT = 15
TAVILY_CANDIDATE_FACTOR = 2
TAVILY_MAX_CANDIDATES = 20


class SearchUnavailable(Exception):
    """Tavily refused a search for a reason that every later search runs into too. `code` names it in warnings."""

    code = "search_unavailable"


class SearchCreditsExhausted(SearchUnavailable):
    """The plan's credits are used up (Tavily's 432 and 433 answers)."""

    code = "search_credits_exhausted"


class SearchKeyRejected(SearchUnavailable):
    """The API key is missing or invalid (Tavily's 401 answer)."""

    code = "search_key_rejected"


_UNAVAILABLE = [
    (re.compile(r"\bError (?:432|433)\b"), SearchCreditsExhausted),
    (re.compile(r"\bError 401\b"), SearchKeyRejected),
]
UNAVAILABLE_CODES = frozenset(unavailable.code for _, unavailable in _UNAVAILABLE)


class WikipediaRateLimited(Exception):
    def __init__(self, retry_after: float):
        super().__init__(f"Wikipedia rate limit hit, retry after {retry_after:.0f}s")
        self.retry_after = retry_after


class _Throttle:
    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = now + delay + self.interval
        if delay:
            time.sleep(delay)


_wikipedia_throttle = _Throttle(WIKIPEDIA_MIN_INTERVAL)
_wikipedia_blocked_until = 0.0


def _wikipedia_request(lang: str, params: dict) -> dict:
    global _wikipedia_blocked_until

    now = time.monotonic()
    if now < _wikipedia_blocked_until:
        raise WikipediaRateLimited(_wikipedia_blocked_until - now)

    _wikipedia_throttle.wait()
    response = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={"action": "query", "format": "json", "formatversion": 2, **params},
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 429:
        try:
            retry_after = float(response.headers["Retry-After"])
        except (KeyError, ValueError):
            retry_after = 30.0
        _wikipedia_blocked_until = time.monotonic() + retry_after
        raise WikipediaRateLimited(retry_after)

    response.raise_for_status()
    return response.json()


_SEARCH_STOPWORDS = frozenset(
    "and the for with how what why which from into about of in on to an "
    "ile ve bir bu şu için nasıl neden nedir gibi olan".split()
)


def _any_word_query(query: str) -> str:
    """Wikipedia search wants every word on the page, so a specific query ("LangGraph çok ajanlı sistem
    tasarımı") finds nothing. This is the relaxed version: pages that match any of the main words."""
    words = [w for w in dict.fromkeys(re.findall(r"\w{3,}", query.lower())) if w not in _SEARCH_STOPWORDS]
    return " OR ".join(words[:6]) if len(words) > 1 else ""


def _wikipedia_pages(lang: str, query: str, limit: int) -> list[dict]:
    """Search and fetch the article introductions in one request: anonymous clients are rate limited
    after only a handful of requests, so one request per search keeps Wikipedia usable."""
    data = _wikipedia_request(lang, {
        "generator": "search", "gsrsearch": query, "gsrlimit": limit,
        "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": "max", "inprop": "url",
    })
    pages = data.get("query", {}).get("pages", [])
    return sorted(pages, key=lambda page: page.get("index", 0))


def wikipedia_search(query: str, lang: str = "en", max_docs: int = 2, max_chars: int = 4000) -> list[SourceDoc]:
    lang = lang if lang in LANGUAGES else "en"
    try:
        pages = _wikipedia_pages(lang, query, max_docs)
        if not pages and (relaxed := _any_word_query(query)):
            pages = _wikipedia_pages(lang, relaxed, max_docs)

        docs: list[SourceDoc] = []
        for page in pages:
            text = (page.get("extract") or "").strip()
            title = page.get("title") or ""
            if not text or not title:
                continue
            docs.append({
                "url": page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                "title": title,
                "content": text[:max_chars],
                "kind": "wikipedia",
            })
        return docs
    except WikipediaRateLimited as exc:
        logger.warning("Skipping Wikipedia search for query %r: %s", query, exc)
    except Exception:
        logger.warning("Wikipedia search failed for query %r", query, exc_info=True)
    return []


@lru_cache(maxsize=8)
def _get_tavily(search_depth: str, max_results: int) -> TavilySearch:
    return TavilySearch(
        max_results=max_results, search_depth=search_depth, exclude_domains=list(BLOCKED_DOMAINS)
    )


def tavily_search(query: str, search_depth: str = "basic", max_results: int = 3) -> list[SourceDoc]:
    depth = search_depth if search_depth in {"basic", "advanced"} else "basic"
    limit = max(1, min(int(max_results), 10))
    try:
        data = _get_tavily(depth, min(limit * TAVILY_CANDIDATE_FACTOR, TAVILY_MAX_CANDIDATES)).invoke({"query": query})
    except Exception as exc:
        data = {"error": exc}
    # The tool returns most API failures as {"error": exception} instead of raising them.
    error = data.get("error") if isinstance(data, dict) else None
    if error is not None:
        cause = error if isinstance(error, BaseException) else None
        for pattern, unavailable in _UNAVAILABLE:
            if pattern.search(str(error)):
                raise unavailable(str(error)) from cause
        logger.warning("Tavily search failed for query %r: %s", query, error, exc_info=cause)
        return []

    results = data.get("results", []) if isinstance(data, dict) else []
    return [
        {
            "url": str(result["url"]),
            "title": clean_title(str(result.get("title") or result["url"]), str(result["url"])),
            "content": str(result["content"]),
            "kind": "web",
        }
        for result in rank_results(results, limit)
    ]
