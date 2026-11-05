import re
from urllib.parse import urlparse

BLOCKED_DOMAINS = (
    "youtube.com", "youtu.be", "linkedin.com", "facebook.com", "instagram.com", "tiktok.com",
    "x.com", "twitter.com", "pinterest.com", "quora.com", "reddit.com", "vk.com",
    "slideshare.net", "scribd.com", "docs.google.com", "eksisozluk.com",
    "aka.ms", "bit.ly", "tinyurl.com", "t.co", "goo.gl", "lnkd.in", "ow.ly", "buff.ly",
)
REDIRECT_PATHS = ("/fwlink",)
AUTHORITY_DOMAINS = (
    "arxiv.org", "doi.org", "nature.com", "science.org", "sciencedirect.com", "springer.com",
    "ieee.org", "acm.org", "jstor.org", "oecd.org", "worldbank.org", "un.org", "imf.org",
    "europa.eu", "w3.org", "ietf.org", "iso.org", "readthedocs.io", "britannica.com",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
)
AUTHORITY_LABELS = frozenset({"gov", "edu", "ac", "mil", "int"})
DOCS_SUBDOMAINS = frozenset({"docs", "developer", "developers", "documentation", "learn"})
DOCS_PATHS = ("/docs", "/documentation")

GENERIC_TITLE_TAILS = frozenset({"articles", "article", "blog", "docs", "documentation", "news", "home", "resources"})
TAGLINE_WORDS = frozenset({"leader", "speaker", "advisor", "consultant", "founder"})
TITLE_DASHES = (" — ", " – ", " - ", " · ")
TITLE_ACRONYMS = frozenset({"ai", "llm", "llms", "api", "apis", "ui", "ux", "ml", "sql", "aws", "gpu", "cpu", "sdk", "mcp", "a2a", "rag"})
MIN_TITLE_LENGTH = 8
MIN_TRIMMED_WORDS = 3
MIN_HOST_LETTERS = 6
MAX_TITLE_LENGTH = 160
SPACE_AFTER_COMMA = re.compile(r"([,;])(?=[^\W\d_])")
PDF_MARKER = re.compile(r"^[(\[]\s*pdf\s*[)\]]\s*", re.IGNORECASE)
CUT_OFF = re.compile(r"(?:\s+\.{3}|\s*…)$")
TRAILING_PERIOD = re.compile(r"(?<=[^\W\d_]{4})\.$")
DANGLING_WORDS = frozenset({
    "a", "an", "and", "or", "of", "the", "in", "on", "for", "to", "with", "by", "at", "from", "vs", "&",
    "via", "using", "through", "into", "as", "than", "is", "are", "randomized", "randomised", "controlled",
    "ve", "ile", "için", "bir", "bu", "de", "da", "ki",
})

LOW_TRUST_DOMAINS = ("cloudns.asia", "cloudns.net", "duckdns.org", "no-ip.org", "ddns.net", "dynu.net")
FREE_TLDS = frozenset({"tk", "ml", "ga", "cf", "gq"})
STANDARD_PORTS = (None, 80, 443)
MARKETING_PATHS = ("/services", "/solutions", "/pricing", "/hire", "/consulting", "/agency", "/company", "/contact")
MARKETING_TITLE = re.compile(
    r"development (?:company|services|agency)|consulting|outsourc|hire (?:dedicated|expert|top)|\bagency\b", re.IGNORECASE
)
CATALOG_DOMAINS = ("coursera.org", "udemy.com", "udacity.com", "skillshare.com", "oboe.com")
MARKETING_HOST = re.compile(r"consulting|consultancy|agency|outsourc")
IP_ADDRESS = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

AUTHORITY_BONUS = 0.25
MARKETING_PENALTY = 0.4
INVALID_URL_PENALTY = 0.25
DEFAULT_RELEVANCE = 0.5
AUTHORITY_MIN_RELEVANCE = 0.45
MIN_RELEVANCE = 0.3


def host_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def is_blocked(url: str) -> bool:
    host = host_of(url)
    try:
        port = urlparse(url).port
    except ValueError:
        return True
    return (
        _matches(host, BLOCKED_DOMAINS)
        or _matches(host, LOW_TRUST_DOMAINS)
        or host.rsplit(".", 1)[-1] in FREE_TLDS
        or bool(IP_ADDRESS.match(host))
        or port not in STANDARD_PORTS
        or urlparse(url).path.lower().startswith(REDIRECT_PATHS)
    )


def quality_bonus(url: str) -> float:
    host = host_of(url)
    if not host:
        return -INVALID_URL_PENALTY
    labels = host.split(".")
    path = urlparse(url).path.lower()
    if (
        AUTHORITY_LABELS & set(labels)
        or _matches(host, AUTHORITY_DOMAINS)
        or labels[0] in DOCS_SUBDOMAINS
        or path.startswith(DOCS_PATHS)
    ):
        return AUTHORITY_BONUS
    return 0.0


def _relevance(result: dict) -> float:
    try:
        return float(result.get("score"))
    except (TypeError, ValueError):
        return DEFAULT_RELEVANCE


def _marketing_penalty(result: dict) -> float:
    url = str(result["url"])
    path = urlparse(url).path.lower()
    if (
        path.startswith(MARKETING_PATHS)
        or _matches(host_of(url), CATALOG_DOMAINS)
        or MARKETING_HOST.search(host_of(url))
        or MARKETING_TITLE.search(str(result.get("title") or ""))
    ):
        return MARKETING_PENALTY
    return 0.0


def _value(result: dict) -> float:
    relevance = _relevance(result)
    bonus = quality_bonus(str(result["url"]))
    if bonus > 0 and relevance < AUTHORITY_MIN_RELEVANCE:
        bonus = 0.0
    return relevance + bonus - _marketing_penalty(result)


def rank_results(results: list[dict], limit: int, per_host: int = 2) -> list[dict]:
    candidates = [r for r in results if r.get("url") and r.get("content") and not is_blocked(str(r["url"]))]
    relevant = [r for r in candidates if _relevance(r) >= MIN_RELEVANCE]
    ranked = sorted(relevant or candidates, key=_value, reverse=True)

    chosen: list[dict] = []
    per_site: dict[str, int] = {}
    for result in ranked:
        host = host_of(str(result["url"]))
        if per_site.get(host, 0) >= per_host:
            continue
        per_site[host] = per_site.get(host, 0) + 1
        chosen.append(result)
        if len(chosen) == limit:
            return chosen

    for result in ranked:
        if len(chosen) == limit:
            break
        if result not in chosen:
            chosen.append(result)
    return chosen


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_site_tail(tail: str, host_labels: list[str], host_letters: str = "") -> bool:
    words = {word.strip(",.").lower() for word in tail.split()}
    tail_letters = _letters(tail)
    return (
        tail.strip().lower() in GENERIC_TITLE_TAILS
        or bool(words & TAGLINE_WORDS)
        or any(label in tail_letters for label in host_labels)
        or (len(host_letters) >= MIN_HOST_LETTERS and host_letters in tail_letters)
    )


def _slug_words(url: str) -> list[str]:
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    words = [word for word in re.split(r"[-_]+", re.sub(r"\.(?:html?|php|aspx?)$", "", slug)) if word]
    while words and re.fullmatch(r"[0-9a-f]{8,}", words[-1]):
        words.pop()
    return words


def _complete_from_words(prefix: str, words: list[str]) -> str | None:
    wanted = _letters(prefix)
    spelled = ""
    for count, word in enumerate(words, 1):
        spelled += _letters(word)
        if spelled == wanted:
            rest = [w.upper() if w.lower() in TITLE_ACRONYMS else w.capitalize() for w in words[count:]]
            return " ".join([prefix, *rest]) if rest else prefix
        if len(spelled) > len(wanted):
            break
    return None


def _complete_truncated(title: str, url: str) -> str:
    if not title.endswith(("...", "…")):
        return title
    prefix = title.rstrip(".… ").rstrip()
    words = _slug_words(url)
    without_id = words[1:] if words and words[0].isdigit() else []
    for candidate in (words, without_id):
        completed = _complete_from_words(prefix, candidate)
        if completed:
            return completed
    return title


def _trim_cut_off(title: str) -> str:
    if not CUT_OFF.search(title):
        return title
    words = CUT_OFF.sub("", title).split()
    while words and words[-1].lower().strip(",:;-–—") in DANGLING_WORDS:
        words.pop()
    if len(words) < MIN_TRIMMED_WORDS:
        return title
    return " ".join(words).rstrip(",:;-–— ")


def _shorten(title: str) -> str:
    if len(title) <= MAX_TITLE_LENGTH:
        return title
    head = title[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0].rstrip(" :;,.-–—")
    return head + "…"


def clean_title(title: str, url: str) -> str:
    cleaned = SPACE_AFTER_COMMA.sub(r"\1 ", " ".join(title.split()))
    unmarked = PDF_MARKER.sub("", cleaned)
    if len(unmarked) >= MIN_TITLE_LENGTH:
        cleaned = unmarked
    unprefixed = re.sub(r"^GitHub\s*[-–—:]\s*", "", cleaned)
    if len(unprefixed) >= MIN_TITLE_LENGTH:
        cleaned = unprefixed
    if " | " in cleaned:
        head = cleaned.split(" | ")[0].strip()
        if len(head) >= MIN_TITLE_LENGTH:
            cleaned = head

    host = host_of(url)
    host_labels = [label for label in host.split(".")[:-1] if len(label) >= 4]
    host_letters = _letters(host)
    stripped = True
    while stripped:
        stripped = False
        for dash in TITLE_DASHES:
            head, found, tail = cleaned.rpartition(dash)
            if not found or len(head.strip()) < MIN_TITLE_LENGTH:
                continue
            repeated = tail.strip().lower() == head.rpartition(dash)[2].strip().lower()
            if repeated or _is_site_tail(tail, host_labels, host_letters):
                cleaned = head.strip()
                stripped = True
                break
    completed = _trim_cut_off(_complete_truncated(cleaned, url)).rstrip(" :;,")
    return _shorten(TRAILING_PERIOD.sub("", completed))
