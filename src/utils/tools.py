import logging
import os
import threading
import time
from functools import lru_cache
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
import requests
import wikipedia
from wikipedia import wikipedia as wikipedia_client

load_dotenv()

logger = logging.getLogger(__name__)

# Wikimedia asks API clients to put contact details in the User-Agent and
# throttles anonymous ones harder. Set WIKIPEDIA_USER_EMAIL in .env to your own address.
_user_email = os.getenv("WIKIPEDIA_USER_EMAIL", "").strip()
if _user_email:
    wikipedia.set_user_agent(f"research-agent/0.1 ({_user_email})")
else:
    wikipedia.set_user_agent("research-agent/0.1")
    logger.warning("WIKIPEDIA_USER_EMAIL is not set; Wikipedia may rate limit anonymous requests sooner")

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_MIN_INTERVAL = 0.25

class WikipediaRateLimited(Exception):
    def __init__(self, retry_after: float):
        super().__init__(f"Wikipedia rate limit hit, retry after {retry_after:.0f}s")
        self.retry_after = retry_after

_wikipedia_lock = threading.Lock()
_wikipedia_next_request = 0.0
_wikipedia_blocked_until = 0.0

def _wikipedia_request(params: dict) -> dict:

    global _wikipedia_next_request, _wikipedia_blocked_until

    params["format"] = "json"
    params.setdefault("action", "query")

    with _wikipedia_lock:
        now = time.monotonic()
        if now < _wikipedia_blocked_until:
            raise WikipediaRateLimited(_wikipedia_blocked_until - now)

        time.sleep(max(0.0, _wikipedia_next_request - now))
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers={"User-Agent": wikipedia_client.USER_AGENT},
            timeout=15,
        )
        _wikipedia_next_request = time.monotonic() + WIKIPEDIA_MIN_INTERVAL

        if response.status_code == 429:
            try:
                retry_after = float(response.headers["Retry-After"])
            except (KeyError, ValueError):
                retry_after = 30.0
            _wikipedia_blocked_until = time.monotonic() + retry_after
            raise WikipediaRateLimited(retry_after)

    response.raise_for_status()
    return response.json()

wikipedia_client._wiki_request = _wikipedia_request

@lru_cache(maxsize=1)
def _get_tavily() -> TavilySearch:
    return TavilySearch(max_results=3)

def tavily_search(query: str) -> str:
    
    try:
        data = _get_tavily().invoke({"query": query})
    except Exception:
        logger.warning("Tavily search failed for query %r", query, exc_info=True)
        return ""

    search_docs = data.get("results", []) if isinstance(data, dict) else []

    return "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

def wikipedia_search(query: str) -> str:
   
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    except WikipediaRateLimited as exc:
        logger.warning("Skipping Wikipedia search for query %r: %s", query, exc)
        return ""
    except Exception:
        logger.warning("Wikipedia search failed for query %r", query, exc_info=True)
        return ""

    return "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
