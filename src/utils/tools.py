import logging
from functools import lru_cache
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
import wikipedia

load_dotenv()

logger = logging.getLogger(__name__)

wikipedia.set_user_agent("research-agent/0.1")

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
    except Exception:
        logger.warning("Wikipedia search failed for query %r", query, exc_info=True)
        return ""

    return "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
