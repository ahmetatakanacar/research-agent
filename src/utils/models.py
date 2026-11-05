from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.settings import get_settings

load_dotenv()


@lru_cache(maxsize=4)
def _chat(model: str, timeout: int, max_retries: int) -> ChatOpenAI:
    return ChatOpenAI(model=model, timeout=timeout, max_retries=max_retries)


def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return _chat(settings.openai_model, settings.llm_timeout, settings.llm_max_retries)


def get_fast_llm() -> ChatOpenAI:
    settings = get_settings()
    return _chat(settings.fast_model, settings.llm_timeout, settings.llm_max_retries)
