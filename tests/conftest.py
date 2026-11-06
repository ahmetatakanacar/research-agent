import os
import sys
from pathlib import Path

import pytest

# The tests must never send traces to a real LangSmith account (a key in .env would otherwise be picked up
# when the app is imported). Set before anything imports `src`: load_dotenv does not overwrite these.
for _name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"):
    os.environ[_name] = "false"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakes  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Every test gets its own data dir and dummy keys, and never sees real ones."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("APP_API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "1000")


@pytest.fixture
def llm(monkeypatch):
    model = fakes.make_llm()
    monkeypatch.setattr("src.utils.nodes.get_llm", lambda: model)
    monkeypatch.setattr("src.utils.nodes.get_fast_llm", lambda: model)
    monkeypatch.setattr("src.utils.nodes.tavily_search", fakes.fake_tavily)
    monkeypatch.setattr("src.utils.nodes.wikipedia_search", fakes.fake_wikipedia)
    return model
