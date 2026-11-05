import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("playwright")

import fakes  # noqa: E402
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from src.api import create_app  # noqa: E402
from src.settings import get_settings  # noqa: E402


def wait_until(predicate, timeout: float = 30.0, step: float = 0.1) -> None:
    """Poll from Python: the page's CSP forbids the eval that Playwright's wait_for_function needs."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            if predicate():
                return
        except Exception:
            pass
        time.sleep(step)
    raise TimeoutError("condition not met in time")


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setenv("OPENAI_API_KEY", "test-openai")
    patch.setenv("TAVILY_API_KEY", "test-tavily")
    patch.setenv("DATA_DIR", str(tmp_path_factory.mktemp("e2e-data")))
    patch.setenv("RATE_LIMIT_PER_HOUR", "1000")
    patch.delenv("APP_API_KEY", raising=False)

    model = fakes.make_llm(delay=0.1)
    patch.setattr("src.utils.nodes.get_llm", lambda: model)
    patch.setattr("src.utils.nodes.get_fast_llm", lambda: model)
    patch.setattr("src.utils.nodes.tavily_search", fakes.fake_tavily)
    patch.setattr("src.utils.nodes.wikipedia_search", fakes.fake_wikipedia)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(create_app(get_settings()), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_until(lambda: server.started, timeout=20)

    yield SimpleNamespace(url=f"http://127.0.0.1:{port}", model=model)

    server.should_exit = True
    thread.join(timeout=10)
    patch.undo()


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch(channel=os.getenv("E2E_BROWSER_CHANNEL") or None)
        except Exception as exc:
            pytest.skip(f"No usable browser for the UI tests: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def new_page(browser, live_server):
    contexts = []

    def make(ui_language: str = "en"):
        context = browser.new_context(viewport={"width": 1180, "height": 900}, accept_downloads=True)
        context.add_init_script(f"localStorage.setItem('ra-ui-lang', '{ui_language}')")
        contexts.append(context)
        page = context.new_page()
        page.problems = []
        page.on("console", lambda m: page.problems.append(m.text)
                if m.type in ("error", "warning") and "Failed to load resource" not in m.text else None)
        page.on("pageerror", lambda e: page.problems.append(str(e)))
        return page

    yield make
    for context in contexts:
        context.close()


@pytest.fixture
def page(new_page):
    return new_page()
