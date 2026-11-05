import pytest
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda

from src import store as store_module
from src.security import RateLimiter
from src.usage import UsageTracker, add_usage, make_usage
from src.utils import nodes, tools
from src.utils.edges import route_after_question, route_messages


class ScriptedWriter:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages[-1].content)
        return AIMessage(content=self.texts.pop(0))


def _closing_state(sections):
    return {"topic": "Topic", "language": "tr", "merged_sections": sections}


def test_a_conclusion_without_citations_is_asked_for_once_more(monkeypatch):
    writer = ScriptedWriter("## Sonuç\n\nİddia.", "## Sonuç\n\nİddia [1].")
    monkeypatch.setattr(nodes, "get_llm", lambda: writer)
    result = nodes.write_conclusion(_closing_state(["## A\n\nVeri [1]."]))
    assert result == {"conclusion": "## Sonuç\n\nİddia [1]."}
    assert len(writer.requests) == 2 and "citation marker" in writer.requests[1]


def test_a_conclusion_that_stays_uncited_keeps_the_first_answer(monkeypatch):
    writer = ScriptedWriter("## Sonuç\n\nBir.", "## Sonuç\n\nİki.")
    monkeypatch.setattr(nodes, "get_llm", lambda: writer)
    result = nodes.write_conclusion(_closing_state(["## A\n\nVeri [1]."]))
    assert result == {"conclusion": "## Sonuç\n\nBir."}
    assert len(writer.requests) == 2


def test_cited_or_uncitable_closing_sections_are_written_in_one_call(monkeypatch):
    writer = ScriptedWriter("## Giriş\n\nİddia [1].")
    monkeypatch.setattr(nodes, "get_llm", lambda: writer)
    assert nodes.write_introduction(_closing_state(["## A\n\nVeri [1]."])) == {"introduction": "## Giriş\n\nİddia [1]."}
    assert len(writer.requests) == 1

    writer = ScriptedWriter("## Giriş\n\nİddia.")
    monkeypatch.setattr(nodes, "get_llm", lambda: writer)
    nodes.write_introduction(_closing_state(["## A\n\nKaynak bulunamadı."]))
    assert len(writer.requests) == 1


def test_the_report_prompt_asks_for_theme_sub_headings():
    from src.utils.prompts import report_writer_instructions

    assert "### sub-headings" in report_writer_instructions
    assert "no sub-heading" not in report_writer_instructions


def test_tavily_search_maps_results_normalizes_options_and_survives_errors(monkeypatch):
    seen = []

    class Fake:
        def invoke(self, payload):
            return {"results": [
                {"url": "https://a", "title": "A", "content": "text"},
                {"url": "", "content": "no url"},
                {"url": "https://b", "content": ""},
            ]}

    monkeypatch.setattr(tools, "_get_tavily", lambda depth, n: seen.append((depth, n)) or Fake())
    docs = tools.tavily_search("q", search_depth="weird", max_results=99)
    assert docs == [{"url": "https://a", "title": "A", "content": "text", "kind": "web"}]
    assert seen == [("basic", 20)]

    class Broken:
        def invoke(self, payload):
            raise RuntimeError("quota exceeded")

    monkeypatch.setattr(tools, "_get_tavily", lambda depth, n: Broken())
    assert tools.tavily_search("q") == []


def test_tavily_search_over_fetches_and_keeps_the_best_sources(monkeypatch):
    seen = []

    class Fake:
        def invoke(self, payload):
            return {"results": [
                {"url": "https://www.youtube.com/watch?v=1", "title": "Video", "content": "x", "score": 0.99},
                {"url": "https://medium.com/@a/post", "title": "Blog", "content": "x", "score": 0.9},
                {"url": "https://example.com/news", "title": "News", "content": "x", "score": 0.7},
                {"url": "https://docs.example.org/guide", "title": "Docs", "content": "x", "score": 0.6},
                {"url": "https://other.com/a", "title": "Other", "content": "x", "score": 0.3},
            ]}

    monkeypatch.setattr(tools, "_get_tavily", lambda depth, n: seen.append((depth, n)) or Fake())
    docs = tools.tavily_search("q", search_depth="advanced", max_results=3)

    assert seen == [("advanced", 6)]
    assert [d["title"] for d in docs] == ["Blog", "Docs", "News"]
    assert all(d["kind"] == "web" for d in docs)


class StubPicker:
    def __init__(self, best=None, error=None):
        self.best = best
        self.error = error

    def with_structured_output(self, schema, **kwargs):
        def run(messages):
            if self.error:
                raise self.error
            return schema(best=self.best)

        return RunnableLambda(run)


def docs_list(n):
    return [{"url": f"https://site{i}.org/a", "title": f"Page {i}", "content": "Text.", "kind": "web"} for i in range(1, n + 1)]


def test_curation_keeps_only_the_models_picks_in_its_order(monkeypatch):
    monkeypatch.setattr(nodes, "get_fast_llm", lambda: StubPicker(best=[3, 99, 3, 0, 1]))
    chosen = nodes.curate_sources(docs_list(7), "query", 3)
    assert [d["title"] for d in chosen] == ["Page 3", "Page 1"]


def test_curation_uses_the_ranking_when_the_model_returns_nothing_usable(monkeypatch):
    monkeypatch.setattr(nodes, "get_fast_llm", lambda: StubPicker(best=[0, 99]))
    chosen = nodes.curate_sources(docs_list(7), "query", 3)
    assert [d["title"] for d in chosen] == ["Page 1", "Page 2", "Page 3"]


def test_curation_skips_the_model_when_there_is_nothing_to_choose(monkeypatch):
    def forbidden():
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(nodes, "get_fast_llm", forbidden)
    docs = docs_list(4)
    assert nodes.curate_sources(docs, "query", 5) == docs


def test_curation_falls_back_to_the_ranking_when_the_model_fails(monkeypatch):
    monkeypatch.setattr(nodes, "get_fast_llm", lambda: StubPicker(error=RuntimeError("down")))
    chosen = nodes.curate_sources(docs_list(8), "query", 5)
    assert [d["title"] for d in chosen] == [f"Page {i}" for i in range(1, 6)]


def test_curation_prompt_lists_candidates_with_their_site(monkeypatch):
    seen = []

    class Spy(StubPicker):
        def with_structured_output(self, schema, **kwargs):
            def run(messages):
                seen.append(messages[0].content)
                return schema(best=[1])

            return RunnableLambda(run)

    monkeypatch.setattr(nodes, "get_fast_llm", lambda: Spy())
    nodes.curate_sources(docs_list(6), "langgraph interrupts", 2)
    assert "langgraph interrupts" in seen[0] and "[2] Page 2 — site2.org" in seen[0] and "Pick up to 2 candidates" in seen[0]


def test_tavily_titles_lose_their_site_names(monkeypatch):
    class Fake:
        def invoke(self, payload):
            return {"results": [{
                "url": "https://docs.langchain.com/oss/python/langgraph/interrupts",
                "title": "Interrupts - Docs by LangChain", "content": "text", "score": 0.9,
            }]}

    monkeypatch.setattr(tools, "_get_tavily", lambda depth, n: Fake())
    assert [d["title"] for d in tools.tavily_search("q")] == ["Interrupts"]


def test_finalized_report_has_tidy_citations():
    sources = [{"n": n, "url": f"https://s{n}.org", "title": f"S{n}"} for n in (1, 2, 3)]
    out = nodes.finalize_report({
        "language": "en",
        "sources": sources,
        "introduction": "# T\n\n## Introduction\n\nFirst claim. [1][2] Second claim. [3]",
        "content": "## Insights\n\nA point. [2][3] [1] Another. [3]",
        "conclusion": "## Conclusion\n\nWrap up. [1][2][3]",
    })
    report = out["final_report"]
    assert "First claim [1][2]. Second claim [3]." in report
    assert "A point [1][2][3]. Another [3]." in report
    assert "Wrap up [1][2][3]." in report
    assert [s["n"] for s in out["final_sources"]] == [1, 2, 3]


def test_finalized_report_never_cites_more_than_three_sources_for_one_statement():
    sources = [
        {"n": 1, "url": "https://blog-a.com/p", "title": "A"},
        {"n": 2, "url": "https://docs.example.org/guide", "title": "B"},
        {"n": 3, "url": "https://blog-c.com/p", "title": "C"},
        {"n": 4, "url": "https://cs.stanford.edu/paper", "title": "D"},
        {"n": 5, "url": "https://nature.com/x", "title": "E"},
    ]
    out = nodes.finalize_report({
        "language": "en",
        "sources": sources,
        "introduction": "# T\n\n## Introduction\n\nIntro claim. [1]",
        "content": "## Insights\n\nBig claim. [1][2][3][4][5]",
        "conclusion": "## Conclusion\n\nEnd. [2]",
    })
    report = out["final_report"]
    assert "Big claim [2][3][4]." in report
    assert [s["url"] for s in out["final_sources"]] == [
        "https://blog-a.com/p",
        "https://docs.example.org/guide",
        "https://cs.stanford.edu/paper",
        "https://nature.com/x",
    ]


def test_tavily_is_told_to_skip_social_and_video_sites(monkeypatch):
    captured = {}

    class Capture:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    tools._get_tavily.cache_clear()
    monkeypatch.setattr(tools, "TavilySearch", Capture)
    tools._get_tavily("advanced", 10)
    tools._get_tavily.cache_clear()

    assert captured["search_depth"] == "advanced" and captured["max_results"] == 10
    assert {"youtube.com", "linkedin.com", "reddit.com"} <= set(captured["exclude_domains"])


def test_search_tools_never_raise(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(tools, "_wikipedia_request", boom)
    assert tools.wikipedia_search("x") == []


def test_wikipedia_search_builds_documents_per_language(monkeypatch):
    calls = []

    def fake_request(lang, params):
        calls.append(lang)
        return {"query": {"pages": [
            {"title": "Second", "index": 2, "extract": "Other.", "fullurl": "https://tr.wikipedia.org/wiki/Second"},
            {"title": "Kentsel orman", "index": 1, "extract": "Metin " * 1000,
             "fullurl": "https://tr.wikipedia.org/wiki/Kentsel_orman"},
            {"title": "No text", "index": 3, "extract": ""},
        ]}}

    monkeypatch.setattr(tools, "_wikipedia_request", fake_request)
    docs = tools.wikipedia_search("kentsel orman", lang="tr", max_docs=3)
    assert [d["title"] for d in docs] == ["Kentsel orman", "Second"]
    assert docs[0]["url"] == "https://tr.wikipedia.org/wiki/Kentsel_orman" and docs[0]["kind"] == "wikipedia"
    assert len(docs[0]["content"]) == 4000
    assert calls == ["tr"]


def test_wikipedia_retries_with_any_of_the_words_when_nothing_matches_them_all(monkeypatch):
    searches = []

    def fake_request(lang, params):
        searches.append(params["gsrsearch"])
        if " OR " not in params["gsrsearch"]:
            return {"query": {"pages": []}}
        return {"query": {"pages": [{"title": "Ajan tabanlı modelleme", "index": 1, "extract": "Metin."}]}}

    monkeypatch.setattr(tools, "_wikipedia_request", fake_request)
    docs = tools.wikipedia_search("LangGraph ile çok ajanlı sistemler nasıl tasarlanır", lang="tr")

    assert [d["title"] for d in docs] == ["Ajan tabanlı modelleme"]
    assert docs[0]["url"] == "https://tr.wikipedia.org/wiki/Ajan_tabanl%C4%B1_modelleme"
    assert searches == [
        "LangGraph ile çok ajanlı sistemler nasıl tasarlanır",
        "langgraph OR çok OR ajanlı OR sistemler OR tasarlanır",
    ]


def test_wikipedia_does_not_relax_a_single_word_query(monkeypatch):
    searches = []
    monkeypatch.setattr(tools, "_wikipedia_request", lambda lang, params: searches.append(params["gsrsearch"]) or {})
    assert tools.wikipedia_search("LangGraph") == []
    assert searches == ["LangGraph"]


def test_wikipedia_rate_limit_is_skipped_quietly(monkeypatch):
    def limited(lang, params):
        raise tools.WikipediaRateLimited(30)

    monkeypatch.setattr(tools, "_wikipedia_request", limited)
    assert tools.wikipedia_search("anything") == []


def test_wikipedia_unknown_language_falls_back_to_english(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_wikipedia_request", lambda lang, params: seen.append(lang) or {})
    tools.wikipedia_search("x", lang="zz")
    assert seen == ["en"]


def test_store_round_trips_runs_and_marks_interrupted(tmp_path):
    db = store_module.Store(tmp_path / "t.db")
    db.create_run("a" * 32, "Topic", "tr", {"depth": "quick"})
    db.update_run("a" * 32, analysts=[{"name": "X"}], usage={"calls": 2})
    run = db.get_run("a" * 32)
    assert run["settings"] == {"depth": "quick"} and run["analysts"] == [{"name": "X"}] and run["report"] is None

    assert db.mark_interrupted() == 1
    assert db.get_run("a" * 32)["status"] == "interrupted"
    assert db.mark_interrupted() == 0
    with pytest.raises(ValueError):
        db.update_run("a" * 32, nope=1)

    db.update_run("a" * 32, report="# R")
    assert db.list_runs()[0]["has_report"] is True
    assert db.delete_run("a" * 32) and not db.delete_run("a" * 32)
    db.close()


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(2, window=100, clock=lambda: now[0])
    assert limiter.check("ip") is None and limiter.check("ip") is None
    wait = limiter.check("ip")
    assert wait is not None and 99 <= wait <= 100
    assert limiter.check("other") is None
    now[0] = 101
    assert limiter.check("ip") is None
    assert RateLimiter(0).check("ip") is None


def test_usage_tracker_sums_calls_and_prices_them():
    tracker = UsageTracker(price_input_per_1m=1.0, price_output_per_1m=2.0)
    message = AIMessage(content="x", usage_metadata={"input_tokens": 1_000_000, "output_tokens": 500_000, "total_tokens": 1_500_000})
    tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    tracker.on_llm_end(LLMResult(generations=[[]], llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
    snap = tracker.snapshot(elapsed=3.14)
    assert (snap["input_tokens"], snap["output_tokens"], snap["calls"]) == (1_000_010, 500_005, 2)
    assert snap["cost"] == 2.0 and snap["elapsed"] == 3.1
    assert make_usage(10, 5, 1, 1.0)["cost"] is None


def test_add_usage_accumulates_segments():
    total = add_usage({"input_tokens": 5, "output_tokens": 1, "calls": 1, "elapsed": 2.0}, make_usage(10, 4, 2, 1.5))
    assert (total["input_tokens"], total["output_tokens"], total["calls"], total["elapsed"]) == (15, 5, 3, 3.5)


def expert(text="a"):
    return AIMessage(content=text, name="expert")


def test_route_after_question_needs_an_answer_before_it_will_stop():
    closing = AIMessage(content="Great. Thank you so much for your help!")
    opts = {"options": {"language": "en"}}
    assert route_after_question({**opts, "messages": [HumanMessage(content="hi"), closing]}) == "generate_search_query"
    assert route_after_question({**opts, "messages": [HumanMessage(content="hi"), expert(), closing]}) == "save_interview"
    plain = AIMessage(content="One more question?")
    assert route_after_question({**opts, "messages": [HumanMessage(content="hi"), expert(), plain]}) == "generate_search_query"


def test_route_after_question_recognizes_the_localized_closing():
    turkish = AIMessage(content="Harika. Yardımınız için çok teşekkür ederim")
    state = {"options": {"language": "tr"}, "messages": [HumanMessage(content="hi"), expert(), turkish]}
    assert route_after_question(state) == "save_interview"


def test_route_messages_counts_expert_answers_only():
    messages = [HumanMessage(content="hi"), AIMessage(content="q"), expert(), AIMessage(content="q2")]
    assert route_messages({"messages": messages, "options": {"max_num_turns": 2}}) == "ask_question"
    assert route_messages({"messages": messages + [expert()], "options": {"max_num_turns": 2}}) == "save_interview"
    assert route_messages({"messages": messages}) == "ask_question"


def test_friendly_error_matches_subclasses_of_the_openai_errors():
    from src.runner import friendly_error

    class AuthenticationError(Exception): ...

    class OpenAIAuthenticationError(AuthenticationError): ...

    class RateLimitError(Exception): ...

    class OpenAITimeoutError(TimeoutError): ...

    class APITimeoutError(Exception): ...

    class OpenAITimeout2(APITimeoutError): ...

    secret = "Error code: 401 - key sk-secret-value has expired"
    result = friendly_error(OpenAIAuthenticationError(secret))
    assert result["code"] == "auth" and "sk-secret" not in result["message"]
    assert friendly_error(RateLimitError("quota"))["code"] == "rate_limit"
    assert friendly_error(OpenAITimeout2("slow"))["code"] == "timeout"
    assert friendly_error(ValueError("boom"))["code"] == "unknown"
