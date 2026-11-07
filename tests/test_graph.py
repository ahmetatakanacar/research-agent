import fakes
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from src.agent import builder

BASE = {"topic": "urban forests", "max_analyst": 2, "language": "en", "max_num_turns": 2}


def make_graph():
    saver = MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[("src.utils.objects", "Analyst")]))
    return builder.compile(checkpointer=saver)


def start(graph, extra=None, thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke({**BASE, **(extra or {})}, config)
    assert "__interrupt__" in result, "the run must pause for team approval"
    return config, result


def approve(graph, config, resume=""):
    return graph.invoke(Command(resume=resume), config)


def test_full_run_builds_numbered_sources_in_code(llm):
    graph = make_graph()
    config, paused = start(graph)
    assert [a.name for a in paused["analysts"]] == ["Analyst 1", "Analyst 2"]

    final = approve(graph, config)
    report = final["final_report"]

    assert report.startswith("# Big Title")
    assert "## Introduction" in report and "## Insights" in report and "## Conclusion" in report
    assert "## Sources" in report

    sources = final["final_sources"]
    numbers = [s["n"] for s in sources]
    assert numbers == list(range(1, len(numbers) + 1))
    from src.utils.sources import cited_numbers

    assert cited_numbers(report.split("## Sources")[0]) == set(numbers)
    assert "[9]" not in report
    assert sum(s["url"] == "https://example.com/shared" for s in sources) <= 1


def test_each_turn_generates_one_search_query(llm):
    graph = make_graph()
    config, _ = start(graph)
    approve(graph, config)
    assert llm.calls.count("query") == 4
    assert llm.calls.count("answer") == 4


def test_regenerate_uses_feedback_then_approve(llm):
    graph = make_graph()
    config, _ = start(graph)

    paused = approve(graph, config, resume="add an economist")
    assert "__interrupt__" in paused
    assert all("revised: add an economist" in a.description for a in paused["analysts"])

    final = approve(graph, config)
    assert final["final_report"]


def test_user_edits_replace_the_team(llm):
    graph = make_graph()
    config, _ = start(graph)

    edited = [{"name": "Dr. Edit", "role": "Economist", "affiliation": "Bank", "description": "Costs"}]
    final = approve(graph, config, resume={"feedback": "", "analysts": edited})

    assert [a.name for a in final["analysts"]] == ["Dr. Edit"]
    assert len(final["sections"]) == 1


def test_every_question_searches_web_and_wikipedia_with_one_query(llm, monkeypatch):
    queries = {"web": [], "wikipedia": []}

    def web(query, *args, **kwargs):
        queries["web"].append(query)
        return fakes.fake_tavily(query)

    def wiki(query, *args, **kwargs):
        queries["wikipedia"].append(query)
        return fakes.fake_wikipedia(query)

    monkeypatch.setattr("src.utils.nodes.tavily_search", web)
    monkeypatch.setattr("src.utils.nodes.wikipedia_search", wiki)
    graph = make_graph()
    config, _ = start(graph, {"max_analyst": 1})
    final = approve(graph, config)

    assert queries["web"] and queries["web"] == queries["wikipedia"]
    assert llm.calls.count("query") == len(queries["web"])
    retrieved = {doc["kind"] for section in final["sections"] for doc in section["sources"]}
    assert retrieved == {"web", "wikipedia"}


def test_web_candidates_are_narrowed_to_the_best_five_each_turn(llm, monkeypatch):
    requested = []

    def web(query, search_depth="basic", max_results=3):
        requested.append(max_results)
        return [
            {"url": f"https://site{i}.org/{len(requested)}", "title": f"Page {i}", "content": "Text.", "kind": "web"}
            for i in range(1, 9)
        ]

    monkeypatch.setattr("src.utils.nodes.tavily_search", web)
    graph = make_graph()
    config, _ = start(graph, {"max_analyst": 1})
    final = approve(graph, config)

    assert requested == [10, 10]
    assert llm.calls.count("select") == 2
    web_docs = [d for d in final["sections"][0]["sources"] if d["kind"] == "web"]
    assert len(web_docs) == 10  # five per turn, two turns


def test_a_non_english_report_also_searches_the_web_in_english(llm, monkeypatch):
    queries = []

    def web(query, search_depth="basic", max_results=3):
        queries.append(query)
        return fakes.fake_tavily(query)

    monkeypatch.setattr("src.utils.nodes.tavily_search", web)
    graph = make_graph()
    config, _ = start(graph, {"max_analyst": 1, "language": "tr", "max_num_turns": 1})
    approve(graph, config)

    assert len(queries) == 2
    assert queries[0].startswith("query ") and queries[1].startswith("english ")


def test_an_english_report_searches_the_web_once_per_turn(llm, monkeypatch):
    queries = []

    def web(query, search_depth="basic", max_results=3):
        queries.append(query)
        return fakes.fake_tavily(query)

    monkeypatch.setattr("src.utils.nodes.tavily_search", web)
    graph = make_graph()
    config, _ = start(graph, {"max_analyst": 1, "language": "en", "max_num_turns": 1})
    approve(graph, config)
    assert len(queries) == 1


def test_no_sources_gives_honest_placeholder_and_warning(llm, monkeypatch):
    monkeypatch.setattr("src.utils.nodes.tavily_search", fakes.empty_search)
    monkeypatch.setattr("src.utils.nodes.wikipedia_search", fakes.empty_search)
    graph = make_graph()
    config, _ = start(graph)
    final = approve(graph, config)

    assert {w["code"] for w in final["warnings"]} == {"no_sources"}
    assert len(final["warnings"]) == 2
    assert llm.calls.count("answer") == 0
    assert "## Sources" not in final["final_report"]
    assert "[1]" not in final["final_report"]


def test_used_up_tavily_credits_warn_once_per_analyst_and_the_report_still_gets_written(llm, monkeypatch):
    from src.utils.tools import SearchCreditsExhausted

    def web(query, *args, **kwargs):
        raise SearchCreditsExhausted("Error 432: usage limit")

    monkeypatch.setattr("src.utils.nodes.tavily_search", web)
    graph = make_graph()
    config, _ = start(graph)
    final = approve(graph, config)

    codes = [(w["code"], w["analyst"]) for w in final["warnings"]]
    assert sorted(codes) == [("search_credits_exhausted", "Analyst 1"), ("search_credits_exhausted", "Analyst 2")]
    assert llm.calls.count("answer") == 4  # Wikipedia still gave the experts something to work from
    assert {doc["kind"] for section in final["sections"] for doc in section["sources"]} == {"wikipedia"}
    assert "## Sources" in final["final_report"]


def test_credits_running_out_between_two_queries_keep_what_the_first_one_found(llm, monkeypatch):
    from src.utils.tools import SearchCreditsExhausted

    queries = []

    def web(query, search_depth="basic", max_results=3):
        queries.append(query)
        if len(queries) == 2:
            raise SearchCreditsExhausted("Error 432: usage limit")
        return fakes.fake_tavily(query)

    monkeypatch.setattr("src.utils.nodes.tavily_search", web)
    graph = make_graph()
    config, _ = start(graph, {"max_analyst": 1, "language": "tr", "max_num_turns": 1})
    final = approve(graph, config)

    assert len(queries) == 2 and queries[1].startswith("english ")
    assert [w["code"] for w in final["warnings"]] == ["search_credits_exhausted"]
    web_docs = [d for section in final["sections"] for d in section["sources"] if d["kind"] == "web"]
    assert web_docs, "the first query's results must not be thrown away"


def test_analyst_closing_sentence_ends_the_interview_early(llm):
    llm.close_after = 1
    graph = make_graph()
    config, _ = start(graph, {"max_num_turns": 4, "max_analyst": 1})
    approve(graph, config)
    assert llm.calls.count("answer") == 1


def test_max_num_turns_is_respected(llm):
    graph = make_graph()
    config, _ = start(graph, {"max_num_turns": 3, "max_analyst": 1})
    approve(graph, config)
    assert llm.calls.count("answer") == 3


def test_fact_check_fixes_flagged_claims(llm):
    llm.review_issues = [{"claim": "Another point", "problem": "not in the source"}]
    graph = make_graph()
    config, _ = start(graph)
    final = approve(graph, config)

    assert final["review_notes"] == [{"claim": "Another point", "problem": "not in the source", "fixed": True}]
    assert "Another point" not in final["final_report"]
    assert "revise" in llm.calls


def test_fact_check_clean_report_is_untouched(llm):
    graph = make_graph()
    config, _ = start(graph)
    final = approve(graph, config)
    assert final["review_notes"] == []
    assert "revise" not in llm.calls


def test_fact_check_failure_keeps_report_and_warns(llm):
    llm.fail_on = "review"
    graph = make_graph()
    config, _ = start(graph)
    final = approve(graph, config)

    assert final["final_report"]
    assert {"code": "review_failed"} in final["warnings"]


def test_every_report_is_fact_checked(llm):
    graph = make_graph()
    config, _ = start(graph)
    approve(graph, config)
    assert "review" in llm.calls


def test_report_draft_is_streamed(llm):
    graph = make_graph()
    config, _ = start(graph)
    drafts = []
    for _, mode, payload in graph.stream(Command(resume=""), config, stream_mode=["updates", "custom"], subgraphs=True):
        if mode == "custom" and "draft" in payload:
            drafts.append(payload["draft"])
    assert "".join(drafts).startswith("## Insights")


def test_turkish_report_uses_localized_labels(llm):
    graph = make_graph()
    config, _ = start(graph, {"language": "tr", "max_analyst": 1})
    final = approve(graph, config)
    assert "## Kaynaklar" in final["final_report"]


def test_wikipedia_gets_the_report_language(llm, monkeypatch):
    langs = []
    monkeypatch.setattr(
        "src.utils.nodes.wikipedia_search",
        lambda query, lang="en", **k: langs.append(lang) or fakes.fake_wikipedia(query, lang),
    )
    graph = make_graph()
    config, _ = start(graph, {"language": "tr", "max_analyst": 1})
    approve(graph, config)
    assert set(langs) == {"tr"}


@pytest.mark.parametrize("graph_name", ["question_answer_graph", "create_analysts", "research_agent"])
def test_studio_graphs_compile(graph_name):
    import json
    from importlib import import_module
    from pathlib import Path

    spec = json.loads((Path(__file__).resolve().parent.parent / "langgraph.json").read_text())["graphs"][graph_name]
    module_path, attribute = spec.removeprefix("./").split(":")
    module = import_module(module_path.removesuffix(".py").replace("/", "."))
    assert getattr(module, attribute) is not None


def test_a_figure_the_cited_source_lacks_goes_to_the_editor_even_when_the_model_finds_nothing(llm):
    from src.utils.nodes import review_report

    state = {
        "language": "en", "topic": "a law", "content": "It passed in 2012 with 6306 articles [1].",
        "sources": [
            {"n": 1, "title": "One", "url": "https://one.example", "content": "It passed in 2012."},
            {"n": 2, "title": "Two", "url": "https://two.example", "content": "The text has 6306 articles."},
        ],
    }
    update = review_report(state)

    assert llm.calls == ["review", "revise"]  # the editor ran although the model's own fact-check found nothing
    [note] = update["review_notes"]
    assert "6306" in note["claim"] and "[2]" in note["problem"] and note["fixed"] is True


def test_a_report_whose_figures_all_hold_is_not_sent_to_the_editor(llm):
    from src.utils.nodes import review_report

    state = {
        "language": "en", "topic": "a law", "content": "It passed in 2012 [1].",
        "sources": [{"n": 1, "title": "One", "url": "https://one.example", "content": "It passed in 2012."}],
    }
    assert review_report(state) == {"review_notes": []} and llm.calls == ["review"]


class FigureChat(fakes.ScriptedChat):
    """Writes an introduction that carries a figure no source has."""

    def respond(self, messages):
        text = super().respond(messages)
        if self.calls[-1] == "intro":
            return text.replace("Summary sentence [1].", "It cites 6306 and several other things at length [1].")
        if self.calls[-1] == "revise":
            return "# Big Title\n\n## Introduction\n\nIt cites several things at length [1]."
        return text


def test_an_introduction_with_a_figure_no_source_has_goes_to_the_editor(monkeypatch):
    from src.utils.nodes import write_conclusion, write_introduction

    model = FigureChat(calls=[])
    monkeypatch.setattr("src.utils.nodes.get_llm", lambda: model)
    state = {
        "language": "en", "topic": "a law", "merged_sections": ["A claim [1]."],
        "sources": [{"n": 1, "title": "One", "url": "https://one.example", "content": "A claim."}],
    }
    assert write_introduction(state) == {
        "introduction": "# Big Title\n\n## Introduction\n\nIt cites several things at length [1]."}
    assert model.calls == ["intro", "revise"]
    # a conclusion without a bad figure is not edited
    assert write_conclusion(state) == {"conclusion": "## Conclusion\n\nSummary sentence [1]."}
    assert model.calls == ["intro", "revise", "conclusion"]


def test_a_failing_editor_leaves_the_introduction_as_written(monkeypatch):
    from src.utils.nodes import write_introduction

    model = FigureChat(calls=[], fail_on="revise")
    monkeypatch.setattr("src.utils.nodes.get_llm", lambda: model)
    state = {
        "language": "en", "topic": "a law", "merged_sections": ["A claim [1]."],
        "sources": [{"n": 1, "title": "One", "url": "https://one.example", "content": "A claim."}],
    }
    assert "6306" in write_introduction(state)["introduction"]


def test_the_fact_checker_is_told_todays_date(monkeypatch):
    from datetime import date

    from langchain_core.runnables import RunnableLambda

    from src.utils.nodes import review_report

    class Recording(fakes.ScriptedChat):
        seen: list = []

        def with_structured_output(self, schema, **kwargs):
            inner = super().with_structured_output(schema, **kwargs)

            def run(messages):
                self.seen.append(fakes.system_text(messages))
                return inner.invoke(messages)

            return RunnableLambda(run)

    model = Recording(calls=[])
    monkeypatch.setattr("src.utils.nodes.get_llm", lambda: model)
    state = {
        "language": "en", "topic": "a law", "content": "It passed in 2012 [1].",
        "sources": [{"n": 1, "title": "One", "url": "https://one.example", "content": "It passed in 2012."}],
    }
    review_report(state)
    assert f"Today's date is {date.today().isoformat()}" in model.seen[0]
