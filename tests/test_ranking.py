from src.utils.ranking import clean_title, host_of, is_blocked, quality_bonus, rank_results


def result(url, score=0.5, content="text"):
    return {"url": url, "title": url, "content": content, "score": score}


def test_social_video_and_user_document_sites_are_blocked():
    for url in (
        "https://www.youtube.com/watch?v=1",
        "https://youtu.be/abc",
        "https://uk.linkedin.com/posts/someone",
        "https://old.reddit.com/r/langgraph",
        "https://docs.google.com/document/d/1",
    ):
        assert is_blocked(url), url
    assert not is_blocked("https://notyoutube.com/x")
    assert not is_blocked("https://docs.langchain.com/oss/python")


def test_official_academic_and_government_sources_get_a_bonus():
    for url in (
        "https://docs.langchain.com/oss/python/langgraph",
        "https://www.nih.gov/news",
        "https://cs.stanford.edu/people",
        "https://www.ankara.edu.tr/duyuru",
        "https://arxiv.org/abs/2401.00001",
        "https://example.com/docs/getting-started",
        "https://numpy.readthedocs.io/en/latest/",
    ):
        assert quality_bonus(url) > 0, url


def test_blog_platforms_are_not_penalized_and_ordinary_sites_are_neutral():
    for url in ("https://medium.com/@someone/post", "https://team.medium.com/post", "https://x.substack.com/p/a",
                "https://example.com/post"):
        assert quality_bonus(url) == 0, url
    assert quality_bonus("not a url") < 0


def test_authority_outranks_a_slightly_more_relevant_ordinary_page():
    results = [
        result("https://example.com/post", 0.9),
        result("https://docs.example.org/guide", 0.7),
        result("https://medium.com/@a/post", 0.5),
    ]
    ranked = rank_results(results, 3)
    assert [r["url"] for r in ranked] == [
        "https://docs.example.org/guide",
        "https://example.com/post",
        "https://medium.com/@a/post",
    ]


def test_a_highly_relevant_blog_post_keeps_its_place():
    results = [
        result("https://medium.com/@expert/deep-dive", 0.95),
        result("https://example.com/news", 0.6),
    ]
    assert rank_results(results, 2)[0]["url"] == "https://medium.com/@expert/deep-dive"


def test_a_single_site_cannot_fill_the_list():
    results = [result(f"https://one.com/{i}", 0.9 - i / 100) for i in range(4)] + [result("https://two.com/a", 0.4)]
    ranked = rank_results(results, 3)
    assert [host_of(r["url"]) for r in ranked] == ["one.com", "one.com", "two.com"]


def test_the_list_is_filled_from_one_site_when_nothing_else_is_left():
    assert len(rank_results([result(f"https://one.com/{i}") for i in range(4)], 3)) == 3


def test_blocked_incomplete_and_unscored_results():
    results = [
        result("https://youtube.com/watch?v=1", 0.99),
        {"url": "https://a.com", "content": ""},
        {"title": "no url", "content": "x"},
        {"url": "https://b.com/x", "content": "text"},
    ]
    assert [r["url"] for r in rank_results(results, 5)] == ["https://b.com/x"]


def test_host_of_survives_garbage():
    assert host_of("https://www.Example.com/a") == "example.com"
    assert host_of("not a url") == ""
    assert host_of("http://[broken") == ""


def test_site_names_and_taglines_are_removed_from_titles():
    cases = [
        ("Subgraphs - Docs by LangChain", "https://docs.langchain.com/oss/python/langgraph/use-subgraphs", "Subgraphs"),
        ("How to Design an AI Agent State Machine — Marius Manolachi", "https://www.mariusmanolachi.com/blog/x",
         "How to Design an AI Agent State Machine"),
        ("Agent tracing: how to trace and debug AI agents - Articles - Braintrust", "https://www.braintrust.dev/a",
         "Agent tracing: how to trace and debug AI agents"),
        ("Durable Execution for AI Agent Runtimes | Zylos Research", "https://zylos.ai/x",
         "Durable Execution for AI Agent Runtimes"),
        ("Build multi-agent systems with LangGraph | Artificial Intelligence", "https://aws.amazon.com/blogs/x",
         "Build multi-agent systems with LangGraph"),
        ("Multi-agent systems, LangGraph · neurals", "https://neurals.ca/tech/x", "Multi-agent systems, LangGraph"),
        ("Preventing Infinite Loops in Agent Workflows — AI Infrastructure Leader | Keynote Speaker", "https://rajatpandit.com/x",
         "Preventing Infinite Loops in Agent Workflows"),
    ]
    for title, url, expected in cases:
        assert clean_title(title, url) == expected, title


def test_meaningful_titles_are_kept():
    for title, url in (
        ("LangGraph Multi-Agent — Supervisor Pattern Guide (2026)", "https://myengineeringpath.dev/x"),
        ("Safe to Resume? Breaking Execution Continuity via Rollback", "https://arxiv.org/html/2608.29381v1"),
        ("Docs - A", "https://example.com/x"),
        ("A", "https://a"),
    ):
        assert clean_title(title, url) == title


def test_truncated_titles_are_completed_from_the_url():
    cases = [
        ("Durable Agent Workflows Need Idempotency Before They ...",
         "https://hackernoon.com/durable-agent-workflows-need-idempotency-before-they-need-checkpoints",
         "Durable Agent Workflows Need Idempotency Before They Need Checkpoints"),
        ("State Management in LangGraph: The Foundation of ...",
         "https://medium.com/algomart/state-management-in-langgraph-the-foundation-of-reliable-ai-workflows-db98dd1499ca",
         "State Management in LangGraph: The Foundation of Reliable AI Workflows"),
        ("Multi-agent systems and ...", "https://example.com/blog/multi-agent-systems-and-swarms.html",
         "Multi-agent systems and Swarms"),
    ]
    for title, url, expected in cases:
        assert clean_title(title, url) == expected


def test_truncated_titles_the_url_cannot_complete_lose_the_ellipsis_and_dangling_words():
    cases = [
        ("State Management in LangGraph: The Foundation of ...", "https://medium.com/algomart/x",
         "State Management in LangGraph: The Foundation"),
        ("Security of large language models and ...", "https://www.researchgate.net/publication/1_x",
         "Security of large language models"),
        ("Something Different Here ...", "https://example.com/blog/unrelated-slug-here", "Something Different Here"),
        ("Yapay zeka ajanları ve …", "https://example.com/x", "Yapay zeka ajanları"),
    ]
    for title, url, expected in cases:
        assert clean_title(title, url) == expected


def test_titles_that_would_become_stubs_or_are_not_cut_off_are_left_alone():
    for title, url in (
        ("Too short ...", "https://example.com/x"),
        ("Complete title...", "https://example.com/x"),
        ("What is this and why...", "https://example.com/x"),
    ):
        assert clean_title(title, url) == title


def test_pdf_markers_are_removed_and_do_not_block_url_completion():
    assert clean_title("(PDF) Annual climate report 2025", "https://example.com/x") == "Annual climate report 2025"
    assert clean_title("[PDF] Annual climate report 2025", "https://example.com/x") == "Annual climate report 2025"
    assert clean_title("(PDF)", "https://example.com/x") == "(PDF)"
    assert clean_title(
        "(PDF) Security of large language models and ...",
        "https://www.researchgate.net/publication/9_Security_of_large_language_models_and_defence_strategies",
    ) == "Security of large language models and Defence Strategies"


def test_an_authoritative_domain_gets_no_bonus_when_the_page_is_barely_relevant():
    results = [
        result("https://ordinary.com/guide", 0.5),
        result("https://polen.itu.edu.tr/tez", 0.35),
        result("https://docs.example.org/guide", 0.6),
    ]
    assert [host_of(r["url"]) for r in rank_results(results, 3)] == ["docs.example.org", "ordinary.com", "polen.itu.edu.tr"]


def test_results_below_the_relevance_floor_are_dropped_unless_nothing_else_is_left():
    assert [r["url"] for r in rank_results([result("https://a.com/x", 0.8), result("https://b.com/x", 0.1)], 5)] == [
        "https://a.com/x"
    ]
    assert [r["url"] for r in rank_results([result("https://b.com/x", 0.1)], 5)] == ["https://b.com/x"]


def test_low_trust_hosts_are_blocked():
    for url in (
        "https://nightcat.cloudns.asia:9981/sitedoc/langgraph/v0.4.3/how-tos/agent-handoffs",
        "https://example.org:8443/page",
        "http://192.168.1.4/page",
        "https://somesite.tk/page",
        "https://x.duckdns.org/page",
    ):
        assert is_blocked(url), url
    assert not is_blocked("https://example.org:443/page")
    assert not is_blocked("https://docs.langchain.com/oss/python/langgraph/overview")


def test_company_service_pages_are_pushed_down():
    results = [
        result("https://uvik.net/services/langgraph-development-company", 0.8),
        result("https://example.com/agency-notes", 0.8),
        result("https://example.com/blog/langgraph-state", 0.6),
    ]
    results[1]["title"] = "We are a creative agency"
    results[0]["title"] = "LangGraph Development Company: Python AI Agent Experts"
    results[2]["title"] = "How LangGraph state works"
    assert rank_results(results, 3)[0]["url"] == "https://example.com/blog/langgraph-state"


def test_github_prefix_is_removed_from_titles():
    assert clean_title(
        "GitHub - kennethleungty/Handoffs-in-LangGraph: Clearly Explained", "https://github.com/kennethleungty/x"
    ) == "kennethleungty/Handoffs-in-LangGraph: Clearly Explained"
