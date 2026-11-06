from src.utils.ranking import clean_title, host_of, is_blocked, page_key, quality_bonus, rank_results


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


def test_a_repeated_trailing_title_segment_is_dropped():
    title = (
        "Testing and Debugging - Build a Multi-Agent Chat System with Langgraph"
        " - Build a Multi-Agent Chat System with Langgraph"
    )
    assert clean_title(title, "https://oboe.com/learn/x") == (
        "Testing and Debugging - Build a Multi-Agent Chat System with Langgraph"
    )
    assert clean_title("A very long title - A very long title", "https://example.com/x") == "A very long title"


def test_commas_glued_to_the_next_word_get_a_space_but_numbers_do_not():
    assert clean_title(
        "Contract for Checkpoint,Interrupt, and Resume Semantics", "https://example.com/x"
    ) == "Contract for Checkpoint, Interrupt, and Resume Semantics"
    assert clean_title("Over 1,000 agents in production", "https://example.com/x") == "Over 1,000 agents in production"


def test_a_site_name_that_contains_the_whole_host_is_stripped_from_the_title():
    title = "Replay, Resume, and Idempotency · LangGraph Field Guide · Doc Holiday Library"
    assert clean_title(title, "https://doc.holiday/library/x") == "Replay, Resume, and Idempotency · LangGraph Field Guide"
    assert clean_title("Something useful here · Library of Ai", "https://ai.co/x") == "Something useful here · Library of Ai"


def test_redirect_links_and_shorteners_are_blocked_so_a_page_is_not_listed_twice():
    for url in (
        "https://go.microsoft.com/fwlink?linkid=2316094&clcid=0x41f",
        "https://aka.ms/vector-search",
        "https://bit.ly/3abc",
    ):
        assert is_blocked(url), url
    assert not is_blocked("https://go.microsoft.com/docs")
    assert not is_blocked("https://learn.microsoft.com/en-us/azure/documentdb/vector-search")


def test_trailing_colons_and_dangling_via_are_trimmed_from_titles():
    assert clean_title("FINOS AI Governance Framework:", "https://air-governance-framework.finos.org/x") == (
        "FINOS AI Governance Framework"
    )
    assert clean_title(
        "Defending LLM Embeddings Against Inversion Attacks via ...", "https://ojs.aaai.org/index.php/AAAI/article/view/1"
    ) == "Defending LLM Embeddings Against Inversion Attacks"


def test_academic_titles_cut_short_with_a_glued_ellipsis_are_trimmed():
    cases = [
        ("Intermittent fasting strategies and their effects on body weight: systematic review and network meta-analysis of…",
         "https://pmc.ncbi.nlm.nih.gov/articles/PMC12175170",
         "Intermittent fasting strategies and their effects on body weight: systematic review and network meta-analysis"),
        ("Effects of Intermittent Energy Restriction on Body Composition and Cardiometabolic Risk Markers – A…",
         "https://www.sciencedirect.com/science/article/pii/S2161831323013911",
         "Effects of Intermittent Energy Restriction on Body Composition and Cardiometabolic Risk Markers"),
        ("Comparison of Different Intermittent Fasting Patterns for Weight Loss in Adults: A…",
         "https://pubmed.ncbi.nlm.nih.gov/40367516",
         "Comparison of Different Intermittent Fasting Patterns for Weight Loss in Adults"),
        ("Intermittent versus continuous energy restriction on weight loss: a systematic review and meta-analysis of randomized controlled…",
         "https://link.springer.com/article/10.1186/s12967-018-1748-4",
         "Intermittent versus continuous energy restriction on weight loss: a systematic review and meta-analysis"),
    ]
    for title, url, expected in cases:
        assert clean_title(title, url) == expected


def test_a_trailing_period_is_dropped_unless_it_belongs_to_an_abbreviation():
    assert clean_title("Intermittent fasting for adults with overweight or obesity.", "https://pubmed.ncbi.nlm.nih.gov/1") == (
        "Intermittent fasting for adults with overweight or obesity"
    )
    for title in ("Made in the U.S.", "Wait for it...", "Annual report of Acme Inc."):
        assert clean_title(title, "https://example.com/x") == title


def test_the_conclusion_prompt_forbids_new_claims():
    from src.utils.prompts import intro_conclusion_instructions

    assert "never add a claim, number or recommendation" in intro_conclusion_instructions


def test_very_long_titles_are_shortened_at_a_word_boundary():
    question = (
        "In a LangGraph workflow, several parallel nodes update the same state and a process can be resumed "
        "from a checkpoint. How would you design reducers, checkpoint boundaries, and conflict handling so that "
        "the result is deterministic?"
    )
    shortened = clean_title(question, "https://example.com/interview/x")
    assert shortened.endswith("…") and len(shortened) <= 161
    assert question.startswith(shortened[:-1])
    academic = (
        "Resume Means Resume: A Machine-Checked Conformance Contract for Checkpoint, Interrupt, and Resume "
        "Semantics in Workflow Persistence Layers"
    )
    assert clean_title(academic, "https://arxiv.org/html/2608.03836v1") == academic


def test_the_review_and_report_prompts_guard_against_self_contradicting_sentences():
    from src.utils.prompts import report_writer_instructions, review_instructions

    assert "framing is contradicted" in review_instructions
    assert "framing match its evidence" in report_writer_instructions


def test_consultancy_hosts_rank_below_ordinary_articles():
    results = [
        result("https://simorconsulting.com/blog/vector-database-checklist", 0.7),
        result("https://example.com/blog/vector-database-checklist", 0.6),
    ]
    assert rank_results(results, 2)[0]["url"] == "https://example.com/blog/vector-database-checklist"


def test_course_catalog_pages_rank_below_ordinary_articles():
    results = [
        result("https://www.coursera.org/learn/multi-agent-systems", 0.7),
        result("https://udemy.com/course/langgraph", 0.7),
        result("https://example.com/blog/langgraph-state", 0.6),
    ]
    assert rank_results(results, 3)[0]["url"] == "https://example.com/blog/langgraph-state"


def test_the_prompts_steer_away_from_catalogs_and_double_claims():
    from src.utils.prompts import revise_instructions, source_selection_instructions

    assert "course catalogs" in source_selection_instructions
    assert "never leave the original claim next to its corrected version" in revise_instructions


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


def test_two_urls_of_one_page_share_a_key():
    ietf = "RFC 7168: The Hyper Text Coffee Pot Control Protocol for Tea Efficiently Brewing"
    assert page_key("https://datatracker.ietf.org/doc/html/rfc7168", ietf) == page_key(
        "https://datatracker.ietf.org/doc/rfc7168", ietf)
    uzh = "Albert Einstein - University of Zurich"
    assert page_key("https://www.uzh.ch/cmsssl/de/about/portrait/nobelprize/einstein.html", uzh) == page_key(
        "https://www.uzh.ch/de/researchinnovation/excellence/nobelprize/einstein.html", uzh)
    wiki = "Hyper Text Coffee Pot Control Protocol - Wikipedia"
    assert page_key("https://en.wikipedia.org/wiki/Hyper_Text_Coffee_Pot_Control_Protocol", wiki) == page_key(
        "https://en.m.wikipedia.org/wiki/Hyper_Text_Coffee_Pot_Control_Protocol", wiki)


def test_different_pages_never_share_a_key():
    # a different title, a different last segment, or the same page name on another site
    assert page_key("https://arxiv.org/pdf/2312.10997", "Retrieval-Augmented Generation for Large Language Models: A Survey") != page_key(
        "https://arxiv.org/html/2609.06391v1", "Fine-grained Claim-level RAG Benchmark for Law")
    assert page_key("https://docs.example.com/a/guide", "Setting up the service") != page_key(
        "https://docs.example.com/a/setup", "Setting up the service")
    assert page_key("https://example.com/news/story", "A long enough title") != page_key(
        "https://other.org/news/story", "A long enough title")


def test_generic_titles_and_bare_segments_are_not_merged():
    one_word = "Overview"
    assert page_key("https://docs.example.com/a/overview", one_word) == "https://docs.example.com/a/overview"
    assert page_key("https://docs.example.com/b/overview", one_word) != page_key("https://docs.example.com/a/overview", one_word)
    assert page_key("https://news.example.com/latest/page/2", "Latest posts and news") == "https://news.example.com/latest/page/2"
    assert page_key("https://example.com/", "A long enough title") == "https://example.com/"
    assert page_key("not a url", "A long enough title") == "not a url"
