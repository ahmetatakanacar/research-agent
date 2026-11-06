from src.utils.sources import (
    cited_numbers,
    finalize_citations,
    format_numbered,
    merge_sections,
    rewrite_citations,
    source_line,
    sources_markdown,
    strip_sources_section,
    tidy_citations,
    unique_documents,
)


def doc(url, title="T", content="text", kind="web"):
    return {"url": url, "title": title, "content": content, "kind": kind}


def test_unique_documents_dedupes_by_url_and_skips_junk():
    docs = unique_documents([
        doc("https://a"), doc("https://a", content="other"), doc("https://b"),
        {"url": "", "content": "x"}, {"url": "https://c", "content": "  "}, "not a dict",
    ])
    assert [d["url"] for d in docs] == ["https://a", "https://b"]
    assert docs[0]["content"] == "text"


def test_unique_documents_merges_two_urls_of_one_page_but_not_two_pages_with_a_generic_title():
    title = "RFC 7168: The Hyper Text Coffee Pot Control Protocol for Tea Efficiently Brewing"
    docs = unique_documents([
        doc("https://datatracker.ietf.org/doc/html/rfc7168", title, "first text"),
        doc("https://datatracker.ietf.org/doc/rfc7168", title, "another part of the same page"),
        doc("https://docs.example.com/a/overview", "Overview"),
        doc("https://docs.example.com/b/overview", "Overview"),
    ])
    assert [d["url"] for d in docs] == [
        "https://datatracker.ietf.org/doc/html/rfc7168", "https://docs.example.com/a/overview",
        "https://docs.example.com/b/overview"]
    assert docs[0]["content"] == "first text"


def test_format_numbered_numbers_from_one():
    out = format_numbered([doc("https://a", "A"), doc("https://b", "B")])
    assert out.startswith("[1] A — https://a")
    assert "[2] B — https://b" in out


def test_rewrite_citations_handles_groups_and_drops_unknown():
    mapping = {1: 5, 2: 6}
    out = rewrite_citations("x [1] y [1, 2] z [9] w [2024]", mapping.get)
    assert out == "x [5] y [5][6] z w [2024]"


def test_removed_citation_leaves_clean_text():
    drop = {1: 1}.get
    assert rewrite_citations("A claim [7]. Next", drop) == "A claim. Next"
    assert rewrite_citations("A claim [7], more", drop) == "A claim, more"
    assert rewrite_citations("see [7] above", drop) == "see above"
    assert rewrite_citations("Kept [1].", drop) == "Kept [1]."


def test_cited_numbers():
    assert cited_numbers("a [1] b [2, 3] c [10]") == {1, 2, 3, 10}


def test_merge_sections_numbers_globally_by_first_citation():
    sections = [
        {"text": "A [2] then [1].", "sources": [doc("https://one"), doc("https://two")]},
        {"text": "B [1] and [2].", "sources": [doc("https://two"), doc("https://three")]},
    ]
    merged, sources = merge_sections(sections)
    assert [s["url"] for s in sources] == ["https://two", "https://one", "https://three"]
    assert merged[0] == "A [1] then [2]."
    assert merged[1] == "B [1] and [3]."


def test_merge_sections_gives_two_urls_of_one_page_one_number():
    title = "Albert Einstein - University of Zurich"
    sections = [
        {"text": "A [1].", "sources": [doc("https://www.uzh.ch/cmsssl/de/about/nobelprize/einstein.html", title)]},
        {"text": "B [1] and [2].", "sources": [doc("https://www.uzh.ch/de/excellence/nobelprize/einstein.html", title),
                                               doc("https://three.example/x", "Three")]},
    ]
    merged, sources = merge_sections(sections)
    assert [s["url"] for s in sources] == ["https://www.uzh.ch/cmsssl/de/about/nobelprize/einstein.html",
                                           "https://three.example/x"]
    assert merged == ["A [1].", "B [1] and [2]."]


def test_merge_sections_drops_citations_to_missing_sources():
    merged, sources = merge_sections([{"text": "Claim [3].", "sources": [doc("https://one")]}])
    assert merged == ["Claim."]
    assert sources == []


def test_merge_sections_ignores_uncited_sources():
    _, sources = merge_sections([{"text": "No cites here.", "sources": [doc("https://one")]}])
    assert sources == []


def test_misplaced_citations_are_those_tidy_citations_would_move():
    from src.utils.sources import count_misplaced_citations

    assert count_misplaced_citations("It rose. [1] Then it fell! [2][3] Done [4].") == 2
    # a citation that starts the next sentence and is followed by a comma is left alone, like tidy_citations does
    assert count_misplaced_citations("The indicators alone explain nothing. [8], pointing to a gap, adds a risk.") == 0
    assert count_misplaced_citations("Several kinds, etc. [3] are listed.") == 0
    for text in ("It rose. [1] Then it fell! [2][3] Done [4].", "A claim. [1], then more. [2] Last one."):
        assert count_misplaced_citations(tidy_citations(text)) == 0


def test_finalize_citations_renumbers_by_reading_order_and_prunes():
    sources = [{"n": 1, "url": "https://a", "title": "A"}, {"n": 2, "url": "https://b", "title": "B"},
               {"n": 3, "url": "https://c", "title": "C"}]
    parts, used = finalize_citations(["Intro [3].", "Body [1][3].", "End [7]."], sources)
    assert parts == ["Intro [1].", "Body [2][1].", "End."]
    assert [(s["n"], s["url"]) for s in used] == [(1, "https://c"), (2, "https://a")]


def test_strip_sources_section_removes_trailing_sources_in_any_language():
    assert strip_sources_section("Body.\n\n## Sources\n[1] x") == "Body."
    assert strip_sources_section("Body.\n\n### Kaynaklar\n1. x") == "Body."
    assert strip_sources_section("Body.\n\n## Sources of energy\nkeep") == "Body.\n\n## Sources of energy\nkeep"


def test_source_line_links_http_and_escapes_title():
    assert source_line({"url": "https://x.org/a(b)", "title": "A [b]"}) == "[A (b)](https://x.org/a%28b%29)"
    assert source_line({"url": "javascript:alert(1)", "title": "Not a link"}) == "Not a link"


def test_sources_markdown_is_an_ordered_list():
    md = sources_markdown([{"n": 1, "url": "https://a", "title": "A"}, {"n": 2, "url": "https://b", "title": "B"}], "Kaynaklar")
    assert md == "## Kaynaklar\n\n1. [A](https://a)\n2. [B](https://b)"
    assert sources_markdown([], "Sources") == ""


def test_tidy_citations_moves_groups_before_the_period_and_merges_neighbours():
    assert tidy_citations("a choice. [11][12] Then") == "a choice [11][12]. Then"
    assert tidy_citations("manageable. [3][13][14] [1][14] Next") == "manageable [1][3][13][14]. Next"
    assert tidy_citations("an error. [2][18] [19] Nodes") == "an error [2][18][19]. Nodes"
    assert tidy_citations("the end. [9][10]") == "the end [9][10]."
    assert tidy_citations("Right? [4] Yes! [5]") == "Right [4]? Yes [5]!"


def test_tidy_citations_leaves_good_text_alone():
    for text in (
        "Already fine [1][2]. Next [3].",
        "use e.g. [1] or etc. [2] here",
        "see [Docs](https://x.org) in [2024] and 1. [Title](https://y.org)",
        "wait... [1] what",
    ):
        assert tidy_citations(text) == text


def test_tidy_citations_is_idempotent():
    once = tidy_citations("one. [3][2] [1] two. [2]")
    assert once == "one [1][2][3]. two [2]."
    assert tidy_citations(once) == once


def test_tidy_citations_caps_a_group_keeping_the_most_authoritative_sources():
    authority = {1: 0.0, 2: 0.25, 3: 0.0, 4: 0.25, 5: 0.25, 6: 0.0}
    assert tidy_citations("claim [1][2][3][4][5][6].", cap=3, priority=authority.get) == "claim [2][4][5]."
    assert tidy_citations("claim [3][1][2].", cap=3, priority=authority.get) == "claim [1][2][3]."
    assert tidy_citations("claim [1][2][3][4].", cap=3) == "claim [1][2][3]."
    assert tidy_citations("one [1]. two [2][3][4][5].", cap=2, priority=authority.get) == "one [1]. two [2][4]."


def test_tidy_citations_also_fixes_groups_glued_to_the_period():
    assert tidy_citations("can be designed.[1][2] Next") == "can be designed [1][2]. Next"
    assert tidy_citations("Çıktı yönlendirilir.[3][7][8]") == "Çıktı yönlendirilir [3][7][8]."
    assert tidy_citations("first.[2] second.[1][3]") == "first [2]. second [1][3]."
    assert tidy_citations("e.g.[1] stays") == "e.g.[1] stays"
    once = tidy_citations("end.[2][1]")
    assert once == "end [1][2]." and tidy_citations(once) == once
