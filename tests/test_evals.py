import json
import re

import pytest
from langchain_core.runnables import RunnableLambda

from evals import canary, judge, metrics, run
from src.utils import figures
from src.utils.languages import LANGUAGES

GOOD_REPORT = """# Title

## Introduction

RAG helps models use outside text [1][2]. It can still fail in practice [3].

---

## Insights

### First theme

Chunking changes what is retrieved [1][3]. Hybrid search often helps [2].

### Second theme

Evidence can be lost before generation [3].

---

## Conclusion

Measure retrieval and answers separately [1][2].

## Sources

1. [Doc one](https://docs.langchain.com/a)
2. [Paper](https://arxiv.org/abs/1)
3. [Blog](https://blog.example.com/x)
"""

GOOD_SOURCES = [
    {"n": 1, "url": "https://docs.langchain.com/a", "title": "Doc one"},
    {"n": 2, "url": "https://arxiv.org/abs/1", "title": "Paper"},
    {"n": 3, "url": "https://blog.example.com/x", "title": "Blog"},
]


def test_a_clean_report_passes_every_hard_check():
    measured = metrics.evaluate_report(GOOD_REPORT, GOOD_SOURCES, "en")
    assert measured["structured"] and measured["subheadings"] == 2
    assert measured["dangling_citations"] == 0 and measured["uncited_sources"] == 0
    assert measured["max_group_size"] == 2
    assert measured["authority_share"] == pytest.approx(0.667, abs=0.001)
    assert all(metrics.run_checks(measured).values())


def test_a_messy_report_fails_the_matching_checks():
    report = """# T

## Introduction

No citation here at all.

---

## Insights

Too many [1][2][3][4]. Out of order [3][1]. Split [2] [1]. After the stop. [1]

---

## Conclusion

Unsorted and fine [1].
"""
    sources = [
        {"n": 1, "url": "https://youtube.com/watch?v=1", "title": "Alpha"},
        {"n": 2, "url": "https://example.com/2", "title": "Same title…"},
        {"n": 3, "url": "https://docs.example.com/3", "title": "Same title…"},
    ]
    measured = metrics.evaluate_report(report, sources)
    failed = {name for name, ok in metrics.run_checks(measured).items() if not ok}
    assert failed == {
        "no_dangling_citations", "groups_at_most_three", "groups_sorted", "no_adjacent_groups",
        "citations_before_punctuation", "introduction_cited", "no_duplicate_sources",
        "no_blocked_sources", "titles_complete",
    }


def test_the_same_title_on_different_sites_is_not_a_duplicate_but_a_redirect_copy_is():
    different_sites = [
        {"n": 1, "url": "https://en.wikipedia.org/wiki/Photoelectric_effect", "title": "Photoelectric effect"},
        {"n": 2, "url": "https://www.physicsbootcamp.org/photoelectric.html", "title": "Photoelectric Effect"},
    ]
    assert metrics.evaluate_report(GOOD_REPORT, different_sites)["duplicate_titles"] == 0
    redirect_copy = [
        {"n": 1, "url": "https://learn.microsoft.com/azure/documentdb/vector-search", "title": "Vector Store - Azure"},
        {"n": 2, "url": "https://go.microsoft.com/fwlink?linkid=1", "title": "Vector Store - Azure"},
    ]
    assert metrics.evaluate_report(GOOD_REPORT, redirect_copy)["duplicate_titles"] == 1


def test_a_report_without_three_parts_is_flagged():
    measured = metrics.evaluate_report("## Insights\n\nOnly a body [1].", GOOD_SOURCES[:1])
    assert not measured["structured"]
    assert not metrics.run_checks(measured)["three_part_structure"]


TURKISH = (
    "# Başlık\n\n## Giriş\n\n"
    "Vektör veritabanları verileri sayısal vektörler olarak saklar ve benzerlik ile arama yapar [1]. "
    "Bu yaklaşım, geleneksel veritabanlarına göre daha esnek bir arama sağlar ve çok sayıda uygulamada kullanılır [2].\n\n"
    "## Kaynaklar\n\n1. x\n"
)
ENGLISH_SENTENCE = "The authors argue that this approach is better than the older methods because of its flexibility. "


def test_a_stray_sentence_in_the_other_language_is_found():
    assert metrics.language_mix(TURKISH, "tr")["wrong"] == 0
    stray = TURKISH.replace("## Kaynaklar", ENGLISH_SENTENCE + "\n\n## Kaynaklar")
    mixed = metrics.language_mix(stray, "tr")
    assert mixed["wrong"] == 1 and mixed["sentences"] == 3 and mixed["examples"][0].startswith("The authors")
    assert metrics.language_mix(TURKISH, "en")["wrong"] == 2
    assert metrics.language_mix(TURKISH, None)["wrong"] == 0

    sources = [{"n": 1, "url": "https://a.org/1", "title": "A"}, {"n": 2, "url": "https://b.org/2", "title": "B"}]
    clean = metrics.run_checks(metrics.evaluate_report(TURKISH, sources, "tr"))
    assert clean["language_consistent"]
    assert not metrics.run_checks(metrics.evaluate_report(stray, sources, "tr"))["language_consistent"]


def test_figures_in_cited_sentences_must_appear_in_the_cited_sources():
    report = (
        "## Insights\n\nThe review covered 6,582 adults and found a change of 1,29 kg [1]. "
        "It also reported a 45% drop in appetite [1]. Only 3 trials were included in the review [1]. "
        "A smaller study ran in 2019 with a 7% dropout [2]. This sentence has no figures at all [2].\n\n"
        "## Sources\n\n1. x\n"
    )
    texts = {
        1: {"content": "A meta-analysis of 6582 adults found a change of 1.29 kg. Three trials were included."},
        2: {"content": "The study ran in 2019."},
    }
    result = figures.number_grounding(report, texts)
    assert result["numbers"] == 5  # 6,582 / 1,29 / 45% / 2019 / 7%, and not the bare 3
    assert [row["number"] for row in result["rows"]] == ["45", "7"]
    assert result["grounding"] == pytest.approx(3 / 5, abs=0.001)
    assert figures.number_grounding("## Insights\n\nNothing numeric here at all [1].", texts)["grounding"] is None


def test_a_figure_from_the_question_is_not_a_claim_and_a_misplaced_citation_is_told_from_an_invented_figure():
    report = ("## Insights\n\nNo source confirms that the protocol was developed in 1987 at all [1]. "
              "The law passed in 2012 became the main legal framework [1]. "
              "The programme reached 9,300 households last year [1].\n\n## Sources\n\n1. x\n")
    texts = {1: {"content": "Nothing relevant is said here."}, 2: {"content": "The law passed in 2012 in the end."}}
    result = figures.number_grounding(report, texts, topic="The development of the protocol in 1987")
    assert result["numbers"] == 2  # 1987 comes from the question and is left out
    by_number = {row["number"]: row for row in result["rows"]}
    assert by_number["2012"]["elsewhere"] == [2] and by_number["9,300"]["elsewhere"] == []
    assert result["ungrounded"] == 2 and result["misplaced"] == 1


def test_claims_are_cited_sentences_without_the_markers():
    report = (
        "# T\n\n## Introduction\n\nChunk size changes which passages are retrieved [1][2]. Short [1]. "
        "No citation in this one at all.\n\n---\n\n## Insights\n\n### Theme\n\n"
        "Hybrid search combines keyword and vector matching [3].\n\n## Sources\n\n1. x\n"
    )
    claims = judge.extract_claims(report)
    assert [c["text"] for c in claims] == [
        "Chunk size changes which passages are retrieved.",
        "Hybrid search combines keyword and vector matching.",
    ]
    assert [c["cites"] for c in claims] == [[1, 2], [3]]
    assert [c["id"] for c in claims] == [1, 2]


def test_claims_over_the_cap_are_sampled_evenly_and_repeatably():
    body = " ".join(f"Claim number {i} says something specific about retrieval [1]." for i in range(40))
    first = judge.extract_claims(f"## Insights\n\n{body}", max_claims=10)
    second = judge.extract_claims(f"## Insights\n\n{body}", max_claims=10)
    assert len(first) == 10 and first == second
    assert first[0]["text"].startswith("Claim number 0") and first[1]["text"].startswith("Claim number 4")
    assert [c["id"] for c in first] == list(range(1, 11))


SOURCE_QUOTE = "says several specific things about retrieval"
REPORT_QUOTE = "SAYS-UNCONFIRMED clearly about retrieval"


class ScriptedJudge:
    """Stands in for the judge. A claim containing INVENTED is unsupported, any other is supported with a quote
    taken from the source. A requirement containing FAILS is not met, one containing UNSURE is unknown, and
    any other is met when the report contains SAYS-UNCONFIRMED."""

    def __init__(self, skip=(), fail=False, unknown=(), quote=None, absence=False):
        self.prompts, self.skip, self.fail, self.unknown = [], set(skip), fail, set(unknown)
        self.quote, self.absence = quote, absence

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            prompt = messages[0].content
            self.prompts.append(prompt)
            if self.fail:
                raise self.fail if isinstance(self.fail, BaseException) else RuntimeError("scripted judge failure")
            if schema is judge.ClaimVerdicts:
                texts = dict(re.findall(r"^\[c(\d+)\] (.*?) \(cites", prompt, re.M))
                return judge.ClaimVerdicts(verdicts=[
                    judge.ClaimVerdict(
                        id=int(i), reasoning="scripted", states_absence=self.absence,
                        quote="" if self.absence else (self.quote or SOURCE_QUOTE),
                        verdict=("unsupported" if "INVENTED" in text else "unknown" if int(i) in self.unknown
                                 else "supported"))
                    for i, text in texts.items() if int(i) not in self.skip
                ])
            requirement, report = prompt.split("REQUIREMENT:")[1].split("REPORT:")
            if "FAILS" in requirement:
                verdict = "not_met"
            elif "UNSURE" in requirement:
                verdict = "unknown"
            else:
                verdict = "met" if "SAYS-UNCONFIRMED" in report else "not_met"
            return judge.GradeVerdict(quote=self.quote or REPORT_QUOTE, reasoning="scripted", verdict=verdict)

        return RunnableLambda(respond)


def _claims(count, invented=()):
    return [
        {"id": i, "text": ("INVENTED " if i in invented else "") + f"Claim {i} is long enough to be judged.", "cites": [1]}
        for i in range(1, count + 1)
    ]


TEXTS = {1: {"title": "Doc", "url": "https://example.com",
             "content": "The source text says several specific things about retrieval and chunking."}}


def test_claims_are_judged_in_batches_and_scored():
    scripted = ScriptedJudge()
    result = judge.judge_claims(scripted, _claims(20, invented={3, 4}), TEXTS)
    assert len(scripted.prompts) == 3
    assert (result["claims"], result["supported"], result["unsupported"]) == (20, 18, 2)
    assert result["faithfulness"] == 0.9 and result["errors"] == 0
    assert "retrieval and chunking" in scripted.prompts[0]
    row = result["rows"][0]
    assert row["quote"] == SOURCE_QUOTE and row["quote_found"] and row["verdict"] == "supported"


class RateLimitError(Exception):
    """Looks, by name and text, like what the openai package raises."""


class AuthenticationError(Exception): ...


OUT_OF_CREDIT = RateLimitError("Error code: 429 - {'error': {'type': 'insufficient_quota'}}")


def test_the_judge_counts_an_ordinary_failure_but_lets_a_dry_account_or_a_bad_key_through():
    ordinary = judge.judge_claims(ScriptedJudge(fail=True), _claims(3), TEXTS)
    assert ordinary["errors"] == 1 and ordinary["judged"] == 0 and ordinary["faithfulness"] is None
    assert judge.judge_criteria(ScriptedJudge(fail=True), "report", ["a criterion"])["passed"] is None
    # a rate limit that is only a rate limit is an ordinary failure too
    assert judge.judge_claims(ScriptedJudge(fail=RateLimitError("slow down")), _claims(3), TEXTS)["errors"] == 1

    for failure in (OUT_OF_CREDIT, AuthenticationError("bad key")):
        assert judge.fatal(failure)
        with pytest.raises(type(failure)):
            judge.judge_claims(ScriptedJudge(fail=failure), _claims(3), TEXTS)
        with pytest.raises(type(failure)):
            judge.judge_criteria(ScriptedJudge(fail=failure), "report", ["a criterion"])
        with pytest.raises(type(failure)):
            judge.judge_coverage(ScriptedJudge(fail=failure), "report", ["a point"])
    assert not judge.fatal(RuntimeError("scripted judge failure"))


def test_a_quote_is_matched_ignoring_case_punctuation_and_spacing():
    pool = [TEXTS[1]["content"]]
    assert judge.quote_found("Says SEVERAL specific,  things about retrieval!", pool)
    assert judge.quote_found("The source text … retrieval and chunking", pool)
    assert not judge.quote_found("The source text … never in there", pool)
    assert not judge.quote_found("retrieval", pool)  # too short to count as evidence
    assert not judge.quote_found("", pool)
    assert not judge.quote_found(SOURCE_QUOTE, ["Some other source entirely."])


def test_a_long_quote_stitched_from_several_passages_counts_but_a_paraphrase_does_not():
    source = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
              "upsilon phi chi psi omega")
    stitched = "alpha beta gamma delta epsilon zeta eta theta pi rho sigma tau upsilon phi"  # two passages, no ellipsis
    assert judge.quote_found(stitched, [source])
    assert not judge.quote_found("omega psi chi phi upsilon tau sigma rho pi omicron xi nu mu lambda", [source])
    # the same words in a different order are a summary, not a quote
    assert not judge.quote_found("theta zeta alpha eta beta gamma epsilon delta kappa iota lambda mu", [source])


def test_a_favourable_verdict_without_a_real_quote_is_not_accepted():
    result = judge.judge_claims(ScriptedJudge(quote="words the source never contains at all"), _claims(3), TEXTS)
    assert result["unverified"] == 3 and result["supported"] == 0 and result["faithfulness"] == 0.0
    row = result["rows"][0]
    assert row["judge_verdict"] == "supported" and row["verdict"] == "unverified" and not row["quote_found"]


def test_a_quote_from_a_source_the_sentence_does_not_cite_does_not_count():
    texts = {**TEXTS, 2: {"title": "Other", "url": "https://other.example", "content": "Something else entirely here."}}
    claims = [{"id": 1, "text": "Claim 1 is long enough to be judged.", "cites": [2]}]
    assert judge.judge_claims(ScriptedJudge(), claims, texts)["unverified"] == 1


class LeakyJudge:
    """Quotes the highest-numbered source it is shown, whichever source the sentence cites."""

    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            prompt = messages[0].content
            claims = re.findall(r"^\[c(\d+)\] (.*?) \(cites", prompt, re.M)
            sources = dict(re.findall(r"^\[(\d+)\] .*\n(.*)$", prompt.split("SOURCES:")[1], re.M))
            self.calls.append(len(claims))
            quote = sources[str(max(int(n) for n in sources))]
            return judge.ClaimVerdicts(verdicts=[
                judge.ClaimVerdict(id=int(i), quote=quote, reasoning="x", verdict="supported") for i, _ in claims])

        return RunnableLambda(respond)


def test_a_claim_whose_quote_came_from_another_sentences_source_is_asked_again_alone():
    texts = {
        1: {"title": "One", "url": "https://one.example", "content": "Alpha beta gamma delta epsilon is in the first source."},
        2: {"title": "Two", "url": "https://two.example", "content": "Zeta eta theta iota kappa is in the second source."},
    }
    claims = [{"id": 1, "text": "Claim 1 is long enough to be judged.", "cites": [1]},
              {"id": 2, "text": "Claim 2 is long enough to be judged.", "cites": [2]}]
    leaky = LeakyJudge()
    result = judge.judge_claims(leaky, claims, texts)
    assert leaky.calls == [2, 1]  # the batch, then claim 1 alone with only its own source
    assert [row["verdict"] for row in result["rows"]] == ["supported", "supported"]
    assert [row["retried"] for row in result["rows"]] == [True, False]
    assert result["supported"] == 2 and result["unverified"] == 0


def test_a_sentence_that_states_an_absence_needs_no_quote():
    result = judge.judge_claims(ScriptedJudge(absence=True), _claims(2), TEXTS)
    assert result["supported"] == 2 and result["unverified"] == 0


def test_unknown_is_left_out_of_the_score_and_counted_on_its_own():
    result = judge.judge_claims(ScriptedJudge(unknown={2}), _claims(4), TEXTS)
    assert result["unknown"] == 1 and result["judged"] == 3 and result["faithfulness"] == 1.0


def test_partial_unverified_and_left_out_verdicts_are_scored_as_documented():
    rows = [{"verdict": v} for v in
            ("supported", "partial", "unsupported", "unverified", "unjudged", "not_a_claim", "unknown")]
    scored = judge.faithfulness(rows)
    assert scored["judged"] == 4 and scored["not_a_claim"] == 1 and scored["unknown"] == 1
    assert scored["faithfulness"] == pytest.approx(1.5 / 4, abs=0.001)
    assert judge.faithfulness([])["faithfulness"] is None

    result = judge.judge_claims(ScriptedJudge(skip={2}), _claims(3), TEXTS)
    assert result["unjudged"] == 1 and result["judged"] == 2


def test_a_failing_judge_does_not_crash_the_run():
    result = judge.judge_claims(ScriptedJudge(fail=True), _claims(3), TEXTS)
    assert result["errors"] == 1 and result["judged"] == 0 and result["faithfulness"] is None
    assert judge.judge_criteria(ScriptedJudge(fail=True), "report", ["x"])["passed"] is None
    assert judge.judge_coverage(ScriptedJudge(fail=True), "report", ["x"])["coverage"] is None


def test_every_criterion_gets_its_own_judge_call_and_all_of_them_must_be_met():
    scripted = ScriptedJudge()
    result = judge.judge_criteria(scripted, "It SAYS-UNCONFIRMED clearly.", ["first", "second"])
    assert len(scripted.prompts) == 2 and result["passed"] is True
    assert [row["requirement"] for row in result["criteria"]] == ["first", "second"]
    assert "first" in scripted.prompts[0] and "second" not in scripted.prompts[0]

    failing = judge.judge_criteria(ScriptedJudge(), "It SAYS-UNCONFIRMED clearly.", ["first", "FAILS second"])
    assert failing["passed"] is False and failing["criteria"][0]["verdict"] == "met"
    assert judge.judge_criteria(ScriptedJudge(), "It invents things.", ["first"])["passed"] is False


def test_an_unknown_criterion_leaves_the_result_open_unless_another_one_fails():
    open_result = judge.judge_criteria(ScriptedJudge(), "It SAYS-UNCONFIRMED.", ["first", "UNSURE second"])
    assert open_result["passed"] is None
    assert judge.judge_criteria(ScriptedJudge(), "x", ["UNSURE a", "FAILS b"])["passed"] is False


def test_coverage_counts_a_point_only_with_a_quote_found_in_the_report():
    report = f"## Insights\n\nRAG {REPORT_QUOTE}.\n\n## Sources\n\n1. x\n"
    covered = judge.judge_coverage(ScriptedJudge(), report, ["point one", "point two"])
    assert (covered["points"], covered["covered"], covered["coverage"]) == (2, 2, 1.0)
    assert covered["rows"][0]["point"] == "point one"

    invented = judge.judge_coverage(ScriptedJudge(quote="a quote that is nowhere in this report"), report, ["p"])
    assert invented["covered"] == 0 and invented["unverified"] == 1 and invented["coverage"] == 0.0

    missing = judge.judge_coverage(ScriptedJudge(), "## Insights\n\nNothing relevant.\n\n## Sources\n\n1. x\n", ["p"])
    assert missing["covered"] == 0 and missing["unverified"] == 0
    unsure = judge.judge_coverage(ScriptedJudge(), report, ["UNSURE p"])
    assert unsure["unknown"] == 1 and unsure["covered"] == 0


def test_evals_do_not_trace_unless_asked_to(monkeypatch):
    import importlib
    import os

    import evals

    monkeypatch.delenv("EVALS_TRACE", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    importlib.reload(evals)
    assert os.environ["LANGSMITH_TRACING_V2"] == "false"

    monkeypatch.setenv("EVALS_TRACE", "1")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    importlib.reload(evals)
    assert os.environ["LANGSMITH_TRACING_V2"] == "true"


def test_the_judge_version_follows_its_prompts():
    assert re.fullmatch(r"[0-9a-f]{8}", judge.VERSION)


def _row(words=800, authority=0.5, failed=(), claims=None, rubric=None, tokens=1000, numbers=None, coverage=None):
    checks = {name: name not in failed for name in metrics.HARD_CHECKS}
    row = {
        "metrics": {"words": words, "subheadings": 3, "sources": 10, "authority_share": authority,
                    "max_host_share": 0.2, "marketing_sources": 0},
        "checks": checks, "usage": {"total_tokens": tokens, "elapsed": 60.0},
    }
    if claims:
        row["claims"] = claims
    if rubric is not None:
        row["rubric"] = {"passed": rubric, "reason": "x"}
    if numbers:
        row["numbers"] = numbers
    if coverage:
        row["coverage"] = coverage
    return row


def test_aggregate_summarises_checks_faithfulness_honesty_numbers_and_coverage():
    rows = [
        _row(claims={"judged": 10, "supported": 8, "partial": 2, "unsupported": 0, "contradicted": 0, "unverified": 0},
             rubric=True, numbers={"numbers": 10, "ungrounded": 1},
             coverage={"judged": 3, "covered": 3}),
        _row(failed=("groups_sorted",),
             claims={"judged": 10, "supported": 6, "partial": 0, "unsupported": 3, "contradicted": 1, "unverified": 2},
             rubric=False, numbers={"numbers": 10, "ungrounded": 3}, coverage={"judged": 3, "covered": 1}),
        {"error": "RuntimeError: boom"},
    ]
    agg = run.aggregate(rows)
    assert (agg["cases"], agg["errors"]) == (3, 1)
    assert agg["faithfulness"] == pytest.approx((9 + 6) / 20)
    assert (agg["unsupported_claims"], agg["contradicted_claims"], agg["unverified_claims"]) == (3, 1, 2)
    assert agg["honesty_pass_rate"] == 0.5
    assert agg["number_grounding"] == pytest.approx(0.8) and agg["ungrounded_numbers"] == 4
    assert agg["coverage"] == pytest.approx(4 / 6, abs=0.001)
    assert agg["hard_checks_passed"] == pytest.approx(1 - 1 / (2 * len(metrics.HARD_CHECKS)), abs=0.001)
    assert agg["tokens"] == 2000


def test_aggregate_without_a_judge_leaves_model_graded_fields_empty():
    agg = run.aggregate([_row()])
    assert agg["faithfulness"] is None and agg["honesty_pass_rate"] is None and agg["coverage"] is None
    assert agg["number_grounding"] is None


def test_check_rates_say_which_checks_fail_everywhere_or_never():
    rows = [_row(failed=("groups_sorted", "body_cited")), _row(failed=("groups_sorted",))]
    assert run.check_rates(rows)["groups_sorted"] == 0 and run.check_rates(rows)["body_cited"] == 0.5
    notes = run.check_notes(rows)
    assert "groups_sorted" in notes[0] and "fail in every case" in notes[0]
    assert f"{len(metrics.HARD_CHECKS) - 2} of {len(metrics.HARD_CHECKS)} checks pass in every case" in notes[-1]
    assert run.check_notes([_row()]) == []


def test_tries_give_pass_at_k_and_pass_hat_k():
    ok, bad = _row(), _row(failed=("groups_sorted",))
    rows = {"a": ok, "a#2": bad, "b": ok, "b#2": ok, "c": bad, "c#2": bad}
    assert run.consistency(rows) == {
        "tries": 2, "pass_at_k": 0.667, "pass_hat_k": 0.333, "per_case": {"a": "1/2", "b": "2/2", "c": "0/2"}}
    assert run.consistency({"a": ok}) is None
    assert run.passed_case(_row(rubric=True)) and not run.passed_case(_row(rubric=False))
    assert not run.passed_case({"error": "boom"})


def test_compare_reports_only_numbers_that_changed():
    lines = run.compare({"faithfulness": 0.8, "words": 800, "tokens": 5}, {"faithfulness": 0.9, "words": 800, "tokens": 7})
    assert len(lines) == 2
    assert any("faithfulness" in line and "+0.1" in line for line in lines)
    assert not any("words" in line for line in lines)


JUDGED_CATEGORIES = {"fictional", "false_premise", "true_premise", "real_topic"}


def test_the_case_file_is_consistent_and_balanced():
    cases = run.load_cases()
    assert len(cases) == len({case["id"] for case in cases}) >= 8
    for case in cases:
        assert case["language"] in LANGUAGES and len(case["topic"]) >= 3
        graded = "criteria" if case["category"] in JUDGED_CATEGORIES else "key_points"
        assert case.get(graded) and all(isinstance(item, str) and item for item in case[graded]), case["id"]
    # an eval that only checks that the agent refuses or corrects would reward an agent that always does
    categories = {case["category"] for case in cases}
    assert {"fictional", "false_premise"} <= categories and {"real_topic", "true_premise"} <= categories
    assert {case["id"] for case in run.load_cases(list(run.SMOKE))} == set(run.SMOKE)
    with pytest.raises(SystemExit):
        run.load_cases(["no-such-case"])


def test_without_yes_nothing_is_run(capsys, monkeypatch):
    monkeypatch.setattr(run, "run_case", lambda *a, **k: pytest.fail("should not run"))
    assert run.main(["--smoke"]) == 0
    printed = capsys.readouterr().out
    assert "Nothing was run" in printed and "rag-en" in printed

    assert run.main(["--smoke", "--trials", "2"]) == 0
    printed = capsys.readouterr().out
    assert "3 case(s) x 2 trial(s)" in printed
    assert f"{6 * (2 * run.TOKENS_PER_ANALYST + run.JUDGE_TOKENS_PER_CASE):,}" in printed


def test_the_plan_counts_the_most_tavily_credits_a_run_can_use(capsys):
    english = {"language": "en"}
    turkish = {"language": "tr"}
    per_query = run.TAVILY_CREDITS_PER_QUERY[run.SEARCH_DEPTH]
    assert run.tavily_credits(english, 2) == 2 * run.INTERVIEW_TURNS * per_query
    assert run.tavily_credits(turkish, 2) == 2 * run.tavily_credits(english, 2)

    cases = run.load_cases(list(run.SMOKE))
    assert run.main(["--smoke", "--trials", "2"]) == 0
    expected = 2 * sum(run.tavily_credits(case, 2) for case in cases)
    assert f"plus up to {expected} Tavily credits" in capsys.readouterr().out


def test_a_try_whose_web_search_ran_dry_is_not_scored(llm, monkeypatch):
    from src.utils.tools import SearchCreditsExhausted

    def dry(query, *args, **kwargs):
        raise SearchCreditsExhausted("Error 432: usage limit")

    monkeypatch.setattr("src.utils.nodes.tavily_search", dry)
    with pytest.raises(run.OutOfCredit, match="Tavily"):
        run.run_case({"id": "demo", "category": "technical", "language": "en", "topic": "A fake topic"}, analysts=1)
    assert run.stop_reason(run.OutOfCredit("the Tavily search credits ran out")) == "the Tavily search credits ran out"
    assert run.stop_reason(OUT_OF_CREDIT) == "the OpenAI account has no credit left"
    assert run.stop_reason(AuthenticationError("bad key")) == "the OpenAI key was rejected"
    assert run.stop_reason(RuntimeError("boom")) is None and run.stop_reason(RateLimitError("slow down")) is None


def test_a_run_stops_when_the_account_runs_dry_and_says_how_to_finish(tmp_path, monkeypatch, capsys):
    started = []

    def run_case(case, *args, **kwargs):
        started.append(case["id"])
        if len(started) == 2:
            raise OUT_OF_CREDIT
        return _finished(case["id"])

    monkeypatch.setattr(run, "run_case", run_case)
    monkeypatch.setattr(run, "make_judge", lambda model: None)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "set-for-the-test")
    monkeypatch.setenv("TAVILY_API_KEY", "set-for-the-test")

    assert run.main(["--cases", "rag-en,fictional-en,false-premise-en", "--yes", "--no-judge", "--parallel", "1"]) == 1
    printed = capsys.readouterr().out
    assert started == ["rag-en", "fictional-en"]  # the third was never started
    assert "Stopped early: the OpenAI account has no credit left" in printed
    assert "--cases fictional-en,false-premise-en --yes" in printed and "set-for-the-test" not in printed
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text(encoding="utf-8"))
    assert summary["stopped"] == "the OpenAI account has no credit left"
    assert summary["cases"]["rag-en"].get("error") is None
    assert summary["cases"]["fictional-en"]["error"].startswith("stopped")
    assert summary["cases"]["false-premise-en"]["error"].startswith("not started")


def _finished(case_id):
    report = "# T\n\n## Introduction\n\nA [1].\n\n---\n\nB [1].\n\n---\n\nC [1].\n\n## Sources\n\n1. [a](https://a.example)\n"
    saved = {"id": case_id, "case_id": case_id, "trial": 1, "topic": "t", "language": "en", "category": "technical",
             "analysts": 1, "usage": {"total_tokens": 1, "elapsed": 1.0}, "report": report,
             "sources": [{"n": 1, "url": "https://a.example", "title": "a", "kind": "web"}], "source_texts": {},
             "warnings": [], "review_notes": []}
    return run.score(saved, {}, {}, None, 24)


def test_a_case_runs_end_to_end_and_is_scored(llm):
    case = {"id": "demo", "category": "fictional", "language": "en", "topic": "A fake topic",
            "criteria": ["The report must say the topic is unconfirmed."], "key_points": ["A point about the topic."]}
    result = run.run_case(case, analysts=2, judge_llm=ScriptedJudge())
    assert result["report"].startswith("# Big Title")
    assert result["usage"]["total_tokens"] > 0 and result["metrics"]["sources"] >= 1
    assert result["checks"]["no_dangling_citations"] and result["checks"]["three_part_structure"]
    assert result["rubric"]["passed"] is False and result["coverage"]["covered"] == 0
    assert "claims" in result and "judge_usage" in result and "numbers" in result
    row = run.slim(result)
    assert "rows" not in row["claims"] and "rows" not in row["coverage"] and "criteria" not in row["rubric"]
    json.dumps(result)


def test_a_later_try_is_its_own_isolated_run_with_its_own_id(llm):
    case = {"id": "demo", "category": "technical", "language": "en", "topic": "A fake topic"}
    second = run.run_case(case, analysts=1, trial=2)
    assert second["id"] == "demo#2" and second["case_id"] == "demo" and second["trial"] == 2
    assert run.run_case(case, analysts=1)["id"] == "demo"


def test_saved_reports_can_be_scored_again_without_running_the_agent(llm):
    case = {"id": "demo", "category": "fictional", "language": "en", "topic": "A fake topic", "criteria": ["unused"]}
    saved = json.loads(json.dumps(run.run_case(case, analysts=1)))
    assert saved["source_texts"] and "claims" not in saved and "numbers" in saved

    again = run.rescore(saved, ScriptedJudge())
    assert again["report"] == saved["report"] and again["usage"] == saved["usage"]
    assert "claims" in again and "judge_usage" in again
    assert run.rescore(saved, None).get("claims") is None


def test_rescoring_a_folder_writes_a_new_run_and_costs_nothing_without_a_judge(llm, tmp_path, monkeypatch, capsys):
    case = next(c for c in run.load_cases() if c["id"] == "rag-en")
    saved = json.loads(json.dumps(run.run_case(case, analysts=1)))
    source = tmp_path / "earlier"
    source.mkdir()
    (source / "rag-en.json").write_text(json.dumps(saved), encoding="utf-8")
    (source / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")

    assert run.main(["--rescore", str(source), "--no-judge", "--label", "again"]) == 0
    out = next((tmp_path / "results").iterdir())
    assert out.name.endswith("-rescore-again")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["cases"]) == ["rag-en"] and summary["judge"] is False
    assert summary["judge_version"] is None and "check_pass_rates" in summary
    assert "Change vs baseline" not in capsys.readouterr().out

    monkeypatch.setattr(run, "make_judge", lambda model: pytest.fail("a judge-free rescore must not build a judge"))
    assert run.main(["--rescore", str(source)]) == 0
    assert "Nothing was run" in capsys.readouterr().out


def test_review_notes_are_counted_in_the_summary():
    notes = [{"claim": "a", "problem": "b", "fixed": True}, {"claim": "c", "problem": "d", "fixed": False}]
    result = {**_finished("demo"), "review_notes": notes, "warnings": [{"code": "review_failed"}]}
    row = run.slim(result)
    assert row["review"] == {"issues": 2, "unfixed": 1, "failed": 1}

    agg = run.aggregate([row, run.slim(_finished("other"))])
    assert (agg["review_issues"], agg["review_unfixed"], agg["review_failed"]) == (2, 1, 1)
    assert run.aggregate([_row()])["review_issues"] is None


def test_cases_given_to_a_rescore_or_a_canary_include_every_try_of_them(tmp_path, capsys):
    for name in ("demo", "demo#2", "other"):
        saved = {**_finished(name.split("#")[0]), "id": name, "source_texts": {"1": {"content": "x"}}}
        (tmp_path / f"{name}.json").write_text(json.dumps(saved), encoding="utf-8")
    assert {s["id"] for s in run.load_saved(str(tmp_path), ["demo"])} == {"demo", "demo#2"}
    assert canary.main([str(tmp_path), "--cases", "demo"]) == 0
    assert "2 report(s)" in capsys.readouterr().out


def test_a_baseline_graded_by_another_judge_version_is_flagged(llm, tmp_path, monkeypatch, capsys):
    case = next(c for c in run.load_cases() if c["id"] == "rag-en")
    saved = json.loads(json.dumps(run.run_case(case, analysts=1)))
    source = tmp_path / "earlier"
    source.mkdir()
    (source / "rag-en.json").write_text(json.dumps(saved), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"label": "old", "judge": True, "judge_version": "00000000",
                                    "cases": {"rag-en": _row()}}), encoding="utf-8")
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run, "make_judge", lambda model: ScriptedJudge())
    monkeypatch.setenv("OPENAI_API_KEY", "set-for-the-test")

    assert run.main(["--rescore", str(source), "--yes", "--baseline", str(baseline)]) == 0
    printed = capsys.readouterr().out
    assert "graded by judge version 00000000" in printed and judge.VERSION in printed
    assert "set-for-the-test" not in printed


def test_a_case_without_a_judge_skips_model_graded_fields(llm):
    result = run.run_case({"id": "x", "category": "technical", "language": "tr", "topic": "Bir konu"}, analysts=1)
    assert "claims" not in result and "rubric" not in result and "coverage" not in result
    assert "numbers" in result


# --- the judge calibration -----------------------------------------------------------------------------

CANARY_REPORT = (
    "# T\n\n## Introduction\n\n"
    "The method increases accuracy by 40 percent on the benchmark [1]. "
    "The tool is available in many countries around the world [1].\n\n## Sources\n\n1. a\n2. b\n"
)
CANARY_TEXTS = {
    "1": {"title": "A", "url": "https://a.example/1", "content": (
        "The method increases accuracy by 40 percent on the benchmark. "
        "The tool is available in many countries around the world.")},
    "2": {"title": "B", "url": "https://b.example/2", "content": "Penguins live in cold southern regions and eat fish."},
}


def _canary_saved(language="en"):
    return {"id": "demo", "language": language, "report": CANARY_REPORT, "source_texts": CANARY_TEXTS}


def test_each_alteration_makes_a_claim_the_sources_do_not_back():
    texts = {int(n): text for n, text in CANARY_TEXTS.items()}
    claim = {"id": 1, "text": "The method increases accuracy by 40 percent on the benchmark.", "cites": [1]}
    assert canary.flip(claim, texts)["text"] == "The method decreases accuracy by 40 percent on the benchmark."
    assert canary.swap_number(claim, texts)["text"] == "The method increases accuracy by 64 percent on the benchmark."
    assert canary.misattribute(claim, texts)["cites"] == [2]
    assert canary.misattribute({**claim, "cites": [1, 2]}, texts) is None  # nothing else left to cite
    negated = canary.flip({"id": 2, "text": "Semantic search can miss exact identifiers.", "cites": [1]}, texts)
    assert negated["text"] == "Semantic search cannot miss exact identifiers."
    # a plain "is" is not turned into "is not": inside a subordinate clause that can leave the sentence true
    assert canary.flip({"id": 3, "text": "Hybrid search is common in practice.", "cites": [1]}, texts) is None
    assert canary.swap_number({"id": 4, "text": "No figures in this sentence at all.", "cites": [1]}, texts) is None


def test_variants_hold_every_claim_once_unchanged_and_once_altered_and_are_shuffled_repeatably():
    variants, texts = canary.build_variants(_canary_saved(), per_report=8)
    assert sorted(v["id"] for v in variants) == list(range(1, len(variants) + 1)) and set(texts) == {1, 2}
    assert [v["kind"] for v in variants].count("control") == 2 and len(variants) == 4
    assert all(v["text"] == v["original"] for v in variants if v["kind"] == "control")
    assert all(v["text"] != v["original"] for v in variants if v["kind"] in ("number_swapped", "flipped"))
    assert all(v["cites"] == [2] for v in variants if v["kind"] == "misattributed")
    assert variants == canary.build_variants(_canary_saved(), per_report=8)[0]
    turkish = canary.build_variants(_canary_saved("tr"), per_report=8)[0]
    assert "flipped" not in {v["kind"] for v in turkish}


def test_what_a_verdict_means_depends_on_whether_the_claim_was_altered():
    assert canary.outcome("control", "supported") == "ok" and canary.outcome("control", "partial") == "ok"
    assert canary.outcome("control", "unverified") == "alarm" and canary.outcome("control", "unknown") == "abstained"
    assert canary.outcome("flipped", "contradicted") == "caught" and canary.outcome("flipped", "unverified") == "caught"
    assert canary.outcome("flipped", "partial") == "soft" and canary.outcome("flipped", "supported") == "missed"
    # a sentence about the report itself has no truth value, altered or not, and "unknown" is not a guess
    for kind in ("control", "flipped"):
        assert canary.outcome(kind, "not_a_claim") == "abstained" and canary.outcome(kind, "unknown") == "abstained"


class AlwaysSupported:
    """A careless judge: every claim is supported, with the claim itself as its quote or with no quote."""

    def __init__(self, quote_the_claim: bool):
        self.quote_the_claim = quote_the_claim

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            claims = re.findall(r"^\[c(\d+)\] (.*?) \(cites", messages[0].content, re.M)
            return judge.ClaimVerdicts(verdicts=[
                judge.ClaimVerdict(id=int(i), quote=text if self.quote_the_claim else "", reasoning="x",
                                   states_absence=not self.quote_the_claim, verdict="supported")
                for i, text in claims
            ])

        return RunnableLambda(respond)


def test_the_quote_check_catches_what_a_careless_judge_would_wave_through():
    careless = canary.run_canary([_canary_saved()], AlwaysSupported(quote_the_claim=True))
    assert careless["tally"]["control"]["false_alarm_rate"] == 0.0
    altered = [row for row in careless["rows"] if row["kind"] != "control"]
    assert altered and all(row["outcome"] == "caught" and row["verdict"] == "unverified" for row in altered)


class RecordingJudge(AlwaysSupported):
    def __init__(self):
        super().__init__(quote_the_claim=True)
        self.batches = []

    def with_structured_output(self, schema, **kwargs):
        inner = super().with_structured_output(schema, **kwargs)

        def respond(messages):
            self.batches.append([text for _, text in re.findall(r"^\[c(\d+)\] (.*?) \(cites", messages[0].content, re.M)])
            return inner.invoke(messages)

        return RunnableLambda(respond)


def test_a_claim_and_its_altered_copy_are_never_judged_in_the_same_batch():
    recorder = RecordingJudge()
    result = canary.run_canary([_canary_saved()], recorder)
    controls = {row["text"] for row in result["rows"] if row["kind"] == "control"}
    misattributed = [row for row in result["rows"] if row["kind"] == "misattributed"]
    assert misattributed and misattributed[0]["text"] in controls  # the same sentence, only the citation differs
    controls_batch, altered_batch = recorder.batches[:2]  # controls in one call, altered claims in the next
    assert len(controls_batch) == len(altered_batch) == 2
    assert all(len(set(batch)) == len(batch) for batch in recorder.batches)


def test_a_misattributed_claim_prefers_a_source_from_another_report():
    foreign = {1001: {"title": "X", "url": "https://x.example", "content": "Volcanoes erupt when magma rises."}}
    variants, texts = canary.build_variants(_canary_saved(), per_report=8, extra=foreign)
    misattributed = [v for v in variants if v["kind"] == "misattributed"]
    assert misattributed and all(v["cites"] == [1001] for v in misattributed) and 1001 in texts


def test_a_judge_that_says_nothing_checkable_is_shown_up_as_missing_everything():
    lazy = canary.run_canary([_canary_saved()], AlwaysSupported(quote_the_claim=False))
    altered = [row for row in lazy["rows"] if row["kind"] != "control"]
    assert altered and all(row["outcome"] == "missed" for row in altered)
    recalls = [entry["recall"] for kind, entry in lazy["tally"].items() if kind != "control" and entry["n"]]
    assert recalls and set(recalls) == {0.0}
    assert lazy["judge_version"] == judge.VERSION and lazy["usage"]["total_tokens"] >= 0


def test_the_canary_cli_plans_first_and_reads_saved_reports(tmp_path, capsys, monkeypatch):
    (tmp_path / "demo.json").write_text(json.dumps(_canary_saved()), encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(canary, "make_judge", lambda model: pytest.fail("a plan-only run must not build a judge"))
    assert canary.main([str(tmp_path)]) == 0
    assert "Nothing was run" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        canary.main([str(tmp_path / "missing")])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        canary.main([str(empty)])
