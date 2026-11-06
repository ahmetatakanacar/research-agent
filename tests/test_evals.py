import json
import re

import pytest
from langchain_core.runnables import RunnableLambda

from evals import judge, metrics, run
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
    measured = metrics.evaluate_report(GOOD_REPORT, GOOD_SOURCES)
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


class ScriptedJudge:
    def __init__(self, skip=(), fail=False):
        self.prompts, self.skip, self.fail = [], set(skip), fail

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            prompt = messages[0].content
            self.prompts.append(prompt)
            if self.fail:
                raise RuntimeError("scripted judge failure")
            if schema is judge.ClaimVerdicts:
                texts = dict(re.findall(r"^\[c(\d+)\] (.*?) \(cites", prompt, re.M))
                return judge.ClaimVerdicts(verdicts=[
                    judge.ClaimVerdict(
                        id=int(i), verdict="unsupported" if "INVENTED" in text else "supported", note="scripted")
                    for i, text in texts.items() if int(i) not in self.skip
                ])
            return judge.RubricVerdict(passed="SAYS-UNCONFIRMED" in prompt.split("REPORT:")[-1], reason="scripted")

        return RunnableLambda(respond)


def _claims(count, invented=()):
    return [
        {"id": i, "text": ("INVENTED " if i in invented else "") + f"Claim {i} is long enough to be judged.", "cites": [1]}
        for i in range(1, count + 1)
    ]


TEXTS = {1: {"title": "Doc", "url": "https://example.com", "content": "Source text."}}


def test_claims_are_judged_in_batches_and_scored():
    scripted = ScriptedJudge()
    result = judge.judge_claims(scripted, _claims(20, invented={3, 4}), TEXTS)
    assert len(scripted.prompts) == 3
    assert (result["claims"], result["supported"], result["unsupported"]) == (20, 18, 2)
    assert result["faithfulness"] == 0.9 and result["errors"] == 0
    assert "Source text." in scripted.prompts[0]


def test_partial_verdicts_count_half_and_unjudged_and_non_claims_are_left_out():
    rows = [{"verdict": v} for v in ("supported", "partial", "unsupported", "unjudged", "not_a_claim")]
    scored = judge.faithfulness(rows)
    assert scored["judged"] == 3 and scored["not_a_claim"] == 1 and scored["faithfulness"] == 0.5
    assert judge.faithfulness([])["faithfulness"] is None

    result = judge.judge_claims(ScriptedJudge(skip={2}), _claims(3), TEXTS)
    assert result["unjudged"] == 1 and result["judged"] == 2


def test_a_failing_judge_does_not_crash_the_run():
    result = judge.judge_claims(ScriptedJudge(fail=True), _claims(3), TEXTS)
    assert result["errors"] == 1 and result["judged"] == 0 and result["faithfulness"] is None
    assert judge.judge_rubric(ScriptedJudge(fail=True), "report", "rubric")["passed"] is None


def test_the_rubric_is_graded_against_the_report():
    assert judge.judge_rubric(ScriptedJudge(), "It SAYS-UNCONFIRMED clearly.", "rubric")["passed"] is True
    assert judge.judge_rubric(ScriptedJudge(), "It invents things.", "rubric")["passed"] is False


def _row(words=800, authority=0.5, failed=(), claims=None, rubric=None, tokens=1000):
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
    return row


def test_aggregate_summarises_checks_faithfulness_and_honesty():
    rows = [
        _row(claims={"judged": 10, "supported": 8, "partial": 2, "unsupported": 0, "contradicted": 0}, rubric=True),
        _row(failed=("groups_sorted",), claims={"judged": 10, "supported": 6, "partial": 0, "unsupported": 3, "contradicted": 1},
             rubric=False),
        {"error": "RuntimeError: boom"},
    ]
    agg = run.aggregate(rows)
    assert (agg["cases"], agg["errors"]) == (3, 1)
    assert agg["faithfulness"] == pytest.approx((9 + 6) / 20)
    assert (agg["unsupported_claims"], agg["contradicted_claims"]) == (3, 1)
    assert agg["honesty_pass_rate"] == 0.5
    assert agg["hard_checks_passed"] == pytest.approx(1 - 1 / (2 * len(metrics.HARD_CHECKS)), abs=0.001)
    assert agg["tokens"] == 2000


def test_aggregate_without_a_judge_leaves_model_graded_fields_empty():
    agg = run.aggregate([_row()])
    assert agg["faithfulness"] is None and agg["honesty_pass_rate"] is None


def test_compare_reports_only_numbers_that_changed():
    lines = run.compare({"faithfulness": 0.8, "words": 800, "tokens": 5}, {"faithfulness": 0.9, "words": 800, "tokens": 7})
    assert len(lines) == 2
    assert any("faithfulness" in line and "+0.1" in line for line in lines)
    assert not any("words" in line for line in lines)


def test_the_case_file_is_consistent():
    cases = run.load_cases()
    assert len(cases) == len({case["id"] for case in cases}) >= 8
    for case in cases:
        assert case["language"] in LANGUAGES and len(case["topic"]) >= 3
        if case["category"] in {"fictional", "false_premise"}:
            assert case.get("rubric"), case["id"]
    assert {case["id"] for case in run.load_cases(list(run.SMOKE))} == set(run.SMOKE)
    with pytest.raises(SystemExit):
        run.load_cases(["no-such-case"])


def test_without_yes_nothing_is_run(capsys, monkeypatch):
    monkeypatch.setattr(run, "run_case", lambda *a, **k: pytest.fail("should not run"))
    assert run.main(["--smoke"]) == 0
    printed = capsys.readouterr().out
    assert "Nothing was run" in printed and "rag-en" in printed


def test_a_case_runs_end_to_end_and_is_scored(llm):
    case = {"id": "demo", "category": "fictional", "language": "en", "topic": "A fake topic",
            "rubric": "The report must say the topic is unconfirmed."}
    result = run.run_case(case, analysts=2, judge_llm=ScriptedJudge())
    assert result["report"].startswith("# Big Title")
    assert result["usage"]["total_tokens"] > 0 and result["metrics"]["sources"] >= 1
    assert result["checks"]["no_dangling_citations"] and result["checks"]["three_part_structure"]
    assert result["rubric"]["passed"] is False
    assert "claims" in result and "judge_usage" in result
    row = run.slim(result)
    assert "rows" not in row["claims"]
    json.dumps(result)


def test_saved_reports_can_be_scored_again_without_running_the_agent(llm):
    case = {"id": "demo", "category": "fictional", "language": "en", "topic": "A fake topic", "rubric": "unused"}
    saved = json.loads(json.dumps(run.run_case(case, analysts=1)))
    assert saved["source_texts"] and "claims" not in saved

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
    assert "Change vs baseline" not in capsys.readouterr().out

    monkeypatch.setattr(run, "make_judge", lambda model: pytest.fail("a judge-free rescore must not build a judge"))
    assert run.main(["--rescore", str(source)]) == 0
    assert "Nothing was run" in capsys.readouterr().out


def test_a_case_without_a_judge_skips_model_graded_fields(llm):
    result = run.run_case({"id": "x", "category": "technical", "language": "tr", "topic": "Bir konu"}, analysts=1)
    assert "claims" not in result and "rubric" not in result
