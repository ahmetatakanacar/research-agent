"""Model-graded checks.

Three kinds, each built on what Anthropic's eval guidance recommends for model graders:

- Claim faithfulness. The judge must copy a word-for-word quote from a cited source before it may call
  a sentence supported, and code checks that the quote really is in that source ("find a supporting quote
  for each claim, retract it if there is none"). It reasons before it gives a verdict and may answer
  "unknown" instead of guessing.
- Criteria (honesty cases). Every criterion is graded by its own isolated judge call, not one call for all.
- Coverage. Every key point a good report must make is graded by its own judge call, and a "covered"
  verdict needs a quote that code finds in the report.
"""

import hashlib
import re
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from evals.claims import extract_claims  # noqa: F401  (re-exported for callers and tests)
from src.utils.sources import strip_sources_section

SOURCE_CHARS = 2500
BATCH_SIZE = 8
MIN_QUOTE_CHARS = 15
GRAM = 4
MIN_QUOTE_COVERAGE = 0.6

UNFAVOURABLE = ("unsupported", "contradicted", "unverified")


class ClaimVerdict(BaseModel):
    id: int = Field(description="The claim id, for example 3 for [c3].")
    quote: str = Field(
        default="",
        description="Word-for-word passage copied from one of the cited sources that backs or contradicts the "
                    "sentence. Empty only if no cited source says anything relevant or the sentence says the "
                    "sources lack something.")
    reasoning: str = Field(
        default="", description="Clause by clause: whether a cited source states each part of the sentence as "
                                "written, including figures, direction words and negations.")
    states_absence: bool = Field(
        default=False, description="True when the sentence itself says the sources do not contain or confirm something.")
    verdict: Literal["supported", "partial", "unsupported", "contradicted", "not_a_claim", "unknown"]


class ClaimVerdicts(BaseModel):
    verdicts: List[ClaimVerdict]


class GradeVerdict(BaseModel):
    quote: str = Field(
        default="", description="Word-for-word passage copied from the report that settles the question, or empty.")
    reasoning: str = Field(description="One or two sentences pointing at the evidence in the report.")
    verdict: Literal["met", "not_met", "unknown"]


CLAIM_INSTRUCTIONS = """You are a strict but fair fact-checker. Below are sentences taken from a report, each with the numbers of the sources it cites, followed by the text of those sources.
Judge every sentence using ONLY the cited sources, never outside knowledge. The goal is to catch facts, figures and statements that the sources do not back, not to demand that the sources use the report's exact words.
For each sentence, work in this order:
1. quote: copy, word for word, the passage from one cited source that best backs or contradicts the sentence. Leave it empty only when no cited source says anything relevant, or when the sentence says that the sources lack something.
2. reasoning: go through the sentence clause by clause. For each clause, including every figure, date, name, direction word (higher or lower, increase or decrease, more or fewer) and negation (not, cannot, no, never), say whether a cited source states it as written. A sentence may draw on several cited sources, so the quote you copy only has to cover its best-backed part. The verdict follows the weakest clause: one clause that a source contradicts makes the whole sentence contradicted, however well the rest matches.
3. states_absence: true only when the sentence itself says the sources do not contain or confirm something.
4. verdict:
- supported: the cited sources state it or clearly imply it. A reasonable paraphrase counts. A recommendation or inference counts when the cited sources give the facts it rests on, even if no source states the exact advice. A sentence saying that the sources do not contain or establish something counts when the cited sources really lack it.
- partial: most of the sentence is backed, but a detail is not stated by the cited sources (a figure, name or date they do not mention), or the sentence overstates (a scope such as "always" or "all", or a stronger claim than the source makes).
- unsupported: a specific claim that the cited sources do not contain.
- contradicted: a cited source says something incompatible with any part of the sentence: the opposite, a different value for a figure, date or name, or a reversed direction. One conflicting detail is enough, even when the rest of the sentence is right.
- not_a_claim: a sentence about the report itself (what it covers or how it is organised) or a transition with no factual content.
- unknown: the source text is missing, cut off or too thin for you to decide. Use this instead of guessing.
A supported or partial verdict is only accepted with a quote that really appears in a cited source.
Return exactly one entry per sentence id.

CLAIMS:
{claims}

SOURCES:
{sources}
"""

GRADE_INSTRUCTIONS = """You grade one requirement against a report. Read the whole report, then decide whether it meets the requirement.
Be strict: the requirement is met only when every part of it is true. The report may be in any language and the requirement may be worded differently from the report.
- quote: copy, word for word, the passage from the report that settles the question. Leave it empty when nothing in the report addresses the requirement, or when the requirement is about something the report must NOT do.
- reasoning: one or two sentences pointing at the evidence.
- verdict: met, not_met, or unknown (use unknown instead of guessing when the report is too unclear to tell).

REQUIREMENT:
{requirement}

REPORT:
{report}
"""

# Anything that changes how verdicts are reached, besides the prompts, goes in the protocol tag.
PROTOCOL = "quote-check by four-word coverage, isolated retry of unverified claims"
VERSION = hashlib.sha1(
    "|".join([CLAIM_INSTRUCTIONS, GRADE_INSTRUCTIONS, str(SOURCE_CHARS), str(BATCH_SIZE), PROTOCOL]).encode("utf-8")
).hexdigest()[:8]


def _fold(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def _grams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + GRAM]) for i in range(len(words) - GRAM + 1)}


def quote_found(quote: str, texts: list[str]) -> bool:
    """True when the quote really comes from one of the texts, ignoring case, punctuation and spacing. An
    ellipsis splits a quote into pieces, and each piece must be found.

    A short piece must appear as it is. A longer one may also be stitched together from several passages
    without an ellipsis, which the judge does with long quotes: it passes when at least MIN_QUOTE_COVERAGE of
    its four-word runs are in the texts. Measured on a full run, real quotes cover 0.67 to 1.0 of their runs
    and a paraphrase of the same sentence at most 0.43, so words that merely recur do not pass."""
    pieces = [piece for piece in (_fold(p) for p in re.split(r"…|\.\.\.", quote or "")) if piece]
    if sum(len(piece) for piece in pieces) < MIN_QUOTE_CHARS:
        return False
    pool = [_fold(text) for text in texts]
    known = set().union(*(_grams(text.split()) for text in pool))

    def found(piece: str) -> bool:
        if any(piece in text for text in pool):
            return True
        grams = _grams(piece.split())
        return bool(grams) and sum(1 for gram in grams if gram in known) / len(grams) >= MIN_QUOTE_COVERAGE

    return all(found(piece) for piece in pieces)


def _claims_block(batch: list[dict]) -> str:
    return "\n".join(
        f"[c{claim['id']}] {claim['text']} (cites: {''.join(f'[{n}]' for n in claim['cites'])})" for claim in batch
    )


def _sources_block(batch: list[dict], texts: dict[int, dict]) -> str:
    numbers = sorted({n for claim in batch for n in claim["cites"] if n in texts})
    return "\n\n---\n\n".join(
        f"[{n}] {texts[n]['title']} — {texts[n]['url']}\n{texts[n]['content'][:SOURCE_CHARS]}" for n in numbers
    )


def _checked(verdict: ClaimVerdict, claim: dict, texts: dict[int, dict]) -> tuple[str, bool]:
    """The verdict that counts, and whether its quote was found. A favourable verdict needs a quote that is
    really in a cited source (a sentence that states an absence has nothing to quote)."""
    cited = [texts[n]["content"] for n in claim["cites"] if n in texts]
    found = quote_found(verdict.quote, cited)
    needs_quote = verdict.verdict in ("supported", "partial") and not verdict.states_absence
    return ("unverified" if needs_quote and not found else verdict.verdict), found


def _ask(checker, config, batch: list[dict], texts: dict[int, dict]) -> dict[int, ClaimVerdict]:
    prompt = CLAIM_INSTRUCTIONS.format(claims=_claims_block(batch), sources=_sources_block(batch, texts))
    result = checker.invoke([SystemMessage(content=prompt)], config=config)
    wanted = {claim["id"] for claim in batch}
    return {verdict.id: verdict for verdict in result.verdicts if verdict.id in wanted}


def _row(claim: dict, verdict: ClaimVerdict | None, texts: dict[int, dict], retried: bool = False) -> dict:
    row = {"id": claim["id"], "text": claim["text"], "cites": claim["cites"], "verdict": "unjudged",
           "judge_verdict": "", "quote": "", "quote_found": False, "note": "", "retried": retried}
    if verdict:
        counted, found = _checked(verdict, claim, texts)
        row.update(verdict=counted, judge_verdict=verdict.verdict, quote=verdict.quote,
                   quote_found=found, note=verdict.reasoning)
    return row


def judge_claims(llm, claims: list[dict], texts: dict[int, dict], callbacks: list | None = None) -> dict:
    """Judge claims in small batches. Returns the verdict per claim plus counts.

    A batch shows the judge the sources of all its claims together, so the judge can quote a source that a
    sentence does not cite. A claim whose quote is not in its own sources is therefore asked about again, alone
    and with only its own sources: that tells a judge that wandered off apart from a sentence whose cited
    sources really do not say it."""
    checker = llm.with_structured_output(ClaimVerdicts, method="function_calling")
    config = {"callbacks": callbacks or []}
    judged: dict[int, ClaimVerdict] = {}
    errors = 0
    for start in range(0, len(claims), BATCH_SIZE):
        try:
            judged.update(_ask(checker, config, claims[start:start + BATCH_SIZE], texts))
        except Exception:
            errors += 1

    rows = [_row(claim, judged.get(claim["id"]), texts) for claim in claims]
    for index, (claim, row) in enumerate(zip(claims, rows, strict=True)):
        if row["verdict"] != "unverified":
            continue
        try:
            again = _ask(checker, config, [claim], texts).get(claim["id"])
        except Exception:
            errors += 1
            continue
        if again:
            rows[index] = _row(claim, again, texts, retried=True)
    return {"rows": rows, "errors": errors, **faithfulness(rows)}


def faithfulness(rows: list[dict]) -> dict:
    names = ("supported", "partial", "unsupported", "contradicted", "unverified", "not_a_claim", "unknown", "unjudged")
    counts = {name: sum(1 for row in rows if row["verdict"] == name) for name in names}
    judged = len(rows) - counts["unjudged"] - counts["not_a_claim"] - counts["unknown"]
    score = (counts["supported"] + 0.5 * counts["partial"]) / judged if judged else None
    return {"claims": len(rows), "judged": judged, **counts, "faithfulness": None if score is None else round(score, 3)}


def _grade(llm, report: str, requirement: str, callbacks: list | None, need_quote: bool) -> dict:
    grader = llm.with_structured_output(GradeVerdict, method="function_calling")
    row = {"requirement": requirement, "verdict": "error", "quote": "", "reason": ""}
    try:
        result = grader.invoke(
            [SystemMessage(content=GRADE_INSTRUCTIONS.format(requirement=requirement, report=report)),
             HumanMessage(content="Grade the report.")],
            config={"callbacks": callbacks or []},
        )
    except Exception as exc:
        row["reason"] = f"judge failed: {type(exc).__name__}"
        return row
    verdict = result.verdict
    if need_quote and verdict == "met" and not quote_found(result.quote, [report]):
        verdict = "unverified"
    row.update(verdict=verdict, quote=result.quote, reason=result.reasoning)
    return row


def judge_criteria(llm, report: str, criteria: list[str], callbacks: list | None = None) -> dict:
    """Every criterion gets its own judge call. The case passes only when all are met; one that is not met
    fails it, and a judge that failed or answered unknown leaves the result open (None)."""
    body = strip_sources_section(report)
    rows = [_grade(llm, body, criterion, callbacks, need_quote=False) for criterion in criteria]
    failed = [row for row in rows if row["verdict"] == "not_met"]
    open_ = [row for row in rows if row["verdict"] in ("unknown", "error")]
    passed = False if failed else (None if open_ else True)
    shown = failed or open_ or rows
    return {"passed": passed, "reason": " | ".join(row["reason"] for row in shown if row["reason"]), "criteria": rows}


def judge_coverage(llm, report: str, points: list[str], callbacks: list | None = None) -> dict:
    """Does the report make each key point? A point counts as covered only with a quote found in the report."""
    body = strip_sources_section(report)
    rows = [_grade(llm, body, f"The report states, in any wording or language: {point}", callbacks, need_quote=True)
            for point in points]
    for row, point in zip(rows, points, strict=True):
        row["point"] = point
    graded = [row for row in rows if row["verdict"] != "error"]
    covered = sum(1 for row in graded if row["verdict"] == "met")
    return {
        "points": len(points), "judged": len(graded), "covered": covered,
        "unverified": sum(1 for row in graded if row["verdict"] == "unverified"),
        "unknown": sum(1 for row in graded if row["verdict"] == "unknown"),
        "coverage": round(covered / len(graded), 3) if graded else None, "rows": rows,
    }
