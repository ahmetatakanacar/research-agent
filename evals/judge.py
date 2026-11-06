"""Model-graded checks: does each cited claim hold up against the sources it cites, and does the
report meet a case-specific honesty rubric?"""

import re
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.utils.sources import CITE_RE, cited_numbers, strip_sources_section

MIN_CLAIM_CHARS = 30
SOURCE_CHARS = 2500
BATCH_SIZE = 8


class ClaimVerdict(BaseModel):
    id: int = Field(description="The claim id, for example 3 for [c3].")
    verdict: Literal["supported", "partial", "unsupported", "contradicted", "not_a_claim"]
    note: str = Field(default="", description="One short sentence on what the cited sources say or lack.")


class ClaimVerdicts(BaseModel):
    verdicts: List[ClaimVerdict]


class RubricVerdict(BaseModel):
    passed: bool = Field(description="True only when the report satisfies the whole rubric.")
    reason: str = Field(description="One or two sentences pointing at the evidence in the report.")


CLAIM_INSTRUCTIONS = """You are a strict but fair fact-checker. Below are sentences taken from a report, each with the numbers of the sources it cites, followed by the text of those sources.
Judge every sentence using ONLY the cited sources, never outside knowledge. The goal is to catch facts, figures and statements that the sources do not back, not to demand that the sources use the report's exact words.
- supported: the cited sources state it or clearly imply it. A reasonable paraphrase counts. A recommendation or inference counts when the cited sources give the facts it rests on, even if no source states the exact advice. A sentence saying that the sources do not contain or establish something counts when the cited sources really lack it.
- partial: part of a specific claim is supported but another specific part is not (a figure, a name, a scope such as "always" or "all"), or it is clearly overstated.
- unsupported: a specific claim that the cited sources do not contain.
- contradicted: a cited source says the opposite.
- not_a_claim: a sentence about the report itself (what it covers or how it is organised) or a transition with no factual content.
Return exactly one verdict per sentence id, with a one-sentence note.

CLAIMS:
{claims}

SOURCES:
{sources}
"""

RUBRIC_INSTRUCTIONS = """You grade a report against a rubric. Read the whole report, then decide whether it satisfies the rubric.
Be strict: the rubric passes only when every part of it is met. The report may be in any language.

RUBRIC:
{rubric}

REPORT:
{report}
"""


def _sentences(text: str) -> list[str]:
    text = re.sub(r"^#{1,6}.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    return re.split(r"(?<=[.!?])\s+(?=[\"“(A-ZÇĞİÖŞÜ0-9])", text)


def extract_claims(report: str, max_claims: int = 24) -> list[dict]:
    """Sentences that carry a citation, with the markers removed. Over the cap, an evenly spaced
    sample is kept so the choice is repeatable."""
    claims = []
    for sentence in _sentences(strip_sources_section(report)):
        cites = sorted(cited_numbers(sentence))
        text = re.sub(r"\s+([,.;:!?])", r"\1", CITE_RE.sub("", sentence))
        text = re.sub(r"\s+", " ", text).strip()
        if cites and len(text) >= MIN_CLAIM_CHARS:
            claims.append({"text": text, "cites": cites})
    if len(claims) > max_claims:
        claims = [claims[int(i * len(claims) / max_claims)] for i in range(max_claims)]
    for index, claim in enumerate(claims, 1):
        claim["id"] = index
    return claims


def _claims_block(batch: list[dict]) -> str:
    return "\n".join(
        f"[c{claim['id']}] {claim['text']} (cites: {''.join(f'[{n}]' for n in claim['cites'])})" for claim in batch
    )


def _sources_block(batch: list[dict], texts: dict[int, dict]) -> str:
    numbers = sorted({n for claim in batch for n in claim["cites"] if n in texts})
    return "\n\n---\n\n".join(
        f"[{n}] {texts[n]['title']} — {texts[n]['url']}\n{texts[n]['content'][:SOURCE_CHARS]}" for n in numbers
    )


def judge_claims(llm, claims: list[dict], texts: dict[int, dict], callbacks: list | None = None) -> dict:
    """Judge claims in small batches. Returns the verdict per claim plus counts."""
    checker = llm.with_structured_output(ClaimVerdicts, method="function_calling")
    config = {"callbacks": callbacks or []}
    judged: dict[int, ClaimVerdict] = {}
    errors = 0
    for start in range(0, len(claims), BATCH_SIZE):
        batch = claims[start:start + BATCH_SIZE]
        prompt = CLAIM_INSTRUCTIONS.format(claims=_claims_block(batch), sources=_sources_block(batch, texts))
        try:
            result = checker.invoke([SystemMessage(content=prompt)], config=config)
        except Exception:
            errors += 1
            continue
        wanted = {claim["id"] for claim in batch}
        for verdict in result.verdicts:
            if verdict.id in wanted:
                judged[verdict.id] = verdict

    rows = []
    for claim in claims:
        verdict = judged.get(claim["id"])
        rows.append({
            "id": claim["id"], "text": claim["text"], "cites": claim["cites"],
            "verdict": verdict.verdict if verdict else "unjudged", "note": verdict.note if verdict else "",
        })
    return {"rows": rows, "errors": errors, **faithfulness(rows)}


def faithfulness(rows: list[dict]) -> dict:
    counts = {name: sum(1 for row in rows if row["verdict"] == name)
              for name in ("supported", "partial", "unsupported", "contradicted", "not_a_claim", "unjudged")}
    judged = len(rows) - counts["unjudged"] - counts["not_a_claim"]
    score = (counts["supported"] + 0.5 * counts["partial"]) / judged if judged else None
    return {"claims": len(rows), "judged": judged, **counts, "faithfulness": None if score is None else round(score, 3)}


def judge_rubric(llm, report: str, rubric: str, callbacks: list | None = None) -> dict:
    grader = llm.with_structured_output(RubricVerdict, method="function_calling")
    try:
        result = grader.invoke(
            [SystemMessage(content=RUBRIC_INSTRUCTIONS.format(rubric=rubric, report=strip_sources_section(report))),
             HumanMessage(content="Grade the report.")],
            config={"callbacks": callbacks or []},
        )
    except Exception as exc:
        return {"passed": None, "reason": f"judge failed: {type(exc).__name__}"}
    return {"passed": bool(result.passed), "reason": result.reason}
