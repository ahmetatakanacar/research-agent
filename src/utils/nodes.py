import logging
from itertools import zip_longest

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import get_buffer_string
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from src.settings import SEARCH_CANDIDATES, SEARCH_DEPTH, SEARCH_RESULTS
from src.utils.figures import figure_issues
from src.utils.languages import label, language_instruction, language_name, normalize_language
from src.utils.models import get_fast_llm, get_llm
from src.utils.objects import Analyst, Perpectives, ReviewResult, SearchQuery, SourcePicks
from src.utils.prompts import (
    analyst_instructions,
    answer_instructions,
    current_analysts_block,
    intro_conclusion_instructions,
    question_instructions,
    report_writer_instructions,
    review_instructions,
    revise_instructions,
    search_instructions,
    section_writer_instructions,
    source_selection_instructions,
)
from src.utils.ranking import host_of, quality_bonus
from src.utils.sources import (
    CITE_RE,
    cited_numbers,
    finalize_citations,
    format_numbered,
    merge_sections,
    sources_markdown,
    strip_sources_section,
    tidy_citations,
    unique_documents,
)
from src.utils.states import GenerateAnalystState, InterviewState, ResearchGraphState
from src.utils.tools import SearchCreditsExhausted, tavily_search, wikipedia_search

load_dotenv()

logger = logging.getLogger(__name__)

APPROVAL_WORDS = {"perfect", "continue", "approved", "yes", "ok", "okay", "tamam", "evet", "onay"}
REVIEW_SOURCE_CHARS = 60_000
REVIEW_MAX_ISSUES = 20
MAX_CITATIONS_PER_STATEMENT = 3
CITATION_REMINDER = (
    ". Every claim you restate from the sections must keep its citation marker, such as [1],"
    " exactly as it appears there."
)


def text_of(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, (str, dict))
        )
    return str(content or "")


def _writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda _payload: None


def _analyst(value) -> Analyst:
    return Analyst.model_validate(value) if isinstance(value, dict) else value


def _opt(state, key: str, default=None):
    return (state.get("options") or {}).get(key, default)


def _language(state) -> str:
    return normalize_language(_opt(state, "language"))


def _query(state) -> str:
    return (state.get("search_query") or "").strip()


def create_analyst(state: GenerateAnalystState):

    topic = state["topic"]
    max_analyst = state["max_analyst"]
    language = normalize_language(state.get("language"))
    feedback = state.get("human_analyst_feedback") or ""
    current = [_analyst(a) for a in state.get("analysts") or []]

    current_block = ""
    if feedback and current:
        current_block = current_analysts_block.format(analysts="\n\n".join(a.persona for a in current))

    structured_llm = get_llm().with_structured_output(Perpectives, method="function_calling")
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=feedback,
        current_analysts=current_block,
        max_analyst=max_analyst,
        language_instruction=language_instruction(language),
    )
    result = structured_llm.invoke(
        [SystemMessage(content=system_message), HumanMessage(content="Please generate the set of analyst.")]
    )

    return {"analysts": list(result.analyst)[:max_analyst]}


def human_feedback(state: GenerateAnalystState):
    feedback = interrupt({
        "question": "Are these analysts okay?",
        "analysts": [
            analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
            for analyst in state.get("analysts", [])
        ],
        "instructions": "Return feedback to regenerate analysts, or return empty/perfect/continue to approve.",
    })

    update: dict = {}

    if isinstance(feedback, dict):
        edited = feedback.get("analysts")
        feedback = feedback.get("feedback")
        if edited:
            try:
                update["analysts"] = [Analyst.model_validate(a) for a in edited]
            except Exception:
                logger.warning("Ignoring invalid analyst edits", exc_info=True)

    text = feedback.strip() if isinstance(feedback, str) else ""
    update["human_analyst_feedback"] = None if not text or text.lower() in APPROVAL_WORDS else text
    return update


def generate_question(state: InterviewState):

    analyst = _analyst(state["analyst"])
    language = _language(state)

    if len(state["messages"]) == 1:
        _writer()({"interview": {"analyst": analyst.name, "role": analyst.role}})

    system_message = question_instructions.format(
        goals=analyst.persona,
        closing=label(language, "closing"),
        language_instruction=language_instruction(language),
    )
    question = get_llm().invoke([SystemMessage(content=system_message)] + state["messages"])

    return {"messages": [question]}


def generate_search_query(state: InterviewState):
    language = _language(state)
    structured_llm = get_fast_llm().with_structured_output(SearchQuery, method="function_calling")
    system_message = SystemMessage(content=search_instructions.format(language=language_name(language)))

    result = structured_llm.invoke([system_message] + state["messages"])

    fallback = text_of(state["messages"][-1]).strip()[:200]
    query = (getattr(result, "search_query", "") or "").strip() or fallback
    english = (getattr(result, "english_query", "") or "").strip()
    return {"search_query": query, "english_query": english}


def curate_sources(docs: list, query: str, limit: int) -> list:
    if len(docs) <= limit:
        return docs
    listing = "\n\n".join(
        f"[{i}] {doc['title']} — {host_of(doc['url'])}\n{doc['content'][:300]}" for i, doc in enumerate(docs, 1)
    )
    try:
        picker = get_fast_llm().with_structured_output(SourcePicks, method="function_calling")
        picks = picker.invoke([SystemMessage(content=source_selection_instructions.format(
            query=query, limit=limit, candidates=listing,
        ))])
        order = [n for n in dict.fromkeys(picks.best) if 1 <= n <= len(docs)]
    except Exception:
        logger.warning("Source selection failed; keeping the ranked order", exc_info=True)
        return docs[:limit]

    return [docs[n - 1] for n in order[:limit]] or docs[:limit]


def search_web(state: InterviewState):
    query = _query(state)
    english = (state.get("english_query") or "").strip()
    queries = [query]
    if _language(state) != "en" and english and english.lower() != query.lower():
        queries.append(english)

    found: list = []
    exhausted = False
    for q in queries:
        try:
            found.append(tavily_search(q, SEARCH_DEPTH, SEARCH_CANDIDATES))
        except SearchCreditsExhausted:
            exhausted = True  # the credits can run out between two queries: keep what the earlier ones found
            break
    candidates: list = []
    seen: set[str] = set()
    for doc in (doc for group in zip_longest(*found) for doc in group if doc):
        if doc["url"] not in seen:
            seen.add(doc["url"])
            candidates.append(doc)
    update: dict = {"context": curate_sources(candidates, query, SEARCH_RESULTS)}
    if exhausted:
        # The report says so, once per analyst.
        analyst = _analyst(state["analyst"]).name
        logger.warning("Tavily credits are used up: %s's interview continues without (some) web results", analyst)
        told = any(w.get("code") == "search_credits_exhausted" and w.get("analyst") == analyst
                   for w in state.get("warnings") or [])
        if not told:
            update["warnings"] = [{"code": "search_credits_exhausted", "analyst": analyst}]
    return update


def search_wikipedia(state: InterviewState):
    return {"context": wikipedia_search(_query(state), lang=_language(state))}


def generate_answer(state: InterviewState):

    analyst = _analyst(state["analyst"])
    language = _language(state)
    docs = unique_documents(state.get("context"))

    if not docs:
        return {"messages": [AIMessage(content=label(language, "no_sources"), name="expert")]}

    system_message = answer_instructions.format(
        goals=analyst.persona,
        context=format_numbered(docs),
        language_instruction=language_instruction(language),
    )
    answer = get_llm().invoke([SystemMessage(content=system_message)] + state["messages"])
    answer.name = "expert"

    return {"messages": [answer]}


def save_interview(state: InterviewState):

    interview = get_buffer_string(state["messages"])

    return {"interview": interview}


def write_section(state: InterviewState):

    analyst = _analyst(state["analyst"])
    language = _language(state)
    docs = unique_documents(state.get("context"))

    if not docs:
        text = f"## {analyst.role}\n\n### {label(language, 'summary')}\n\n{label(language, 'no_sources_section')}"
        return {
            "sections": [{"analyst": analyst.name, "text": text, "sources": []}],
            "warnings": [{"code": "no_sources", "analyst": analyst.name}],
        }

    system_message = section_writer_instructions.format(
        focus=analyst.description,
        summary_header=label(language, "summary"),
        language_instruction=language_instruction(language),
    )
    request = (
        f"Interview:\n{state['interview']}\n\n"
        f"Numbered sources:\n{format_numbered(docs)}\n\n"
        "Use them to write your section."
    )
    section = get_llm().invoke([SystemMessage(content=system_message), HumanMessage(content=request)])

    return {"sections": [{"analyst": analyst.name, "text": text_of(section), "sources": docs}]}


def merge_sources(state: ResearchGraphState):
    merged, sources = merge_sections(state.get("sections") or [])
    return {"merged_sections": merged, "sources": sources}


def write_report(state: ResearchGraphState):

    language = normalize_language(state.get("language"))
    formatted_str_sections = "\n\n".join(state["merged_sections"])

    system_message = report_writer_instructions.format(
        topic=state["topic"],
        context=formatted_str_sections,
        insights_header=label(language, "insights"),
        language_instruction=language_instruction(language),
    )

    write = _writer()
    parts: list[str] = []
    for chunk in get_llm().stream([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos."),
    ]):
        piece = text_of(chunk)
        if piece:
            parts.append(piece)
            write({"draft": piece})

    return {"content": "".join(parts)}


def _intro_or_conclusion(state: ResearchGraphState, request: str) -> str:
    language = normalize_language(state.get("language"))
    sections = state["merged_sections"]
    system = SystemMessage(content=intro_conclusion_instructions.format(
        topic=state["topic"],
        formatted_str_sections="\n\n".join(sections),
        introduction_header=label(language, "introduction"),
        conclusion_header=label(language, "conclusion"),
        language_instruction=language_instruction(language),
    ))
    text = text_of(get_llm().invoke([system, HumanMessage(content=request)]))

    if any(CITE_RE.search(section) for section in sections) and not CITE_RE.search(text):
        logger.info("Section written without citations; asking once more with a reminder")
        retried = text_of(get_llm().invoke([system, HumanMessage(content=request + CITATION_REMINDER)]))
        if CITE_RE.search(retried):
            text = retried
    return _fix_figures(state, text)


def _edit(language: str, text: str, issues: list[dict]) -> str:
    """The editor's pass: the text again, with the problems found in it fixed."""
    return text_of(get_llm().invoke([
        SystemMessage(content=revise_instructions.format(
            issues="\n".join(f"- {i['claim']} -> {i['problem']}" for i in issues),
            report=text,
            language_instruction=language_instruction(language),
        )),
        HumanMessage(content="Return the corrected report."),
    ])).strip()


def _fix_figures(state: ResearchGraphState, text: str) -> str:
    """The introduction and the conclusion are written beside the fact-checked body, so the fact-check never sees
    them. Their figures are checked here against every source gathered, and the editor is asked once when one is
    not there. The text is kept as it was when anything goes wrong."""
    language = normalize_language(state.get("language"))
    try:
        issues = figure_issues(
            text, {s["n"]: s for s in state.get("sources") or []}, state.get("topic", ""), language)
        if not issues:
            return text
        revised = _edit(language, text, issues)
    except Exception:
        logger.warning("Figure check failed; keeping the text as written", exc_info=True)
        return text
    return revised if revised and len(revised) >= 0.5 * len(text) else text


def write_introduction(state: ResearchGraphState):
    return {"introduction": _intro_or_conclusion(state, "Write the report introduction")}


def write_conclusion(state: ResearchGraphState):
    return {"conclusion": _intro_or_conclusion(state, "Write the report conclusion")}


def _sources_for_review(sources: list[dict]) -> str:
    per_source = max(500, REVIEW_SOURCE_CHARS // max(1, len(sources)))
    return "\n\n---\n\n".join(
        f"[{s['n']}] {s['title']} — {s['url']}\n{str(s.get('content') or '')[:per_source]}" for s in sources
    )


def review_report(state: ResearchGraphState):
    language = normalize_language(state.get("language"))
    content = state["content"]
    cited = cited_numbers(content)
    sources = [s for s in state.get("sources") or [] if s["n"] in cited]
    if not sources:
        return {"review_notes": []}

    try:
        # Free and precise, so these come first: a figure that the sentence's own sources do not contain. Every
        # source gathered is searched for it, not only those cited in this text: the right one may not be.
        everything = {s["n"]: s for s in state.get("sources") or []}
        figures = figure_issues(content, everything, state.get("topic", ""), language)
    except Exception:
        logger.warning("Figure check failed; the model's fact-check goes on without it", exc_info=True)
        figures = []

    try:
        checker = get_llm().with_structured_output(ReviewResult, method="function_calling")
        result = checker.invoke([SystemMessage(content=review_instructions.format(
            language=language_name(language), report=content, sources=_sources_for_review(sources),
        ))])
        found = [issue.model_dump() for issue in (result.issues if result else [])]
        issues = (figures + found)[:REVIEW_MAX_ISSUES]
        if not issues:
            return {"review_notes": []}

        revised = _edit(language, content, issues)
    except Exception:
        logger.warning("Fact-check failed; keeping the unreviewed draft", exc_info=True)
        return {"review_notes": [], "warnings": [{"code": "review_failed"}]}

    fixed = bool(revised) and len(revised) >= 0.5 * len(content)
    return {
        "content": revised if fixed else content,
        "review_notes": [{**issue, "fixed": fixed} for issue in issues],
    }


def finalize_report(state: ResearchGraphState):

    language = normalize_language(state.get("language"))
    parts = [
        strip_sources_section(state["introduction"].strip()),
        strip_sources_section(state["content"].strip()),
        strip_sources_section(state["conclusion"].strip()),
    ]
    sources = state.get("sources") or []
    authority = {source["n"]: quality_bonus(source["url"]) for source in sources}
    capped = [tidy_citations(part, MAX_CITATIONS_PER_STATEMENT, lambda n: authority.get(n, 0.0)) for part in parts]
    renumbered, used = finalize_citations(capped, sources)
    introduction, content, conclusion = (tidy_citations(part) for part in renumbered)

    final_report = introduction + "\n\n---\n\n" + content + "\n\n---\n\n" + conclusion
    sources_section = sources_markdown(used, label(language, "sources"))
    if sources_section:
        final_report += "\n\n" + sources_section

    return {"final_report": final_report, "final_sources": used}
