from dotenv import load_dotenv
from src.utils.states import GenerateAnalystState, InterviewState, ResearchGraphState
from src.utils.models import llm
from src.utils.objects import Analyst, Perpectives, SearchQuery
from src.utils.prompts import analyst_instructions, question_instructions, search_instructions, answer_instructions, section_writer_instructions
from src.utils.prompts import report_writer_instructions, intro_conclusion_instructions
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
import wikipedia
from langchain_core.messages import get_buffer_string

load_dotenv()

# Wikipedia rejects the wikipedia package's default User-Agent with a 403
wikipedia.set_user_agent("research-agent/0.1")

def create_analyst(state: GenerateAnalystState):

    topic = state["topic"]
    max_analyst = state["max_analyst"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    structured_llm = llm.with_structured_output(Perpectives, method="function_calling")
    system_message = analyst_instructions.format(topic=topic, human_analyst_feedback=human_analyst_feedback, max_analyst=max_analyst)
    analyst = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Please generate the set of analyst.")])
    
    return {"analyst": analyst.analyst}

def human_feedback(state: GenerateAnalystState):
    feedback = interrupt({
        "question": "Are these analysts okay?",
        "analyst": [
            analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
                for analyst in state.get("analyst", [])
        ],
        "instructions": "Return feedback to regenerate analysts, or return empty/perfect/continue to approve."
    })

    if feedback is None:
        return {"human_analyst_feedback": None}
    if isinstance(feedback, str):
        feedback = feedback.strip()

        if feedback == "":
            return {"human_analyst_feedback": None}
        if feedback.lower() in {"perfect", "continue", "approved", "yes"}:
            return {"human_analyst_feedback": None}
        return {"human_analyst_feedback": feedback}

    return {"human_analyst_feedback": None}

def generate_question(state: InterviewState):

    analyst = state["analyst"]
    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    messages = state["messages"]
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}

def search_web(state: InterviewState):

    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_instruction_system_message = SystemMessage(content=search_instructions)
    tavily_search = TavilySearch(max_results=3)

    search_query = structured_llm.invoke([search_instruction_system_message] + state["messages"])

    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data) 

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState):

    structured_llm = llm.with_structured_output(SearchQuery, method="function_calling")
    search_instruction_system_message = SystemMessage(content=search_instructions)

    search_query = structured_llm.invoke([search_instruction_system_message] + state["messages"])

    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def generate_answer(state: InterviewState):

    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
    answer.name = "expert"

    return {"messages": [answer]}

def save_interview(state: InterviewState):

    messages = state["messages"]
    interview = get_buffer_string(messages)

    return {"interview": interview}

def write_section(state: InterviewState):

    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)] + 
                         [HumanMessage(content=f"Use this source to write your section: {context}")])

    return {"sections": [section.content]}

def write_report(state: ResearchGraphState):

    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):

    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):

    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):

    content = state["content"]
    if content.startswith("## Insights"):
        content = content.removeprefix("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}