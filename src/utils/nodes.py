from dotenv import load_dotenv
from src.utils.states import GenerateAnalystState, InterviewState
from src.utils.models import llm
from src.utils.objects import Analyst, Perpectives, SearchQuery
from src.utils.prompts import analyst_instructions, question_instructions, search_instructions
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch

load_dotenv()

def create_analyst(state: GenerateAnalystState):

    topic = state["topic"]
    max_analyst = state["max_analyst"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    structured_llm = llm.with_structured_output(Perpectives)
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

    structured_llm = llm.with_structured_output(SearchQuery)
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

def search_web2(state: InterviewState):

    structured_llm = llm.with_structured_output(SearchQuery)
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