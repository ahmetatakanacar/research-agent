from dotenv import load_dotenv
from states import GenerateAnalystState
from models import llm
from objects import Analyst, Perpectives
from prompts import analyst_instructions
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

def create_analyst(state: GenerateAnalystState):

    topic = state["topic"]
    max_analyst = state["max_analyst"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    structured_llm = llm.with_structured_output(Perpectives)
    system_message = analyst_instructions.format(topic=topic, human_analyst_feedback=human_analyst_feedback, max_analyst=max_analyst)
    analyst = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Please generate the set of analyst.")])
    
    return {"analyst": analyst.analyst}