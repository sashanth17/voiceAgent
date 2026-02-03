# app/agent/workers/booking_agent.py
from app.agent.state import AgentState
from app.agent.llms.grok import get_groq_llm


async def booking_agent(state: AgentState) -> AgentState:
    instruction = state["query"]
    model=get_groq_llm()
    response = model.invoke(instruction).content

    state["agent_response"] = response
    return state