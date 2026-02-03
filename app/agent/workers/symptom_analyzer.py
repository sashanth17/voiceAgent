# app/agent/workers/symptom_analyzer.py
from app.agent.state import AgentState
from app.agent.llms.grok import get_groq_llm


async def symptom_agent(state: AgentState) -> AgentState:
    instruction = state["query"]
    model=get_groq_llm()

    # Placeholder logic (replace with medical reasoning later)
    response = model.invoke(instruction).content

    state["agent_response"] = response
    return state