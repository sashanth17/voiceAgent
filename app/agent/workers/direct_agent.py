from app.agent.state import AgentState

from app.agent.llms.gemini import GeminiLLM

async def direct_agent(state: AgentState) -> AgentState:
    """Handles general queries by passing them directly."""
    llm = GeminiLLM()
    response = await llm.complete(state["query"])
    state["agent_response"] = response
    return state
