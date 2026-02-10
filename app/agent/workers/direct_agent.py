from app.agent.state import AgentState

from app.agent.llms.hybrid_llm import HybridLLM

async def direct_agent(state: AgentState) -> AgentState:
    """Handles general queries by passing them directly."""
    llm = HybridLLM()
    response = await llm.complete(state["query"])
    state["agent_response"] = response
    return state
