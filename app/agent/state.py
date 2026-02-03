# app/agent/state.py
from typing import TypedDict, Optional, Dict, Any, List


class AgentState(TypedDict):
    # raw input
    query: str

    # accumulated context (conversation summary etc.)
    context: Optional[str]

    # structured payload (optional)
    payload: Optional[Dict[str, Any]]

    # routing
    route_agent: Optional[str]

    # agent output
    agent_response: Optional[str]

    # final user response (post translation)
    final_response: Optional[str]

    # memory placeholder
    summary_memory: Optional[List[str]]