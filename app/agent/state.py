"""
LangGraph state definitions.
Defines the structure of data passed between graph nodes.
"""
from typing import TypedDict, List


class AgentState(TypedDict):
    """
    State schema for the agent graph.
    
    This structure is passed through all graph nodes.
    Each node can read from and write to this state.
    """
    # Current user message
    user_message: str
    
    # Conversation history (list of dicts with 'role' and 'content')
    messages: List[dict]
    
    # Intermediate reasoning/analysis from first LLM
    reasoning: str
    
    # Final response to send to user
    final_response: str
    
    # Control flag for conditional routing
    needs_deep_analysis: bool