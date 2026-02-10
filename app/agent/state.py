# app/agent/state.py
from typing import TypedDict, Optional, Dict, Any, List

class AgentState(TypedDict):
    # raw input
    query: str
    user_message: str
    
    # accumulated context
    messages: List[Dict[str, str]]
    context: Optional[str]

    # structured payload (optional)
    payload: Optional[Dict[str, Any]]

    # routing
    route_agent: Optional[str]

    # Medical Triage State
    symptoms: Optional[List[str]]
    diagnosis_probabilities: Optional[List[Dict[str, Any]]]
    urgency_score: Optional[int]
    missing_information: Optional[str] # Information we need to ask for
    next_step: Optional[str] # "ask", "booking", "escalate", "advice"
    asked_questions: Optional[List[str]] # Loop prevention
    duration_context: Optional[str] # Persisted duration information (e.g., "5 days")

    # agent output
    agent_response: Optional[str]
    reasoning: Optional[str]

    # final user response (post translation)
    final_response: Optional[str]
    needs_deep_analysis: Optional[bool]

    # Patient Demographics & History
    patient_info: Optional[Dict[str, Any]] # {"age": int, "gender": str, "is_pregnant": bool, "history": List[str]}
    
    # Booking State
    booking_state: Optional[Dict[str, Any]]
