# app/agent/graph.py
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.manager import ManagerAgent
from app.agent.llms.hybrid_llm import HybridLLM
from app.agent.workers.booking_agent import booking_agent
from app.agent.workers.direct_agent import direct_agent
from app.agent.workers.medical_worker import (
    symptom_extractor,
    disease_inference,
    urgency_scorer,
    symptom_completion,
    response_generator
)

def build_graph() -> StateGraph:
    llm = HybridLLM()
    manager = ManagerAgent(llm)

    graph = StateGraph(AgentState)

    # --- Nodes ---
    # 1. Routing & Entry
    graph.add_node("manager", manager.decide_route)
    
    # 2. Medical Pipeline
    graph.add_node("symptom_extractor", symptom_extractor)
    graph.add_node("disease_inference", disease_inference)
    graph.add_node("urgency_scorer", urgency_scorer)
    graph.add_node("symptom_completion", symptom_completion)
    graph.add_node("response_generator", response_generator)
    
    # 3. Other Agents
    graph.add_node("booking_agent", booking_agent)
    graph.add_node("direct_agent", direct_agent)

    # --- Edges ---
    graph.set_entry_point("manager")

    # Manager -> Route
    def route_selector(state: AgentState):
        route = state.get("route_agent", "direct")
        # Ensure we map compatible names
        if route == "symptom_agent" or route == "symptomps_agent":
            return "symptom_extractor"
        return route

    graph.add_conditional_edges(
        "manager",
        route_selector,
        {
            "symptom_extractor": "symptom_extractor",
            "booking_agent": "booking_agent",
            "direct": "direct_agent",
            "direct_agent": "direct_agent" # safe fallback
        },
    )

    # Medical Pipeline Flow
    graph.add_edge("symptom_extractor", "disease_inference")
    graph.add_edge("disease_inference", "urgency_scorer")
    graph.add_edge("urgency_scorer", "symptom_completion")
    graph.add_edge("symptom_completion", "response_generator")
    graph.add_edge("response_generator", END)

    # Other Endings
    graph.add_edge("booking_agent", END)
    graph.add_edge("direct_agent", END)

    return graph.compile()