# app/agent/graph.py
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.manager import ManagerAgent
from app.agent.llms.gemini import GeminiLLM
from app.agent.workers.booking_agent import booking_agent
from app.agent.workers.symptom_analyzer import symptom_agent


def build_graph() -> StateGraph:
    llm = GeminiLLM()
    manager = ManagerAgent(llm)

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("manager", manager.decide_route)
    graph.add_node("booking_agent", booking_agent)
    graph.add_node("symptomps_agent", symptom_agent)

    # Entry
    graph.set_entry_point("manager")

    # Conditional routing
    def route_selector(state: AgentState):
        return state["route_agent"]

    graph.add_conditional_edges(
        "manager",
        route_selector,
        {
            "booking_agent": "booking_agent",
            "symptomps_agent": "symptomps_agent",
            "direct": END,
        },
    )

    graph.add_edge("booking_agent", END)
    graph.add_edge("symptomps_agent", END)

    return graph.compile()