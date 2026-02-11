from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.manager import ManagerAgent
from app.agent.llms.hybrid_llm import HybridLLM

from app.agent.workers.direct_agent import direct_agent

# Medical workers
from app.agent.workers.medical_worker import (
    symptom_extractor,
    disease_inference,
    urgency_scorer,
    symptom_completion,
    response_generator,
)

# Booking workers (state machine)
from app.agent.workers.booking_agent import (
    extract_intent,
    ask_pincode,
    fetch_hospitals,
    ask_hospital_selection,
    ask_appointment_date,
    ask_confirmation,
    perform_booking,
    decide_next_step,
)


def build_graph():
    llm = HybridLLM()
    manager = ManagerAgent(llm)

    graph = StateGraph(AgentState)

    # --------------------
    # Nodes
    # --------------------

    # Entry / Router
    graph.add_node("manager", manager.decide_route)

    # Medical pipeline
    graph.add_node("symptom_extractor", symptom_extractor)
    graph.add_node("disease_inference", disease_inference)
    graph.add_node("urgency_scorer", urgency_scorer)
    graph.add_node("symptom_completion", symptom_completion)
    graph.add_node("response_generator", response_generator)

    # Direct agent
    graph.add_node("direct_agent", direct_agent)

    # Booking flow nodes
    graph.add_node("extract_intent", extract_intent)
    graph.add_node("ask_pincode", ask_pincode)
    graph.add_node("fetch_hospitals", fetch_hospitals)
    graph.add_node("ask_hospital_selection", ask_hospital_selection)
    graph.add_node("ask_appointment_date", ask_appointment_date)
    graph.add_node("ask_confirmation", ask_confirmation)
    graph.add_node("perform_booking", perform_booking)

    # --------------------
    # Entry Point
    # --------------------
    graph.set_entry_point("manager")

    # --------------------
    # Manager Routing
    # --------------------

    def route_selector(state: AgentState):
        decision = state.get("route_agent")
        current_flow = state.get("current_flow")

        # 1. Priority: If explicit routing decision exists, follow it
        if decision:
            if decision in {"symptom_agent", "symptoms_agent"}:
                return "symptom_extractor"
            if decision == "booking_agent":
                return "extract_intent"
            if decision == "direct_agent":
                return "direct_agent"
        
        # 2. Fallback: If no decision but we are in a flow, continue
        if current_flow == "booking":
             return "extract_intent"

        return "direct_agent"

    graph.add_conditional_edges(
        "manager",
        route_selector,
        {
            "symptom_extractor": "symptom_extractor",
            "extract_intent": "extract_intent",
            "direct_agent": "direct_agent",
        },
    )

    # --------------------
    # Medical Pipeline
    # --------------------
    graph.add_edge("symptom_extractor", "disease_inference")
    graph.add_edge("disease_inference", "urgency_scorer")
    graph.add_edge("urgency_scorer", "symptom_completion")
    graph.add_edge("symptom_completion", "response_generator")
    
    # Updated Transition Logic
    def medical_handoff(state: AgentState):
        if state.get("next_step") == "booking":
            return "extract_intent"
        return END

    graph.add_conditional_edges(
        "response_generator",
        medical_handoff,
        {
            "extract_intent": "extract_intent",  # Seamless transition
            END: END
        }
    )

    # --------------------
    # Booking State Machine
    # --------------------
    graph.add_conditional_edges(
        "extract_intent",
        decide_next_step,
        {
            "ask_pincode": "ask_pincode",
            "fetch_hospitals": "fetch_hospitals",
            "ask_hospital_selection": "ask_hospital_selection",
            "ask_appointment_date": "ask_appointment_date",
            "ask_confirmation": "ask_confirmation",
            "perform_booking": "perform_booking",
        },
    )

    graph.add_edge("ask_pincode", END)
    graph.add_edge("fetch_hospitals", "ask_hospital_selection")
    graph.add_edge("ask_hospital_selection", END)
    graph.add_edge("ask_appointment_date", END)
    graph.add_edge("ask_confirmation", END)
    graph.add_edge("perform_booking", END)

    # --------------------
    # Direct Agent End
    # --------------------
    graph.add_edge("direct_agent", END)

    return graph.compile()