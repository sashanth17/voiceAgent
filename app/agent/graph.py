"""
AgentGraph: LangGraph structure definition.

This class defines the BLUEPRINT of the graph (structure, nodes, edges).
Each Agent instance calls build() to get a FRESH compiled graph.
"""
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.logger import logger
import asyncio


class AgentGraph:
    """
    Defines the agent's LangGraph structure.
    
    Graph flow:
    1. analyze_intent → decides if deep analysis is needed
    2. If yes: deep_analysis → final_response
       If no: quick_response (direct to final)
    
    This is a 2-3 LLM chain depending on routing.
    """
    
    def __init__(self):
        """
        Initialize graph builder.
        No state is stored here—only structure.
        """
        self.graph = StateGraph(AgentState)
        self._setup_nodes()
        self._setup_edges()
    
    def _setup_nodes(self):
        """Define graph nodes (LLM calls)."""
        self.graph.add_node("analyze_intent", self._analyze_intent)
        self.graph.add_node("deep_analysis", self._deep_analysis)
        self.graph.add_node("quick_response", self._quick_response)
        self.graph.add_node("final_response", self._final_response)
    
    def _setup_edges(self):
        """Define graph edges (routing logic)."""
        # Entry point
        self.graph.set_entry_point("analyze_intent")
        
        # Conditional routing after intent analysis
        self.graph.add_conditional_edges(
            "analyze_intent",
            self._route_after_intent,
            {
                "deep": "deep_analysis",
                "quick": "quick_response"
            }
        )
        
        # Both paths converge to final_response
        self.graph.add_edge("deep_analysis", "final_response")
        self.graph.add_edge("quick_response", "final_response")
        
        # End after final response
        self.graph.add_edge("final_response", END)
    
    async def _analyze_intent(self, state: AgentState) -> AgentState:
        """
        Node 1: Analyze user intent (LLM call 1).
        
        Simulates calling an LLM to determine if deep analysis is needed.
        In production, replace with actual LLM call.
        """
        logger.info("[Node: analyze_intent] Processing user message")
        
        user_msg = state["user_message"]
        
        # Simulate async LLM call
        await asyncio.sleep(0.1)  # Replace with: await llm.ainvoke(...)
        
        # Simple heuristic (replace with real LLM reasoning)
        needs_deep = len(user_msg.split()) > 10 or "?" in user_msg
        
        logger.info(f"[Node: analyze_intent] Deep analysis needed: {needs_deep}")
        
        return {
            **state,
            "reasoning": f"Analyzed intent for: '{user_msg[:50]}...'",
            "needs_deep_analysis": needs_deep
        }
    
    async def _deep_analysis(self, state: AgentState) -> AgentState:
        """
        Node 2a: Deep analysis (LLM call 2 - optional path).
        
        Used for complex queries requiring detailed reasoning.
        """
        logger.info("[Node: deep_analysis] Performing deep analysis")
        
        # Simulate async LLM call with context
        await asyncio.sleep(0.15)  # Replace with: await llm.ainvoke(...)
        
        analysis = f"Deep analysis of: {state['reasoning']}"
        
        return {
            **state,
            "reasoning": analysis
        }
    
    async def _quick_response(self, state: AgentState) -> AgentState:
        """
        Node 2b: Quick response (LLM call 2 - alternative path).
        
        Used for simple queries that don't need deep analysis.
        """
        logger.info("[Node: quick_response] Generating quick response")
        
        # Simulate async LLM call
        await asyncio.sleep(0.1)  # Replace with: await llm.ainvoke(...)
        
        response = f"Quick answer for: {state['user_message'][:30]}"
        
        return {
            **state,
            "final_response": response
        }
    
    async def _final_response(self, state: AgentState) -> AgentState:
        """
        Node 3: Generate final response (LLM call 3 - always runs).
        
        Synthesizes reasoning into a user-friendly response.
        """
        logger.info("[Node: final_response] Generating final response")
        
        # If quick_response already set final_response, use it
        if state.get("final_response"):
            return state
        
        # Otherwise, synthesize from deep analysis
        await asyncio.sleep(0.1)  # Replace with: await llm.ainvoke(...)
        
        final = f"Based on {state['reasoning']}, here's my response to: {state['user_message']}"
        
        return {
            **state,
            "final_response": final
        }
    
    def _route_after_intent(self, state: AgentState) -> str:
        """
        Conditional routing function.
        
        Returns:
            "deep" if needs_deep_analysis is True
            "quick" otherwise
        """
        return "deep" if state.get("needs_deep_analysis", False) else "quick"
    
    def build(self):
        """
        Compile and return a FRESH graph instance.
        
        CRITICAL: This must be called per-agent to ensure isolation.
        
        Returns:
            Compiled LangGraph instance
        """
        logger.info("[AgentGraph] Building fresh graph instance")
        return self.graph.compile()