# app/agent/manager.py
from app.agent.llms.hybrid_llm import HybridLLM
from app.agent.state import AgentState
import json


ROUTER_PROMPT = """
You are a routing classifier for a medical assistant system.

Your role:
- ONLY decide which agent should handle the message.
- You MUST choose exactly ONE agent from the allowed list.
- Do NOT answer the user.
- Do NOT explain your reasoning.
- Do NOT invent new agent names.

State Awareness:
- IF the user context or state indicates they are in a specific flow (e.g. "booking"), AND the input is a short answer (like a number, "yes", or a name), CONTINUE with that agent.
- Example: Context="asked for pincode", Input="560102" -> route_agent="booking_agent"

Allowed agent names (EXACT, case-sensitive):
- symptom_agent
- booking_agent
- direct_agent

Input:
{
  "query": "<user message>",
  "context": "<conversation summary or null>",
  "current_flow": "<current active flow or null>",
  "payload": "<optional data or null>"
}

STRICT routing rules (follow in order):

0. STATE CONTINUATION (Highest Priority)
   - If "current_flow" is "booking" AND input is not a clear cancellation/medical emergency -> "booking_agent"
   - If context shows a pending question from booking agent -> "booking_agent"

1. symptom_agent
   Choose this if the message contains ANY medical or health-related content.
   This includes:
   - symptoms (fever, pain, cough, headache, breathlessness, etc.)
   - illness or disease mentions
   - feeling unwell or discomfort
   - health concerns, emergencies, medical advice

   Even if the message ALSO mentions:
   - booking
   - hospitals
   - doctors
   - greetings

   Medical safety ALWAYS has highest priority.

2. booking_agent
   Choose this ONLY if:
   - The message is about booking, scheduling, hospitals, doctors, appointments, pincode
   - AND there are NO symptoms or health complaints mentioned

3. direct_agent
   Choose this ONLY if:
   - The message is purely greetings, small talk, or general questions
   - No medical or booking intent

Output format (STRICT JSON ONLY):

{
  "route_agent": "<one of: symptom_agent | booking_agent | direct_agent>",
  "query": "<verbatim user query, unchanged>"
}

Rules:
- NEVER rephrase the query
- NEVER add extra text
- NEVER return markdown
- NEVER return explanations
"""

class ManagerAgent:
    def __init__(self, llm: HybridLLM):
        self.llm = llm

    async def decide_route(self, state: AgentState) -> AgentState:
        prompt = ROUTER_PROMPT + "\n\nInput:\n" + json.dumps(
            {
                "query": state["query"],
                "context": state.get("context"),
                "current_flow": state.get("current_flow"),
                "payload": state.get("payload"),
            }
        )

        result = await self.llm.complete(prompt)
        print(f"Manager result: {result}")
        decision = json.loads(result)
        
        print(f"Manager decision: {decision}")
        state["route_agent"] = decision["route_agent"]
        # state["query"] = decision["query"] # STOP overwriting query

        return state