# app/agent/manager.py
from app.agent.llms.gemini import GeminiLLM
from app.agent.state import AgentState
import json


ROUTER_PROMPT = """
You are a Manager Agent.

Your task:
- Decide which agent should handle the user query.

Available agents:
1. booking_agent
2. symptomps_agent
3. direct

Input format:
{
  "query": "<user query>",
  "context": "<conversation summary or null>",
  "payload": "<optional payload or null>"
}

Rules:
- If the query is about appointments, scheduling, hospital visits → booking_agent
- If the query describes health issues, symptoms, pain → symptomps_agent
- Greetings, general questions → direct

Return STRICT JSON ONLY:
{
  "route_agent": "<agent_name>",
  "querry": "<instruction for the agent including context summary>"
}
"""


class ManagerAgent:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    async def decide_route(self, state: AgentState) -> AgentState:
        prompt = ROUTER_PROMPT + "\n\nInput:\n" + json.dumps(
            {
                "query": state["query"],
                "context": state.get("context"),
                "payload": state.get("payload"),
            }
        )

        result = await self.llm.complete(prompt)
        decision = json.loads(result)

        state["route_agent"] = decision["route_agent"]
        state["query"] = decision["querry"]

        return state