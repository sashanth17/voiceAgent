"""
WebSocket handler for multi-agent system using LangGraph.
"""
from fastapi import WebSocket, WebSocketDisconnect
from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.agent.logger import logger
import json
import uuid
from typing import Optional


async def stream_translation(text: str):
    """
    Placeholder for streaming translation.
    Later replace with real translation model.
    """
    for token in text.split():
        yield token + " "


class MultiAgentWebSocketHandler:
    """
    Handle WebSocket lifecycle using LangGraph-based agent system.
    """

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.connection_id = str(uuid.uuid4())[:8]
        self.graph = build_graph()

        # Conversation-level memory placeholder
        self.summary_memory: list[str] = []

    async def handle(self):
        """
        Connection lifecycle:
        1. Accept connection
        2. Receive messages
        3. Execute LangGraph
        4. Stream response
        5. Cleanup on disconnect
        """
        await self.websocket.accept()
        logger.info(f"[WS {self.connection_id}] Connection accepted")

        try:
            while True:
                data = await self.websocket.receive_text()

                try:
                    message = json.loads(data)
                    await self._handle_message(message)

                except json.JSONDecodeError:
                    logger.error(f"[WS {self.connection_id}] Invalid JSON")
                    await self._send_error("Invalid JSON format")

        except WebSocketDisconnect:
            logger.info(f"[WS {self.connection_id}] Client disconnected")

        except Exception as e:
            logger.exception(f"[WS {self.connection_id}] Error: {e}")

        finally:
            logger.info(f"[WS {self.connection_id}] Cleanup complete")

    async def _handle_message(self, message: dict):
        """
        Process incoming WebSocket message.
        """
        msg_type = message.get("type")

        if msg_type != "user_message":
            await self._send_error(f"Unknown message type: {msg_type}")
            return

        user_query = message.get("content", "").strip()
        if not user_query:
            await self._send_error("Empty message received")
            return

        logger.info(f"[WS {self.connection_id}] User: {user_query[:80]}")

        # -----------------------------
        # Build LangGraph State
        # -----------------------------
        state: AgentState = {
            "query": user_query,
            "context": self._build_context(),
            "payload": message.get("payload"),
            "route_agent": None,
            "agent_response": None,
            "final_response": None,
            "summary_memory": self.summary_memory,
        }

        # -----------------------------
        # Execute graph
        # -----------------------------
        result_state: AgentState = await self.graph.ainvoke(state)

        agent_output: Optional[str] = result_state.get("agent_response")
        if not agent_output:
            await self._send_error("Agent returned empty response")
            return

        # Update memory (very naive for now)
        self.summary_memory.append(user_query)

        # -----------------------------
        # Stream final response (translation placeholder)
        # -----------------------------
        await self._stream_response(agent_output)

    def _build_context(self) -> Optional[str]:
        """
        Build a simple conversation summary.
        Replace with summarizer later.
        """
        if not self.summary_memory:
            return None
        return " | ".join(self.summary_memory[-5:])

    async def _stream_response(self, text: str):
        """
        Stream response token-by-token.
        """
        await self.websocket.send_json({
            "event": "response_start"
        })

        async for chunk in stream_translation(text):
            await self.websocket.send_json({
                "event": "stream",
                "content": chunk
            })

        await self.websocket.send_json({
            "event": "done"
        })

    async def _send_error(self, error: str):
        """
        Send error message to client.
        """
        await self.websocket.send_json({
            "event": "error",
            "content": error
        })