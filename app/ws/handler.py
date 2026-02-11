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
from typing import TypedDict, Optional, Dict, Any, List

from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Optional
from contextlib import AsyncExitStack
import asyncio
class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    async def connect_to_server(self, server_script_path: str):
         is_python = server_script_path.endswith('.py')
         is_js = server_script_path.endswith('.js')
         if not (is_python or is_js):
             raise ValueError("Server script must be a .py or .js file")
         command = "python" if is_python else "node"
         server_params = StdioServerParameters(
             command=command,
             args=[server_script_path],
             env=None)
         stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
         self.stdio, self.write = stdio_transport
         self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
         await self.session.initialize()
         # List available tools
         response = await self.session.list_tools()
         tools = response.tools
         print("\nConnected to server with tools:", [tool.name for tool in tools])
    async def call_tool(self, tool_name: str, arguments: dict):
        """
        🔑 This is the ONLY method LangGraph nodes use
        """
        result = await self.session.call_tool(tool_name, arguments)
        print(result.content[0])
        return result.content  # important

    async def close(self):
        await self.exit_stack.aclose()

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
        logger.info(f"[WS {self.connection_id}] CONNECTION ACCEPTED")

        try:
            while True:
                logger.info(f"[WS {self.connection_id}] Waiting for message...")
                data = await self.websocket.receive_text()
                
                logger.info("-" * 50)
                logger.info(f"RECEIVED FROM PIEHOST: {data}")
                logger.info("-" * 50)
                
                try:
                    logger.info(f"[WS {self.connection_id}] Parsing JSON...")
                    message = json.loads(data)
                    logger.info(f"[WS {self.connection_id}] Processing: {message}")
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
        # Persistence: Initialize or Update State
        # -----------------------------
        if not hasattr(self, "agent_state"):
            self.agent_state: AgentState = {
                "query": user_query,
                "user_message": user_query,
                "messages": [],
                "context": None,
                "payload": message.get("payload"),
                "route_agent": None,
                "symptoms": [],
                "diagnosis_probabilities": [],
                "urgency_score": 0,
                "missing_information": None,
                "next_step": None,
                "asked_questions": [],
                "agent_response": None,
                "final_response": None,
                "summary_memory": self.summary_memory,
                "pincode" : None,
                "hospital_options": None,
                "hospital_id": None,
                "appointment_date": None,
                "booking_confirmed": None,
                "final_response": None,
                "patient_contact": "7558187099",
                "current_flow": None  # Added for flow tracking
            }
            # Initialize MCP client as instance attribute, NOT in state
            self.mcp_client = MCPClient()
            await self.mcp_client.connect_to_server("app/tools/mcp-servers/server.py")
            
        else:
            # Update dynamic fields for the new turn
            self.agent_state["query"] = user_query
            self.agent_state["user_message"] = user_query
            # Don't overwrite context if it's already built, or rebuild it?
            # Rebuilding context from memory is safer.
            self.agent_state["context"] = self._build_context()
            self.agent_state["context"] = self._build_context()
            self.agent_state["payload"] = message.get("payload")

        # -----------------------------
        # Message History Update
        # -----------------------------
        # Append the user's message to the conversation history
        # This is CRITICAL for agents like 'extract_intent' that look at history
        self.agent_state.setdefault("messages", []).append({
            "role": "user", 
            "content": user_query
        })

        # -----------------------------
        # Execute graph
        # -----------------------------
        # Inject mcp_client into the input state at runtime
        # This creates a shallow copy so we don't modify self.agent_state with non-serializable objects
        input_state = self.agent_state.copy()
        input_state["mcp_client"] = self.mcp_client

        # Capture result state explicitly
        result_state: AgentState = await self.graph.ainvoke(input_state)
        
        # PERSIST: Update our persistent state with the graph output
        # Filter out mcp_client if it happens to be returned (it shouldn't be in AgentState definition soon)
        if "mcp_client" in result_state:
            del result_state["mcp_client"]
            
        self.agent_state.update(result_state)

        agent_output: Optional[str] = self.agent_state.get("agent_response")
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