"""
WebSocket handler for multi-agent system.
"""
from fastapi import WebSocket, WebSocketDisconnect
from app.agent.manager import ManagerAgent
from app.agent.logger import logger
import json
import uuid


class MultiAgentWebSocketHandler:
    """
    Handle WebSocket connection lifecycle with manager agent.
    """
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.connection_id = str(uuid.uuid4())[:8]
        self.manager: ManagerAgent | None = None
    
    async def handle(self):
        """
        Main connection handler.
        
        Lifecycle:
        1. Accept connection
        2. Create manager agent
        3. Process messages
        4. Cleanup on disconnect
        """
        await self.websocket.accept()
        logger.info(f"[WS {self.connection_id}] Connection accepted")
        
        # Create manager instance
        self.manager = ManagerAgent(self.connection_id)
        
        try:
            while True:
                # Receive message
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
            logger.error(f"[WS {self.connection_id}] Error: {e}")
        
        finally:
            if self.manager:
                await self.manager.cleanup()
            logger.info(f"[WS {self.connection_id}] Cleanup complete")
    
    async def _handle_message(self, message: dict):
        """Process incoming message."""
        msg_type = message.get("type")
        
        if msg_type == "user_message":
            content = message.get("content", "")
            logger.info(f"[WS {self.connection_id}] User: {content[:50]}...")
            
            # Process through manager
            response = await self.manager.process_message(content)
            
            # Send complete response (streaming can be added later)
            await self._send_response(response)
        
        else:
            logger.warning(f"[WS {self.connection_id}] Unknown type: {msg_type}")
            await self._send_error(f"Unknown message type: {msg_type}")
    
    async def _send_response(self, content: str):
        """Send response to client."""
        await self.websocket.send_json({
            "event": "response",
            "content": content
        })
    
    async def _send_error(self, error: str):
        """Send error to client."""
        await self.websocket.send_json({
            "event": "error",
            "content": error
        })