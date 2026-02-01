"""
Agent: Per-connection agent instance.

Each WebSocket connection creates ONE Agent.
Agent owns:
- Memory (conversation history)
- LangGraph instance
- Async execution task
"""
import asyncio
from typing import Optional, AsyncGenerator
from app.agent.graph import AgentGraph
from app.agent.state import AgentState
from app.agent.logger import logger


class Agent:
    """
    Stateful agent instance tied to a single WebSocket connection.
    
    Lifecycle:
    1. __init__: Create memory, build graph
    2. process_message: Execute graph with user input
    3. cleanup: Cancel running tasks, destroy state
    """
    
    def __init__(self, connection_id: str):
        """
        Initialize agent for a new connection.
        
        Args:
            connection_id: Unique identifier for this WebSocket connection
        """
        self.connection_id = connection_id
        
        # Isolated memory (destroyed on disconnect)
        self.memory: list[dict] = []
        
        # Build a FRESH graph instance (NEVER shared)
        graph_builder = AgentGraph()
        self.graph = graph_builder.build()
        
        # Track running async task for cancellation
        self.current_task: Optional[asyncio.Task] = None
        
        logger.info(f"[Agent {connection_id}] Initialized with fresh graph and memory")
    
    async def process_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Process user message through the agent graph.
        
        Args:
            user_message: User's input text
        
        Yields:
            Streaming response chunks (word-by-word or sentence-by-sentence)
        """
        logger.info(f"[Agent {self.connection_id}] Processing message: {user_message[:50]}...")
        
        # Update conversation memory
        self.memory.append({"role": "user", "content": user_message})
        
        # Build initial state
        initial_state: AgentState = {
            "user_message": user_message,
            "messages": self.memory.copy(),
            "reasoning": "",
            "final_response": "",
            "needs_deep_analysis": False
        }
        
        # Execute graph asynchronously
        try:
            # Stream graph execution
            async for chunk in self._stream_graph_execution(initial_state):
                yield chunk
        
        except asyncio.CancelledError:
            logger.warning(f"[Agent {self.connection_id}] Task cancelled during execution")
            raise
        
        except Exception as e:
            logger.error(f"[Agent {self.connection_id}] Error during graph execution: {e}")
            yield f"[Error: {str(e)}]"
    
    async def _stream_graph_execution(self, initial_state: AgentState) -> AsyncGenerator[str, None]:
        """
        Execute graph and stream output incrementally.
        
        Args:
            initial_state: Initial AgentState to pass to graph
        
        Yields:
            Response chunks as they become available
        """
        # Invoke graph (async execution)
        final_state = await self.graph.ainvoke(initial_state)
        
        # Extract final response
        response = final_state.get("final_response", "")
        
        # Update memory with assistant response
        self.memory.append({"role": "assistant", "content": response})
        
        # Stream output word-by-word (simulate streaming)
        words = response.split()
        for i, word in enumerate(words):
            # Add space except for first word
            chunk = word if i == 0 else f" {word}"
            yield chunk
            
            # Small delay to simulate streaming (remove in production)
            await asyncio.sleep(0.05)
    
    async def cleanup(self):
        """
        Clean up agent resources on disconnect.
        
        - Cancels running tasks
        - Clears memory
        - Destroys graph instance
        """
        logger.info(f"[Agent {self.connection_id}] Cleaning up resources")
        
        # Cancel running task if exists
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
        
        # Clear memory
        self.memory.clear()
        
        # Graph instance will be garbage collected
        self.graph = None
        
        logger.info(f"[Agent {self.connection_id}] Cleanup complete")