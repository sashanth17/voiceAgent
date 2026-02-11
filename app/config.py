"""
Centralized configuration for multi-agent system.
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Model configuration for each agent."""
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30


@dataclass
class AgentConfig:
    """Multi-agent system configuration."""
    
    # Manager Agent (Ollama Phi3)
    manager_model: ModelConfig = ModelConfig(
        name="phi3:mini",
        base_url="http://localhost:11434",  # Ollama default
        timeout=30
    )
    
    # Worker Agents (Gemini/Grok)
    symptom_analyzer_model: ModelConfig = ModelConfig(
        name="gemini-1.5-flash",  # or "grok-beta"
        api_key=os.getenv("GEMINI_API_KEY"),  # or GROK_API_KEY
        timeout=20
    )
    
    booking_agent_model: ModelConfig = ModelConfig(
        name="gemini-1.5-flash",  # or "grok-beta"
        api_key=os.getenv("GEMINI_API_KEY"),  # or GROK_API_KEY
        timeout=20
    )
    
    # System settings
    max_memory_messages: int = 20
    log_worker_outputs: bool = True


# Global config instance
config = AgentConfig()