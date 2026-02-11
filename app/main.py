"""
FastAPI application entry point.

Runs the async agent server with WebSocket support.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket

from app.ws.handler import MultiAgentWebSocketHandler
from app.agent.llms.biobert import get_biobert_classifier
from app.agent.llms.symptom_extractor import get_symptom_extractor
from app.agent.logger import logger

import os

os.environ.pop("DYLD_LIBRARY_PATH", None)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    """
    logger.info("=" * 60)
    logger.info("Async AI Agent Server Starting")
    
    # Pre-load heavy models
    try:
        logger.info("Pre-loading Disease Inference Model...")
        get_biobert_classifier()
        logger.info("Pre-loading Symptom Extractor Model...")
        get_symptom_extractor()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        
    logger.info("WebSocket endpoint: ws://localhost:8000/ws")
    logger.info("=" * 60)

    yield  # ---- application runs here ----

    logger.info("Async AI Agent Server Shutting Down")


app = FastAPI(
    title="Async AI Agent Server",
    description="Production-quality stateful AI agent server with LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "async-agent-server"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for agent communication.

    Each connection creates a new Agent instance with:
    - Isolated memory
    - Fresh LangGraph instance
    - Independent execution lifecycle
    """
    handler = MultiAgentWebSocketHandler(websocket)
    await handler.handle()