#!/usr/bin/env python3
"""
Telemedicine MCP Server

This MCP server provides tools for interacting with a telemedicine platform API.
It exposes patient management, hospital search, and booking functionalities.
"""
import asyncio
import logging
from typing import Any
from mcp.server import Server
from mcp.types import TextContent, Tool
from src.tools import ALL_TOOLS, ALL_TOOL_HANDLERS
from src.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize the MCP server
app = Server("telemedicine-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools.
    
    Returns:
        List of Tool objects representing available operations
    """
    logger.info(f"Listing {len(ALL_TOOLS)} available tools")
    return ALL_TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """
    Handle tool execution requests.
    
    Args:
        name: Name of the tool to execute
        arguments: Arguments for the tool
        
    Returns:
        List containing the tool execution result
        
    Raises:
        ValueError: If the tool name is unknown
    """
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    # Check if tool exists
    if name not in ALL_TOOL_HANDLERS:
        error_msg = f"Unknown tool: {name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Execute the tool handler
        handler = ALL_TOOL_HANDLERS[name]
        result = await handler(arguments)
        
        logger.info(f"Tool {name} executed successfully")
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        error_msg = f"Error executing tool {name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=error_msg)]


async def main() -> None:
    """
    Main entry point for the MCP server.
    """
    logger.info("Starting Telemedicine MCP Server")
    logger.info(f"Base URL: {settings.base_url}")
    logger.info(f"Available tools: {len(ALL_TOOLS)}")
    
    # Import and run the server using stdio transport
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server running on stdio transport")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise