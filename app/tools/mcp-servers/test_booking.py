#!/usr/bin/env python3
"""
Telemedicine MCP Console Client

An interactive console for spawning the MCP server as a subprocess,
connecting to it via stdio transport, and executing tools.

Usage:
    python console.py

Commands:
    fetch_tools                     - List all available tools
    call <tool_name> <json_args>    - Execute a tool with arguments
    describe <tool_name>            - Show tool schema and description
    clear                           - Clear the console
    help                            - Show help message
    exit | quit                     - Exit the console
"""

import asyncio
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ─────────────────────────────────────────────
#  ANSI colour helpers
# ─────────────────────────────────────────────
SUPPORTS_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if SUPPORTS_COLOR else text

def green(t):   return _c("32", t)
def red(t):     return _c("31", t)
def yellow(t):  return _c("33", t)
def cyan(t):    return _c("36", t)
def bold(t):    return _c("1",  t)
def dim(t):     return _c("2",  t)
def magenta(t): return _c("35", t)


# ─────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────
BANNER = f"""
{cyan('╔══════════════════════════════════════════════════╗')}
{cyan('║')}    {bold('🏥  Telemedicine MCP Console')}                  {cyan('║')}
{cyan('║')}    {dim('Interactive tool executor for MCP server')}      {cyan('║')}
{cyan('╚══════════════════════════════════════════════════╝')}

  {bold('Commands:')}
  {green('fetch_tools')}                     → List all available tools
  {green('call')} {yellow('<tool_name>')} {yellow('<json_args>')}     → Execute a tool
  {green('describe')} {yellow('<tool_name>')}          → Show tool schema
  {green('clear')}                           → Clear screen
  {green('help')}                            → Show this message
  {green('exit')}  {dim('|')}  {green('quit')}                → Exit console

  {dim('Example:')}
  {dim('>> call register_user {"phone_number":"9999999999","name":"John","age":30,"gender":"male","pincode":"560001"}')}
"""

HELP_TEXT = BANNER


# ─────────────────────────────────────────────
#  Pretty printers
# ─────────────────────────────────────────────
def print_separator(char: str = "─", width: int = 54) -> None:
    print(dim(char * width))


def print_success(msg: str) -> None:
    print(f"  {green('✔')}  {msg}")


def print_error(msg: str) -> None:
    print(f"  {red('✘')}  {red(msg)}")


def print_info(msg: str) -> None:
    print(f"  {cyan('ℹ')}  {msg}")


def print_warning(msg: str) -> None:
    print(f"  {yellow('⚠')}  {yellow(msg)}")


def print_json(data: str) -> None:
    """Pretty-print a JSON string with colour."""
    try:
        parsed = json.loads(data)
        formatted = json.dumps(parsed, indent=2)

        for line in formatted.splitlines():
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)
            pad      = " " * indent

            if stripped.startswith('"') and ":" in stripped:
                # key : value pair
                key, _, rest = stripped.partition(":")
                print(f"{pad}{cyan(key)}:{rest}")
            elif stripped.startswith('"'):
                print(f"{pad}{yellow(stripped)}")
            elif stripped in ("{", "}", "[", "]", "{,", "},", "],"):
                print(f"{pad}{dim(stripped)}")
            else:
                # number / bool / null / closing
                print(f"{pad}{magenta(stripped)}")
    except json.JSONDecodeError:
        # Not JSON – just print raw
        print(f"  {data}")


def print_tool_card(tool) -> None:
    """Render a single tool summary card."""
    print_separator()
    print(f"  {bold(green(tool.name))}")
    print(f"  {dim(tool.description)}")

    schema = tool.inputSchema or {}
    props  = schema.get("properties", {})
    req    = schema.get("required",   [])

    if props:
        print(f"\n  {bold('Parameters:')}")
        for param_name, param_info in props.items():
            required_tag = f" {red('*required')}" if param_name in req else f" {dim('optional')}"
            ptype        = param_info.get("type", "any")
            desc         = param_info.get("description", "")
            enum_vals    = param_info.get("enum", [])

            print(f"    {yellow(param_name)}{required_tag}")
            print(f"      type : {cyan(ptype)}")
            if desc:
                print(f"      info : {dim(desc)}")
            if enum_vals:
                print(f"      enum : {dim(str(enum_vals))}")


# ─────────────────────────────────────────────
#  Command handlers
# ─────────────────────────────────────────────
async def cmd_fetch_tools(session: ClientSession) -> None:
    """List every tool registered on the server."""
    print()
    print_info("Fetching tools from server…")

    tools_response = await session.list_tools()
    tools          = tools_response.tools

    if not tools:
        print_warning("No tools found on the server.")
        return

    print()
    print(f"  {bold(f'Found {len(tools)} tool(s):')}")
    print()

    for idx, tool in enumerate(tools, 1):
        schema   = tool.inputSchema or {}
        req      = schema.get("required", [])
        req_str  = ", ".join(req) if req else dim("none")
        print(f"  {dim(str(idx) + '.')}  {bold(green(tool.name))}")
        print(f"       {dim(tool.description[:70] + ('…' if len(tool.description) > 70 else ''))}")
        print(f"       required args: {cyan(req_str)}")
        print()

    print_separator()
    print(f"  {dim('Tip: use')} {green('describe <tool_name>')} {dim('to see the full schema')}")
    print()


async def cmd_describe(session: ClientSession, tool_name: str) -> None:
    """Print full schema of a single tool."""
    tools_response = await session.list_tools()
    match          = next((t for t in tools_response.tools if t.name == tool_name), None)

    if not match:
        print_error(f"Tool '{tool_name}' not found.")
        print_info("Run 'fetch_tools' to see available tools.")
        return

    print()
    print_tool_card(match)
    print()
    print(f"  {bold('Full schema (JSON):')}")
    print_json(json.dumps(match.inputSchema or {}))
    print_separator()

    # Show a ready-to-use example call
    props   = (match.inputSchema or {}).get("properties", {})
    example = {}
    for k, v in props.items():
        t = v.get("type", "string")
        if t == "string":
            example[k] = f"<{k}>"
        elif t == "integer":
            example[k] = 0
        elif t == "boolean":
            example[k] = False
        else:
            example[k] = None

    if example:
        print()
        print(f"  {dim('Example call:')}")
        print(f"  {green('call')} {yellow(tool_name)} {json.dumps(example)}")
    print()


async def cmd_call_tool(
    session: ClientSession,
    tool_name: str,
    raw_args: str,
) -> None:
    """Execute a tool and display the response."""

    # ── Parse arguments ──────────────────────────────────────
    arguments: dict = {}
    if raw_args.strip():
        try:
            arguments = json.loads(raw_args.strip())
            if not isinstance(arguments, dict):
                print_error("Arguments must be a JSON object  e.g.  {\"key\": \"value\"}")
                return
        except json.JSONDecodeError as exc:
            print_error(f"Invalid JSON: {exc}")
            print_info('Arguments must be valid JSON.  e.g.  {"phone_number": "9999999999"}')
            return

    # ── Validate tool exists ──────────────────────────────────
    tools_response = await session.list_tools()
    tool_names     = [t.name for t in tools_response.tools]
    if tool_name not in tool_names:
        print_error(f"Unknown tool: '{tool_name}'")
        print_info(f"Available tools: {', '.join(tool_names)}")
        return

    # ── Validate required fields ──────────────────────────────
    match  = next(t for t in tools_response.tools if t.name == tool_name)
    schema = match.inputSchema or {}
    required_fields = schema.get("required", [])
    missing = [f for f in required_fields if f not in arguments]
    if missing:
        print_error(f"Missing required argument(s): {', '.join(missing)}")
        print_info(f"Run 'describe {tool_name}' to see all required parameters.")
        return

    # ── Execute ───────────────────────────────────────────────
    print()
    print_info(f"Calling  {bold(tool_name)}  …")
    print(f"  {dim('args:')} {dim(json.dumps(arguments))}")
    print_separator()

    try:
        result = await session.call_tool(tool_name, arguments=arguments)

        print(f"  {bold('Response:')}")
        print()

        for content_block in result.content:
            if hasattr(content_block, "text"):
                print_json(content_block.text)
            else:
                print(f"  {content_block}")

    except Exception as exc:
        print_error(f"Tool call failed: {exc}")

    print_separator()
    print()


# ─────────────────────────────────────────────
#  Input parsing
# ─────────────────────────────────────────────
def parse_input(raw: str) -> tuple[str, list[str]]:
    """
    Split user input into (command, rest_tokens).

    Handles:
        fetch_tools
        call register_user {"phone_number": "999"}
        describe get_user
    """
    raw = raw.strip()
    if not raw:
        return "", []

    parts   = raw.split(None, 2)      # max 3 parts: cmd  tool_name  json_blob
    command = parts[0].lower()
    rest    = parts[1:] if len(parts) > 1 else []
    return command, rest


# ─────────────────────────────────────────────
#  REPL loop
# ─────────────────────────────────────────────
async def repl(session: ClientSession) -> None:
    """Read-Evaluate-Print-Loop for the console."""
    print(BANNER)
    print_success("Server connected and ready.")
    print()

    while True:
        try:
            raw = input(f"{bold(cyan('>>>'))} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print_info("Session ended.")
            break

        if not raw:
            continue

        command, rest = parse_input(raw)

        # ── exit / quit ────────────────────────────────────
        if command in ("exit", "quit"):
            print_info("Goodbye! 👋")
            break

        # ── help ───────────────────────────────────────────
        elif command == "help":
            print(HELP_TEXT)

        # ── clear ──────────────────────────────────────────
        elif command == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)

        # ── fetch_tools ────────────────────────────────────
        elif command == "fetch_tools":
            await cmd_fetch_tools(session)

        # ── describe <tool_name> ───────────────────────────
        elif command == "describe":
            if not rest:
                print_error("Usage:  describe <tool_name>")
            else:
                await cmd_describe(session, rest[0])

        # ── call <tool_name> <json> ────────────────────────
        elif command == "call":
            if not rest:
                print_error("Usage:  call <tool_name> <json_args>")
                print_info('Example: call get_user {"phone_number": "9999999999"}')
            else:
                tool_name = rest[0]
                raw_args  = rest[1] if len(rest) > 1 else "{}"
                await cmd_call_tool(session, tool_name, raw_args)

        # ── unknown ────────────────────────────────────────
        else:
            print_error(f"Unknown command: '{command}'")
            print_info("Type 'help' to see available commands.")
            print()


# ─────────────────────────────────────────────
#  Server bootstrap
# ─────────────────────────────────────────────
def resolve_server_script() -> Path:
    """
    Locate src/server.py relative to this file.
    Works whether console.py is in the project root or anywhere else.
    """
    here        = Path(__file__).parent.resolve()
    server_path = here / "server.py"
    if not server_path.exists():
        raise FileNotFoundError(
            f"Cannot find server script at: {server_path}\n"
            f"Make sure console.py is in the telemedicine-mcp-server root directory."
        )
    return server_path


async def main() -> None:
    """Spawn the MCP server subprocess and connect to it."""
    # Resolve server path
    try:
        server_script = resolve_server_script()
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    # Find python executable (prefer the one running this script)
    python_bin = sys.executable or shutil.which("python3") or "python"

    print(f"\n  {bold('🚀 Spawning MCP server…')}")
    print(f"  {dim('script :')} {server_script}")
    print(f"  {dim('python :')} {python_bin}")
    print()

    server_params = StdioServerParameters(
        command=python_bin,
        args=[str(server_script)],
        env=None,          # inherit environment (picks up .env via dotenv inside server)
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                # MCP handshake
                await session.initialize()

                # Hand off to the interactive loop
                await repl(session)

    except FileNotFoundError:
        print_error(f"Could not launch Python interpreter: {python_bin}")
        print_info("Make sure your virtual environment is activated.")
        sys.exit(1)

    except ConnectionError as exc:
        print_error(f"Connection error: {exc}")
        sys.exit(1)

    except Exception as exc:
        print_error(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())