"""
Tools package for the telemedicine MCP server.

This package consolidates all available tools from different modules:
- Patient tools: register_user, get_user
- Hospital tools: fetch_hospitals, fetch_specialization
- Booking tools: fetch_pharmacy, do_booking
"""
from .patient_tools import PATIENT_TOOLS, PATIENT_TOOL_HANDLERS
from .hospital_tools import HOSPITAL_TOOLS, HOSPITAL_TOOL_HANDLERS
from .booking_tools import BOOKING_TOOLS, BOOKING_TOOL_HANDLERS

# Consolidate all tools
ALL_TOOLS = PATIENT_TOOLS + HOSPITAL_TOOLS + BOOKING_TOOLS

# Consolidate all tool handlers
ALL_TOOL_HANDLERS = {
    **PATIENT_TOOL_HANDLERS,
    **HOSPITAL_TOOL_HANDLERS,
    **BOOKING_TOOL_HANDLERS,
}

__all__ = [
    "ALL_TOOLS",
    "ALL_TOOL_HANDLERS",
    "PATIENT_TOOLS",
    "PATIENT_TOOL_HANDLERS",
    "HOSPITAL_TOOLS",
    "HOSPITAL_TOOL_HANDLERS",
    "BOOKING_TOOLS",
    "BOOKING_TOOL_HANDLERS",
]