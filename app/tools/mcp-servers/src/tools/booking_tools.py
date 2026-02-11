"""
Booking and pharmacy-related tools for the telemedicine MCP server.

This module contains placeholder implementations for:
- fetch_pharmacy: Get list of pharmacies by pincode
- do_booking: Book an appointment at a hospital
"""
from typing import Any, Dict
from mcp.types import Tool
import json
from src.utils import api_client


# Tool definitions
FETCH_PHARMACY_TOOL = Tool(
    name="fetch_pharmacy",
    description=(
        "Fetch list of pharmacies in a specific area by pincode. "
        "Returns pharmacy details including ID, name, address, and contact information."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "pincode": {
                "type": "string",
                "description": "Area pincode to search for pharmacies",
            },
        },
        "required": ["pincode"],
    },
)


async def fetch_pharmacy(arguments: Dict[str, Any]) -> str:
    """
    Fetch pharmacies by pincode using the actual API endpoint.

    Args:
        arguments: Dictionary containing:
            - pincode (int | str): Area pincode

    Returns:
        JSON string in the required output format:
        {
            "results": [...],
            "message": ""
        }
    """
    try:
        # 1. Validate input
        pincode = arguments.get("pincode")
        if not pincode:
            raise ValueError("pincode is required")

        # 2. Call actual API
        response = await api_client.get(
            endpoint="pharmacies/api/search/",
            params={"pincode": pincode},
        )

        # 3. Normalize response (defensive, MCP-safe)
        result = {
            "results": response.get("results", []),
            "message": response.get("message", ""),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_result = {
            "results": [],
            "message": f"Failed to fetch pharmacies: {str(e)}",
        }
        return json.dumps(error_result, indent=2)


DO_BOOKING_TOOL = Tool(
    name="do_booking",
    description=(
        "Book a hospital appointment. "
        "Returns token number and allocated doctor."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "hospital_id": {
                "type": "number",
                "description": "Unique identifier of the hospital",
            },
            "patient_contact": {
                "type": "string",
                "description": "Patient phone number",
            },
            "appointment_date": {
                "type": "string",
                "description": "Appointment date (YYYY-MM-DD)",
            },
            "urgency_score": {
                "type": "number",
                "description": "Urgency score (0–100)",
                "default": 50
            }
        },
        "required": ["hospital_id", "patient_contact", "appointment_date"],
    },
)


async def do_booking(arguments: Dict[str, Any]) -> str:
    """
    Create a hospital appointment booking.
    Returns token number and allocated doctor.
    """
    try:
        # Extract arguments safely
        hospital_id = arguments.get("hospital_id")
        patient_contact = arguments.get("patient_contact")
        appointment_date = arguments.get("appointment_date")
        urgency_score = arguments.get("urgency_score", 50)

        if not all([hospital_id, patient_contact, appointment_date]):
             return json.dumps({
                 "success": False, 
                 "message": "Missing required fields: hospital_id, patient_contact, or appointment_date"
             })

        if isinstance(appointment_date, str):
             appointment_date = appointment_date.strip().strip('"')

        payload = {
            "hospital_id": hospital_id,
            "patient_contact": patient_contact,
            "appointment_date": appointment_date,
            "urgency_score": urgency_score,
        }

        # Call the actual backend API
        # Note: api_client.post returns a dict, not a response object usually in this codebase (based on other files)
        response = await api_client.post(
            "/appointments/",
            data=payload
        )
        
        return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "message": "Failed to create appointment",
                "error": str(e)
            },
            indent=2
        )

# Tool handler mapping
BOOKING_TOOLS = [FETCH_PHARMACY_TOOL, DO_BOOKING_TOOL]

BOOKING_TOOL_HANDLERS = {
    "fetch_pharmacy": fetch_pharmacy,
    "do_booking": do_booking,
}