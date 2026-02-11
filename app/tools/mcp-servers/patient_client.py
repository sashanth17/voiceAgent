"""
Patient-related tools for the telemedicine MCP server.
"""
from typing import Any, Dict
from mcp.server import Server
from mcp.types import Tool, TextContent
import json
from src.utils import api_client


# Tool definitions
REGISTER_USER_TOOL = Tool(
    name="register_user",
    description=(
        "Register a new patient in the telemedicine system. "
        "Requires phone number, name, age, gender, and pincode."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Patient's phone number (mobile number)",
            },
            "name": {
                "type": "string",
                "description": "Patient's full name",
            },
            "age": {
                "type": "integer",
                "description": "Patient's age",
                "minimum": 0,
                "maximum": 150,
            },
            "gender": {
                "type": "string",
                "description": "Patient's gender",
                "enum": ["male", "female", "other"],
            },
            "pincode": {
                "type": "string",
                "description": "Patient's area pincode",
            },
        },
        "required": ["phone_number", "name", "age", "gender", "pincode"],
    },
)

GET_USER_TOOL = Tool(
    name="get_user",
    description="Retrieve patient information by phone number.",
    inputSchema={
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Patient's phone number to retrieve information",
            },
        },
        "required": ["phone_number"],
    },
)


async def register_user(arguments: Dict[str, Any]) -> str:
    """
    Register a new patient.
    
    Args:
        arguments: Dictionary containing:
            - phone_number: Patient's phone number
            - name: Patient's name
            - age: Patient's age
            - gender: Patient's gender
            - pincode: Patient's pincode
    
    Returns:
        JSON string with registration result
    """
    try:
        # Prepare request data
        data = {
            "phone_number": arguments["phone_number"],
            "mobileno": arguments["phone_number"],  # Using phone_number as mobileno
            "name": arguments["name"],
            "age": arguments["age"],
            "gender": arguments["gender"],
            "pincode": arguments["pincode"],
        }
        
        # Make POST request
        response = await api_client.post("patient/register", data=data)
        
        # Check if registration was successful
        # Note: The API returns {message: 'registration sucessfull'} on success
        result = {
            "success": True,
            "message": response.get("message", "Registration completed"),
            "data": data,
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": "Failed to register user",
        }
        return json.dumps(error_result, indent=2)


async def get_user(arguments: Dict[str, Any]) -> str:
    """
    Retrieve patient information by phone number.
    
    Args:
        arguments: Dictionary containing:
            - phone_number: Patient's phone number
    
    Returns:
        JSON string with patient information
    """
    try:
        phone_number = arguments["phone_number"]
        
        # Make GET request with phone_number as query parameter
        response = await api_client.get(
            "patient",
            params={"phone_number": phone_number}
        )
        
        # Format the response
        result = {
            "success": True,
            "data": response,
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"Failed to retrieve user with phone number: {arguments.get('phone_number')}",
        }
        return json.dumps(error_result, indent=2)


# Tool handler mapping
PATIENT_TOOLS = [REGISTER_USER_TOOL, GET_USER_TOOL]

PATIENT_TOOL_HANDLERS = {
    "register_user": register_user,
    "get_user": get_user,
}