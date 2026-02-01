"""
Booking Agent Worker.

Stateless agent that handles appointment bookings.
"""
import json
from typing import Dict, Any
from app.agent.logger import logger
from app.config import config
import asyncio


class BookingAgent:
    """
    Handles appointment booking based on symptom analysis.
    
    Input: Structured context with symptom analysis + booking request
    Output: Strict JSON with booking status and details
    """
    
    def __init__(self):
        self.model_config = config.booking_agent_model
        logger.info(f"[BookingAgent] Initialized with model: {self.model_config.name}")
    
    async def book_appointment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process appointment booking request.
        
        Args:
            context: Dictionary with keys:
                - symptom_analysis: dict (from SymptomAnalyzerAgent)
                - patient_preferences: dict (date, time, location preferences)
                - user_request: str (original booking request)
        
        Returns:
            {
                "booking_status": "success" | "failed",
                "appointment_details": {...},
                "message": str
            }
        """
        symptom_analysis = context.get("symptom_analysis", {})
        preferences = context.get("patient_preferences", {})
        request = context.get("user_request", "")
        
        logger.info(f"[BookingAgent] Processing booking request: {request[:50]}...")
        
        # Build structured prompt
        prompt = self._build_prompt(symptom_analysis, preferences, request)
        
        try:
            # Call LLM
            response_json = await self._call_llm(prompt)
            
            # Validate and parse
            result = self._validate_response(response_json)
            
            if config.log_worker_outputs:
                logger.info(f"[BookingAgent] Result: {json.dumps(result, indent=2)}")
            
            return result
        
        except Exception as e:
            logger.error(f"[BookingAgent] Booking failed: {e}")
            raise
    
    def _build_prompt(self, symptom_analysis: Dict, preferences: Dict, request: str) -> str:
        """Build structured prompt for booking."""
        return f"""You are an appointment booking assistant. Process this booking request and return ONLY valid JSON.

Symptom Analysis:
{json.dumps(symptom_analysis, indent=2)}

Patient Preferences:
{json.dumps(preferences, indent=2) if preferences else "Not specified"}

User Request: {request}

Return STRICT JSON format:
{{
  "booking_status": "success" or "failed",
  "appointment_details": {{
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "specialist": "<specialist name>",
    "location": "<clinic/hospital>"
  }},
  "message": "<confirmation or error message>"
}}

JSON Response:"""
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (replace with actual implementation).
        Same pattern as SymptomAnalyzerAgent.
        """
        # SIMULATION - Replace with actual API call
        await asyncio.sleep(0.5)
        
        # Mock response
        return json.dumps({
            "booking_status": "success",
            "appointment_details": {
                "date": "2026-02-05",
                "time": "14:00",
                "specialist": "Dr. Smith (General Practitioner)",
                "location": "City Medical Center"
            },
            "message": "Appointment successfully booked"
        })
    
    def _validate_response(self, response: str) -> Dict[str, Any]:
        """Validate and parse LLM JSON response."""
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            
            # Validate required fields
            required_fields = ["booking_status", "appointment_details", "message"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate booking_status
            if data["booking_status"] not in ["success", "failed"]:
                raise ValueError("booking_status must be 'success' or 'failed'")
            
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"[BookingAgent] Invalid JSON response: {response}")
            raise ValueError(f"Failed to parse JSON: {e}")