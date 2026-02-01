"""
Symptom Analyzer Worker Agent.

Stateless agent that analyzes symptoms and returns structured JSON.
"""
import json
from typing import Dict, Any
from app.agent.logger import logger
from app.config import config
import asyncio


class SymptomAnalyzerAgent:
    """
    Analyzes patient symptoms and provides urgency scoring.
    
    Input: Structured context with symptoms
    Output: Strict JSON with urgency_score, specialist, summary
    """
    
    def __init__(self):
        self.model_config = config.symptom_analyzer_model
        logger.info(f"[SymptomAnalyzer] Initialized with model: {self.model_config.name}")
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze symptoms from structured context.
        
        Args:
            context: Dictionary with keys:
                - symptoms: str (patient's described symptoms)
                - patient_info: dict (age, gender, etc. - optional)
        
        Returns:
            {
                "urgency_score": float (0.0 to 1.0),
                "recommended_specialist": str,
                "summary": str
            }
        """
        symptoms = context.get("symptoms", "")
        patient_info = context.get("patient_info", {})
        
        logger.info(f"[SymptomAnalyzer] Analyzing symptoms: {symptoms[:50]}...")
        
        # Build structured prompt for worker
        prompt = self._build_prompt(symptoms, patient_info)
        
        try:
            # Call LLM (simulated - replace with actual API call)
            response_json = await self._call_llm(prompt)
            
            # Validate and parse response
            result = self._validate_response(response_json)
            
            if config.log_worker_outputs:
                logger.info(f"[SymptomAnalyzer] Result: {json.dumps(result, indent=2)}")
            
            return result
        
        except Exception as e:
            logger.error(f"[SymptomAnalyzer] Analysis failed: {e}")
            raise
    
    def _build_prompt(self, symptoms: str, patient_info: Dict) -> str:
        """Build structured prompt for symptom analysis."""
        return f"""You are a medical symptom analyzer. Analyze the following symptoms and return ONLY valid JSON.

Symptoms: {symptoms}
Patient Info: {json.dumps(patient_info) if patient_info else "Not provided"}

Return STRICT JSON format:
{{
  "urgency_score": <float 0.0-1.0>,
  "recommended_specialist": "<specialist type>",
  "summary": "<brief analysis>"
}}

JSON Response:"""
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (replace with actual implementation).
        
        For Gemini:
            import google.generativeai as genai
            genai.configure(api_key=self.model_config.api_key)
            model = genai.GenerativeModel(self.model_config.name)
            response = await model.generate_content_async(prompt)
            return response.text
        
        For Grok (via OpenAI-compatible API):
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.model_config.api_key, base_url="https://api.x.ai/v1")
            response = await client.chat.completions.create(...)
            return response.choices[0].message.content
        """
        # SIMULATION - Replace with actual API call
        await asyncio.sleep(0.5)
        
        # Mock response
        return json.dumps({
            "urgency_score": 0.7,
            "recommended_specialist": "General Practitioner",
            "summary": "Symptoms suggest moderate urgency requiring medical attention"
        })
    
    def _validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate and parse LLM JSON response.
        
        Raises:
            ValueError: If response is not valid JSON or missing required fields
        """
        try:
            # Clean response (remove markdown, whitespace)
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
            required_fields = ["urgency_score", "recommended_specialist", "summary"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate types
            if not isinstance(data["urgency_score"], (int, float)):
                raise ValueError("urgency_score must be a number")
            if not isinstance(data["recommended_specialist"], str):
                raise ValueError("recommended_specialist must be a string")
            if not isinstance(data["summary"], str):
                raise ValueError("summary must be a string")
            
            # Normalize urgency_score to 0.0-1.0
            data["urgency_score"] = max(0.0, min(1.0, float(data["urgency_score"])))
            
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"[SymptomAnalyzer] Invalid JSON response: {response}")
            raise ValueError(f"Failed to parse JSON: {e}")