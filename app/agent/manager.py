"""
Manager Agent - The orchestrator.

Responsibilities:
- Own conversation memory
- Route to workers or respond directly
- Synthesize final user-facing responses
- Control the entire workflow
"""
from typing import List, Dict, Any, Optional
from app.agent.logger import logger
from app.agent.workers.symptom_analyzer import SymptomAnalyzerAgent
from app.agent.workers.booking_agent import BookingAgent
from app.config import config
import asyncio
import re


class ManagerAgent:
    """
    Manager agent that orchestrates the multi-agent workflow.
    
    Owns:
    - Conversation memory (user + manager messages only)
    - Routing logic
    - Worker invocation
    - Final response generation
    """
    
    def __init__(self, connection_id: str):
        """
        Initialize manager for a connection.
        
        Args:
            connection_id: Unique WebSocket connection identifier
        """
        self.connection_id = connection_id
        self.model_config = config.manager_model
        
        # Conversation memory (user + final manager responses ONLY)
        self.memory: List[Dict[str, str]] = []
        
        # Worker agents (stateless)
        self.symptom_analyzer = SymptomAnalyzerAgent()
        self.booking_agent = BookingAgent()
        
        logger.info(f"[Manager {connection_id}] Initialized with model: {self.model_config.name}")
    
    async def process_message(self, user_message: str) -> str:
        """
        Process user message through routing logic.
        
        Args:
            user_message: User's input
        
        Returns:
            Final response to send to user
        """
        logger.info(f"[Manager {self.connection_id}] Processing: {user_message[:50]}...")
        
        # Add user message to memory
        self.memory.append({"role": "user", "content": user_message})
        
        # Routing decision
        route = await self._decide_route(user_message)
        logger.info(f"[Manager {self.connection_id}] Route decision: {route}")
        
        final_response = ""
        
        try:
            if route == "direct":
                # Manager responds directly (greeting, general question, etc.)
                final_response = await self._generate_direct_response(user_message)
            
            elif route == "symptom_analysis":
                # Invoke symptom analyzer → synthesize response
                final_response = await self._handle_symptom_analysis(user_message)
            
            elif route == "booking":
                # Invoke symptom analyzer → booking agent → synthesize response
                final_response = await self._handle_booking(user_message)
            
            else:
                # Fallback
                final_response = "I'm not sure how to help with that. Could you rephrase?"
        
        except Exception as e:
            logger.error(f"[Manager {self.connection_id}] Error processing message: {e}")
            final_response = "I encountered an issue processing your request. Please try again."
        
        # Add final response to memory
        self.memory.append({"role": "assistant", "content": final_response})
        
        # Trim memory if too long
        if len(self.memory) > config.max_memory_messages:
            self.memory = self.memory[-config.max_memory_messages:]
        
        logger.info(f"[Manager {self.connection_id}] Final response: {final_response[:50]}...")
        
        return final_response
    
    async def _decide_route(self, user_message: str) -> str:
        """
        Decide routing based on user message content.
        
        Returns:
            "direct" | "symptom_analysis" | "booking"
        """
        message_lower = user_message.lower()
        
        # Booking indicators
        booking_keywords = ["book", "appointment", "schedule", "reservation", "slot"]
        if any(kw in message_lower for kw in booking_keywords):
            return "booking"
        
        # Symptom indicators
        symptom_keywords = ["pain", "symptom", "sick", "fever", "cough", "hurt", "ache", "feel"]
        if any(kw in message_lower for kw in symptom_keywords):
            return "symptom_analysis"
        
        # Greeting/general indicators
        greeting_keywords = ["hello", "hi", "hey", "help", "what can you", "who are you"]
        if any(kw in message_lower for kw in greeting_keywords):
            return "direct"
        
        # Default to direct for unclear cases
        return "direct"
    
    async def _generate_direct_response(self, user_message: str) -> str:
        """
        Generate direct response without worker invocation.
        
        Used for greetings, general questions, irrelevant queries.
        """
        logger.info(f"[Manager {self.connection_id}] Generating direct response")
        
        # Build context from memory
        context = self._build_memory_context()
        
        prompt = f"""{context}

User: {user_message}

You are a helpful medical assistant. Respond naturally and helpfully.
Assistant:"""
        
        # Call manager LLM
        response = await self._call_manager_llm(prompt)
        
        return response.strip()
    
    async def _handle_symptom_analysis(self, user_message: str) -> str:
        """
        Handle symptom analysis workflow.
        
        Flow:
        1. Extract symptoms from message
        2. Call SymptomAnalyzerAgent
        3. Synthesize user-friendly response
        """
        logger.info(f"[Manager {self.connection_id}] Handling symptom analysis")
        
        # Extract symptoms (simplified - in production, use LLM extraction)
        symptoms = user_message
        
        # Call worker
        analysis_result = await self.symptom_analyzer.analyze({
            "symptoms": symptoms,
            "patient_info": {}
        })
        
        # Synthesize final response
        final_response = await self._synthesize_symptom_response(user_message, analysis_result)
        
        return final_response
    
    async def _handle_booking(self, user_message: str) -> str:
        """
        Handle booking workflow.
        
        Flow:
        1. Extract symptoms from message/memory
        2. Call SymptomAnalyzerAgent
        3. Call BookingAgent with analysis
        4. Synthesize final response
        """
        logger.info(f"[Manager {self.connection_id}] Handling booking request")
        
        # Extract symptoms from current message or recent memory
        symptoms = self._extract_symptoms_from_context(user_message)
        
        # Step 1: Analyze symptoms
        analysis_result = await self.symptom_analyzer.analyze({
            "symptoms": symptoms,
            "patient_info": {}
        })
        
        # Step 2: Book appointment
        booking_result = await self.booking_agent.book_appointment({
            "symptom_analysis": analysis_result,
            "patient_preferences": {},
            "user_request": user_message
        })
        
        # Step 3: Synthesize final response
        final_response = await self._synthesize_booking_response(
            user_message,
            analysis_result,
            booking_result
        )
        
        return final_response
    
    async def _synthesize_symptom_response(
        self,
        user_message: str,
        analysis: Dict[str, Any]
    ) -> str:
        """
        Synthesize user-friendly response from symptom analysis.
        
        Manager NEVER exposes raw worker output.
        """
        prompt = f"""You are a medical assistant. A patient described symptoms and received this analysis:

Urgency Score: {analysis['urgency_score']}
Recommended Specialist: {analysis['recommended_specialist']}
Summary: {analysis['summary']}

User's original message: {user_message}

Generate a compassionate, clear response to the patient that:
1. Acknowledges their symptoms
2. Provides the key insights
3. Recommends next steps
4. Does NOT use technical jargon

Response:"""
        
        response = await self._call_manager_llm(prompt)
        
        return response.strip()
    
    async def _synthesize_booking_response(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        booking: Dict[str, Any]
    ) -> str:
        """
        Synthesize final booking response.
        """
        if booking["booking_status"] == "success":
            details = booking["appointment_details"]
            prompt = f"""You are a medical assistant. A patient requested an appointment and it was successfully booked.

Appointment Details:
- Date: {details.get('date')}
- Time: {details.get('time')}
- Specialist: {details.get('specialist')}
- Location: {details.get('location')}

User's request: {user_message}

Generate a friendly confirmation message that includes all details clearly.

Response:"""
        else:
            prompt = f"""You are a medical assistant. A patient requested an appointment but it could not be booked.

Error: {booking['message']}

User's request: {user_message}

Generate a sympathetic response explaining the issue and suggesting alternatives.

Response:"""
        
        response = await self._call_manager_llm(prompt)
        
        return response.strip()
    
    def _build_memory_context(self) -> str:
        """Build conversation context from memory."""
        if not self.memory:
            return ""
        
        context_lines = []
        for msg in self.memory[-6:]:  # Last 6 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_lines)
    
    def _extract_symptoms_from_context(self, user_message: str) -> str:
        """
        Extract symptoms from current message and recent memory.
        
        Simplified implementation - in production, use LLM extraction.
        """
        # Check recent messages for symptom keywords
        recent_messages = [msg["content"] for msg in self.memory[-4:] if msg["role"] == "user"]
        recent_messages.append(user_message)
        
        symptom_keywords = ["pain", "symptom", "sick", "fever", "cough", "hurt", "ache"]
        
        symptom_messages = [
            msg for msg in recent_messages
            if any(kw in msg.lower() for kw in symptom_keywords)
        ]
        
        return " ".join(symptom_messages) if symptom_messages else user_message
    
    async def _call_manager_llm(self, prompt: str) -> str:
        """
        Call manager LLM (Ollama Phi3).
        
        Replace with actual Ollama API call:
```python
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.model_config.base_url}/api/generate",
                json={
                    "model": self.model_config.name,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                data = await response.json()
                return data["response"]
```
        """
        # SIMULATION - Replace with actual Ollama call
        await asyncio.sleep(0.3)
        
        # Mock intelligent routing-aware responses
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! I'm your medical assistant. I can help you with symptom analysis and booking appointments. How can I assist you today?"
        
        if "urgency_score" in prompt.lower():
            return "Based on your symptoms, I recommend seeing a General Practitioner soon. Your symptoms suggest moderate urgency. Would you like me to help you book an appointment?"
        
        if "appointment details" in prompt.lower():
            return "Great! I've successfully booked your appointment with Dr. Smith at City Medical Center on February 5th at 2:00 PM. You'll receive a confirmation email shortly."
        
        return "I'm here to help with your medical needs. Could you tell me more about your symptoms or what you need assistance with?"
    
    async def cleanup(self):
        """
        Cleanup manager resources on disconnect.
        """
        logger.info(f"[Manager {self.connection_id}] Cleaning up")
        self.memory.clear()