# app/agent/llms/hybrid_llm.py
from typing import AsyncGenerator
import warnings
import google.generativeai as genai
import os
from dotenv import load_dotenv

from app.agent.llms.groq import get_groq_llm

load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

GEMINI_MODEL = "gemini-1.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

class HybridLLM:
    def __init__(self):
        # Primary Model: Groq
        self.model = get_groq_llm(model=GROQ_MODEL)
        
        # Fallback Model: Gemini
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("Warning: GOOGLE_API_KEY is not set!")
        genai.configure(api_key=api_key)
        self.fallback_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.9,
                "max_output_tokens": 512,
            }
        )

    async def complete(self, prompt: str) -> str:
        try:
            # Try Groq first
            response = await self.model.ainvoke(prompt)
            content = response.content
            if not isinstance(content, str):
                content = str(content)
            return content
        except Exception as e:
            print(f"Groq error: {e}. Falling back to Gemini.")
            try:
                # Fallback to Gemini
                response = await self.fallback_model.generate_content_async(prompt)
                return response.text
            except Exception as ge:
                print(f"Gemini fallback also failed: {ge}")
                return "I'm sorry, I'm having trouble connecting to my brain right now."

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            # Try Groq first
            async for chunk in self.model.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"Groq streaming error: {e}. Falling back to Gemini.")
            try:
                # Fallback to Gemini
                response = await self.fallback_model.generate_content_async(prompt, stream=True)
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as ge:
                print(f"Gemini streaming fallback also failed: {ge}")
                yield "I'm sorry, I'm having trouble connecting to my brain right now."
