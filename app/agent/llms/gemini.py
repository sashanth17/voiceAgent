# app/agent/llm.py
from typing import AsyncGenerator
import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()

from app.agent.llms.grok import get_groq_llm

GEMINI_MODEL = "gemini-1.5-flash"


class GeminiLLM:
    def __init__(self):
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("Warning: GOOGLE_API_KEY is not set!")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.9,
                "max_output_tokens": 512,
            }
        )
        # Fallback model (Groq)
        self.fallback_model = get_groq_llm(model="llama-3.3-70b-versatile")

    async def complete(self, prompt: str) -> str:
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            # Check for Resource Exhausted or other errors
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(f"Gemini Rate Limit Exceeded. Falling back to Groq.")
            else:
                print(f"Gemini error: {e}. Falling back to Groq.")
            
            # Fallback to Groq
            response = self.fallback_model.invoke(prompt)
            content = response.content
            if not isinstance(content, str):
                content = str(content)
            return content

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            response = await self.model.generate_content_async(prompt, stream=True)
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(f"Gemini Rate Limit Exceeded (Stream). Falling back to Groq.")
            else:
                print(f"Gemini streaming error: {e}. Falling back to Groq.")
                
            async for chunk in self.fallback_model.astream(prompt):
                yield chunk.content