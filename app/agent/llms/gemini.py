# app/agent/llm.py
from typing import AsyncGenerator
import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-lite"


class GeminiLLM:
    def __init__(self):
        print("api key:"+os.environ.get('GOOGLE_API_KEY'))
        genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    generation_config={
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 256,
        "response_mime_type": "application/json"
    }
)

    async def complete(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        response = await self.model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text