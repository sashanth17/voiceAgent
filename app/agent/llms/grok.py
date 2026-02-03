from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

def get_groq_llm(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    streaming: bool = False,
):
    """
    LangChain ChatGroq LLM.
    Supports tools, streaming, LangGraph.
    """
    return ChatGroq(
        api_key=os.environ.get('GROQ_API_KEY'),
        model=model,
        temperature=temperature,
        streaming=streaming,
    )