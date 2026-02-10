from app.agent.state import AgentState
from app.agent.llms.biobert import get_biobert_classifier
from app.agent.llms.hybrid_llm import HybridLLM
    
    # 2. Use LLM to format a conversational response
    llm = HybridLLM()
    
    prompt = f"""
    You are a Clinical Reasoning Assistant. 
    BioClinicalBERT has analyzed the user's symptoms and found these top matches:
    {json.dumps(predictions, indent=2)}
    
    User Query: "{query}"
    
    Your task:
    1. Explain these findings to the user professionally and empathetically.
    2. Mention the confidence scores in a natural way.
    3. Ask 1-2 clarifying questions about other symptoms related to the top match.
    4. Include a medical disclaimer.
    """
    
    response = await llm.complete(prompt)
    
    state["agent_response"] = response
    return state