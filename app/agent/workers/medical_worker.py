import json
from app.agent.state import AgentState
from app.agent.llms.biobert import get_biobert_classifier
from app.agent.llms.gemini import GeminiLLM
from app.agent.logger import logger

from app.agent.llms.symptom_extractor import get_symptom_extractor

# --- 1. Symptom Extraction Agent ---
async def symptom_extractor(state: AgentState) -> AgentState:
    """
    Extracts symptoms using specialized NER models as primary, 
    with LLM as an automatic fallback.
    """
    input_text = state.get("query") or state.get("user_message", "")
    existing_symptoms = list(state.get("symptoms") or [])
    last_target = state.get("missing_information")
    
    new_symptoms = []
    
    # --- PHASE 1: Try Specialized Main Model (from symptom/ directory) ---
    try:
        ner_system = get_symptom_extractor()
        if ner_system:
            # Extract using the production BioBERT-based system
            entities = ner_system.extract(input_text, confidence_threshold=0.4)
            if entities:
                extracted_texts = [e.text.lower() for e in entities if e.label in ["SYMPTOM", "DISEASE"]]
                new_symptoms.extend(extracted_texts)
                logger.info(f"Main Model Extracted: {extracted_texts}")
    except Exception as e:
        logger.error(f"Main model extraction failed: {e}")

    # --- PHASE 2: Fallback to LLM if no symptoms found or for stateful logic ---
    # We always run LLM fallback if new_symptoms is empty OR if we are waiting for a yes/no
    # because the local model doesn't handle "Yes, I have that" context.
    
    is_answering_question = last_target is not None and len(input_text.split()) < 5
    
    if not new_symptoms or is_answering_question:
        logger.info("Triggering LLM Fallback for Extraction...")
        llm = GeminiLLM()
        prompt = f"""
        You are a Medical Entity Extractor.
        Task: Extract symptoms from the input.
        
        Context:
        - We just asked the user about: "{last_target}"
        - If the user says "yes" or confirms, ADD "{last_target}" to the symptoms.
        - If the user says "no", do NOT add it.
        - Extract any other specific symptoms mentioned.
        
        Existing symptoms: {json.dumps(existing_symptoms)}
        Input Text: "{input_text}"
        
        Rules:
        - Normalize text (e.g., "head hurting" -> "headache").
        - Return raw JSON only.
        
        Output:
        {{
            "symptoms": ["headache", "fatigue"]
        }}
        """
        try:
            response_text = await llm.complete(prompt)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                data = json.loads(response_text[start_idx : end_idx + 1])
                llm_symptoms = data.get("symptoms", [])
                new_symptoms.extend(llm_symptoms)
                logger.info(f"LLM Fallback Extracted: {llm_symptoms}")
        except Exception as e:
            logger.error(f"LLM Fallback failed: {e}")

    # Final Merge & Deduplication
    normalized = [s.lower().strip() for s in new_symptoms]
    updated_symptoms = list(set(existing_symptoms + normalized))
    
    logger.info(f"Total Combined Symptoms: {updated_symptoms}")
    
    # Return only the update
    return {"symptoms": updated_symptoms}

# --- 2. Disease Inference Agent ---
async def disease_inference(state: AgentState) -> AgentState:
    """
    Uses BioClinicalBERT to match symptoms against disease knowledge base.
    """
    symptoms = state.get("symptoms", [])
    if not symptoms:
        return state
        
    # Create a query string from symptoms for semantic matching
    symptom_query = ", ".join(symptoms)
    
    classifier = get_biobert_classifier()
    # Get Top 3 matches
    predictions = classifier.predict_top_k(symptom_query, k=3)
    
    predictions = classifier.predict_top_k(symptom_query, k=3)
    logger.info(f"Disease Inference: {predictions}")
    
    return {"diagnosis_probabilities": predictions}

# --- 3. Urgency Scoring Agent ---
async def urgency_scorer(state: AgentState) -> AgentState:
    """
    Calculates urgency score (0-100) based on keywords and disease severity.
    Deterministic Logic.
    """
    symptoms = [str(s).lower() for s in state.get("symptoms", []) if s]
    predictions = state.get("diagnosis_probabilities", [])
    
    score = 0
    
    # 1. Red Flags (Base +50)
    red_flags = ["chest pain", "breathing", "unconscious", "bleeding", "severe", "heart", "stroke", "vision loss"]
    for sym in symptoms:
        if any(rf in sym for rf in red_flags):
            score += 50
            break 
            
    # 2. Disease Severity
    if predictions:
        top_disease = predictions[0]
        # We DO NOT use this for user output, only for internal awareness
        confidence = top_disease["confidence"]
        disease_name = top_disease["disease"].lower()
        
        CRITICAL_CONDITIONS = ["heart attack", "stroke", "pneumonia", "appendicitis"]
        for crit in CRITICAL_CONDITIONS:
            if crit in disease_name:
                score += 50  # High urgency for critical matches
                logger.info(f"Critical disease match: {disease_name}")
                break
        
    # 3. Duration Check (Simple Heuristic)
    # Check if user message implies long duration
    duration_keywords = ["days", "week", "month", "long time"]
    user_msg_lower = state.get("user_message", "").lower()
    
    # If "4 days", "5 days", "week" -> bump urgency
    import re
    duration_match = re.search(r"(\d+)\s*days?", user_msg_lower)
    if duration_match:
        days = int(duration_match.group(1))
        if days >= 3:
            score += 15 # Persistent symptoms
            
    if "week" in user_msg_lower or "month" in user_msg_lower:
         score += 20
            
    urgency_score = min(int(score), 100)
    logger.info(f"Urgency Score: {urgency_score}")
    
    return {"urgency_score": urgency_score}

# --- 4. Symptom Completion (Reasoning) ---
async def symptom_completion(state: AgentState) -> AgentState:
    """
    Decides NEXT STEP: Ask more, Booking, or Advice.
    Implements Loop Detection.
    """
    urgency = state.get("urgency_score", 0)
    predictions = state.get("diagnosis_probabilities", [])
    current_symptoms = list(state.get("symptoms") or [])
    asked_questions = list(state.get("asked_questions") or [])
    
    logger.info(f"[Symptom Completion] Entry - Asked Questions: {asked_questions}")
    logger.info(f"[Symptom Completion] Entry - Symptoms: {current_symptoms}")
    
    # CONSTANTS
    MIN_QUESTIONS = 2
    MAX_QUESTIONS = 5 # Increased slightly to ensure thorough triage
    
    # Rule 1: Safety First
    if urgency >= 80:
        state["next_step"] = "escalate"
        return state
        
    if not predictions:
        state["next_step"] = "ask" 
        state["missing_information"] = "main symptoms"
        return state

    top_disease = predictions[0]
    confidence = top_disease["confidence"]
    disease_name = top_disease["disease"]
    
    questions_count = len(asked_questions)
    
    # Rule 2: Force Minimum Interaction
    # If we reached MAX, stop regardless
    if questions_count >= MAX_QUESTIONS:
        logger.info("Max questions reached. Forcing decision.")
        state["next_step"] = "booking" if urgency > 40 else "advice"
        return state
        
    # Rule 3: Confidence Threshold
    if confidence > 0.92 and questions_count >= MIN_QUESTIONS:
        state["next_step"] = "booking" if urgency > 40 else "advice"
        return state
        
    # Rule 4: Find Missing Information (STRICT)
    classifier = get_biobert_classifier()
    expected_symptoms = classifier.get_symptoms_for_disease(disease_name)
    
    # Normalize everything for comparison to prevent overlaps like "chest_pain" vs "chest pain"
    def normalize(s: str) -> str:
        return s.replace("_", " ").lower().strip()

    asked_normalized = {normalize(q) for q in asked_questions}
    known_normalized = {normalize(s) for s in current_symptoms}
    
    target = None
    for expected in expected_symptoms:
        clean_expected = normalize(expected)
        
        # SKIP if we already asked OR if user already mentioned it
        if clean_expected not in asked_normalized and clean_expected not in known_normalized:
            target = clean_expected
            break
             
    if target:
        logger.info(f"Next symptom to ask: {target}")
        # Persist the fact that we are about to ask this
        asked_questions.append(target)
        return {
            "next_step": "ask",
            "missing_information": target,
            "asked_questions": asked_questions
        }
    else:
        # No new symptoms found in the DB for this disease, move to recommendation
        logger.info("No more symptoms to ask for this condition.")
        decision = "booking" if urgency > 40 else "advice"
        return {"next_step": decision}

# --- 5. Response Generator (LLM) ---
async def response_generator(state: AgentState) -> AgentState:
    """
    Generates the final natural language response based on the decision.
    CRITICAL: NEVER mentions the predicted disease name (e.g. AIDS, Cancer).
    """
    step = state.get("next_step")
    urgency = state.get("urgency_score", 0)
    llm = GeminiLLM()
    response = ""
    
    if step == "escalate":
        response = "URGENT: Your reported symptoms indicate a potential medical emergency. Please visit the nearest hospital or contact emergency services immediately."
    
    elif step == "booking":
        # Moderate Urgency -> Suggest Doctor
        response = "Given the duration and nature of your symptoms, it is advisable to consult a doctor for a proper evaluation. Would you like assistance in booking an appointment?"
        
    elif step == "advice":
        # Low/Moderate Urgency -> Home Care / Pharmacy
        if urgency > 30:
            response = "For these symptoms, you might consider visiting a nearby pharmacy for over-the-counter relief. If symptoms persist or worsen, please see a doctor."
        else:
            response = "It sounds like you may be experiencing some general discomfort. Ensure you get plenty of rest and stay hydrated. If you don't see improvement in 24 hours, consult a physician."
        
    elif step == "ask":
        target = state.get("missing_information", "more details")
        prompt = f"""
        You are a professional Medical Triage Agent.
        Task: Ask ONE simple question to check if the user has this specific symptom: '{target}'.
        
        Context: 
        - Patient report: {state.get('symptoms', [])}
        
        Constraints:
        - Do NOT mention any disease names.
        - Do NOT diagnose.
        - Be strictly professional and concise (max 15 words).
        - Example: "Do you also have a sore throat?"
        """
        response = await llm.complete(prompt)
        response = response.strip('"')

    response = response.strip('"').strip()
    if not response:
        response = "I'm processing your symptoms. Could you provide a bit more detail?"

    return {
        "final_response": response,
        "agent_response": response
    }
