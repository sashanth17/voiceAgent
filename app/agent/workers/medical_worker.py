import json
import re
from typing import Dict, Any, List
from app.agent.state import AgentState
from app.agent.llms.biobert import get_biobert_classifier
from app.agent.llms.hybrid_llm import HybridLLM
from app.agent.logger import logger

from app.agent.llms.symptom_extractor import get_symptom_extractor
from app.agent.utils.system_mapping import identify_system, SYSTEM_KEYWORDS

# --- Helper: Demographics Extraction ---
def extract_demographics(text: str) -> Dict[str, Any]:
    info = {}
    text_lower = text.lower()
    
    # Age
    # "I am 25", "25 years old", "age 25"
    age_match = re.search(r'\b(\d{1,3})\s*(?:years?|yrs?|yo)\b', text_lower)
    if not age_match:
        age_match = re.search(r'\bage\s*(\d{1,3})\b', text_lower)
    
    if age_match:
        try:
            val = int(age_match.group(1))
            if 0 <= val <= 120:
                info["age"] = val
        except:
            pass
            
    # Gender
    if any(w in text_lower for w in ["female", "woman", "girl", "lady"]):
        info["gender"] = "female"
    elif any(w in text_lower for w in ["male", "man", "boy", "gentleman"]):
        info["gender"] = "male"
        
    # Pregnancy
    if any(w in text_lower for w in ["pregnant", "pregnancy", "expecting baby"]):
        info["is_pregnant"] = True
        
    # Medical History (Simple Keywords)
    history = []
    conditions = {
        "diabetes": ["diabetes", "sugar", "diabetic"],
        "hypertension": ["bp", "blood pressure", "hypertension"],
        "asthma": ["asthma", "wheezing"],
        "heart_condition": ["heart disease", "cardiac"]
    }
    for cond, keywords in conditions.items():
        if any(k in text_lower for k in keywords):
            history.append(cond)
            
    if history:
        info["history"] = history
        
    return info

# --- 1. Symptom Extraction Agent ---
async def symptom_extractor(state: AgentState) -> AgentState:
    """
    Extracts symptoms using specialized NER models as primary, 
    with LLM as an automatic fallback.
    ALSO extracts basic patient info (age, gender, etc.) from natural text.
    """
    input_text = state.get("query") or state.get("user_message", "")
    existing_symptoms = list(state.get("symptoms") or [])
    last_target = state.get("missing_information")
    
    # --- Demographics Update ---
    current_info = state.get("patient_info") or {}
    new_info = extract_demographics(input_text)
    
    # Merge new info into current (don't overwrite if existing is better, unless updated)
    for k, v in new_info.items():
        if k == "history":
            # Append history items
            existing_hist = current_info.get("history", [])
            updated_hist = list(set(existing_hist + v))
            current_info["history"] = updated_hist
        else:
            current_info[k] = v
            
    state["patient_info"] = current_info
    logger.info(f"Patient Info Updated: {current_info}")
    
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

    # --- PHASE 1.5: Fuzzy Keyword Matching (Fast Fallback) ---
    # Optimized from medicare3 prototype
    classifier = get_biobert_classifier()
    if classifier and hasattr(classifier, 'all_symptoms'):
        all_possible_symptoms = classifier.all_symptoms
        found_keywords = set()
        
        # Normalize helper
        def normalize_key(s):
            return s.lower().replace('_', ' ').replace(' the ', ' ').replace(' in ', ' ').replace(' of ', ' ')
            
        text_norm = normalize_key(input_text)
        
        # Sort by length (desc) to match longest phrases first
        sorted_syms = sorted(all_possible_symptoms, key=len, reverse=True)
        
        for sym in sorted_syms:
            readable = sym.replace('_', ' ')
            readable_norm = normalize_key(readable)
            
            # Check for exact or normalized match
            if readable in input_text.lower() or readable_norm in text_norm:
                found_keywords.add(sym)
                
        if found_keywords:
            new_symptoms.extend(list(found_keywords))
            logger.info(f"Fuzzy Match Extracted: {found_keywords}")

    # --- PHASE 2: Fallback to LLM if STILL no symptoms found or for stateful logic ---
    # We always run LLM fallback if new_symptoms is empty OR if we are waiting for a yes/no
    # because the local model doesn't handle "Yes, I have that" context.
    
    is_answering_question = last_target is not None and not last_target.startswith("ask_") and len(input_text.split()) < 5
    
    if not new_symptoms or is_answering_question:
        logger.info("Triggering LLM Fallback for Extraction...")
        llm = HybridLLM()
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

    # Final Merge & Deduplication (Strict Normalization)
    def strict_normalize(s):
        return s.lower().replace('_', ' ').strip()
        
    normalized = [strict_normalize(s) for s in new_symptoms]
    existing = [strict_normalize(s) for s in existing_symptoms]
    
    # Combine and deduplicate
    current_set = set(existing)
    final_list = list(existing) # Start with what we had
    
    for s in normalized:
        if s not in current_set:
            final_list.append(s)
            current_set.add(s)
    
    logger.info(f"Total Combined Symptoms: {final_list}")
    
    # Return only the update
    return {"symptoms": final_list, "patient_info": current_info}

# --- 2. Disease Inference Agent ---
async def disease_inference(state: AgentState) -> AgentState:
    """
    Uses BioClinicalBERT to match symptoms against disease knowledge base.
    ENHANCED REASONING: Uses Physiological Anchoring (Rule 1, Rule 2).
    """
    symptoms = state.get("symptoms", [])
    user_message = state.get("user_message", "")
    
    if not symptoms:
        return state
        
    # --- Rule 1: Anchor Reasoning ---
    # Identify the physiological system from the user's initial complaint or full message
    phys_system = identify_system(user_message)
    logger.info(f"Physiological System Identified: {phys_system}")
    
    # Create a query string from symptoms for semantic matching
    symptom_query = ", ".join(symptoms)
    
    classifier = get_biobert_classifier()
    # Get Top 5 matches (increased from 3 to allow for re-ranking)
    predictions = classifier.predict_top_k(symptom_query, k=5)
    
    # --- Rule 2: Restrict Hypothesis (Re-ranking) ---
    # Boost diseases that are relevant to the identified system
    reranked = []
    max_raw_conf = 0.0
    
    for pred in predictions:
        disease = pred["disease"]
        conf = pred["confidence"]
        max_raw_conf = max(max_raw_conf, conf)
        
        # Simple heuristic: Check if disease name or its top symptoms match the system
        # Since we don't have disease->system mapping, we re-use identify_system on the disease name
        # + checking if disease name contains system keywords
        
        d_system = identify_system(disease)
        
        # --- Rule 10: Chronicity Penalty ---
        # Check if user mentioned chronic duration, and penalize acute diseases
        is_long_term = any(x in user_message.lower() for x in ['year', 'month', 'chronic', 'long time'])
        if is_long_term and 'acute' in disease.lower():
            conf *= 0.3
            logger.info(f"Penalizing {disease} (Acute) due to chronic presentation.")
            
        # --- Rule 12: Prevalence Boost (Common things are common) ---
        # Heuristic to prevent rare diseases (like HIV) from dominating generic symptoms (like fever)
        COMMON_CONDITIONS = ['Common Cold', 'Flu', 'Allergy', 'Migraine', 'Gastroenteritis', 'Bronchitis', 'Viral']
        if any(c.lower() in disease.lower() for c in COMMON_CONDITIONS):
            conf *= 1.2 # Boost common conditions
            
        # Boost if match
        if phys_system != "General" and (d_system == phys_system or phys_system.lower() in disease.lower()):
            conf *= 1.25 # 25% boost
            logger.info(f"Boosting {disease} ({d_system}) due to system match")
            
        reranked.append({"disease": disease, "confidence": conf})
        
    # Re-sort and take top 3
    reranked.sort(key=lambda x: x["confidence"], reverse=True)
    final_predictions = reranked[:3]
    
    # --- FALLBACK LOGIC (Optimization) ---
    # If the system-based boost leads to very low confidence (e.g. wrong system identified),
    # or if the top result is significantly worse than the raw best, revert to raw.
    
    top_boosted = final_predictions[0]["confidence"] if final_predictions else 0
    if top_boosted < 0.25 and max_raw_conf > top_boosted:
         logger.info("System anchoring yielded low confidence. Reverting to raw BioBERT predictions.")
         final_predictions = predictions[:3]
    
    logger.info(f"Disease Inference (Final): {final_predictions}")
    
    return {
        "diagnosis_probabilities": final_predictions,
        "physiological_system": phys_system
    }

# --- 3. Urgency Scoring Agent ---
async def urgency_scorer(state: AgentState) -> AgentState:
    """
    Calculates urgency score (0-100) using an ADAPTIVE model with SAFEGUARDS & DIMINISHING RETURNS.
    Safeguards: Negation filtering, tiered symptom scoring, and specific risk bonuses.
    """
    # Safeguard 1: Improved Negation Filtering (Language Nuance)
    raw_symptoms = state.get("symptoms", [])
    negations = ["no ", "not ", "dont ", "don't ", "without ", "denies "]
    symptoms = []
    for s in raw_symptoms:
        s_str = str(s).lower()
        # Check if any negation phrase appears BEFORE the symptom in the full text could be complex,
        # but here we check if the extracted string itself contains negations (common in extraction errors).
        if s_str and not any(n in s_str for n in negations):
             symptoms.append(s_str)
    
    predictions = state.get("diagnosis_probabilities", [])
    patient_info = state.get("patient_info", {})
    user_msg_lower = state.get("user_message", "").lower()
    
    # --- 1. Symptom Load (Diminishing Returns) ---
    # Fixes "Shopping List Inflation" AND "Ceiling Effect"
    # Logic: Core symptoms matter most. Endless list makes less difference.
    # Tier 1 (1-3 symptoms): 5 points each (High impact)
    # Tier 2 (4-6 symptoms): 3 points each (Moderate impact)
    # Tier 3 (7+ symptoms):  1 point each  (Low impact)
    symptom_load = 0
    count = len(symptoms)
    
    if count <= 3:
        symptom_load = count * 5
    elif count <= 6:
        symptom_load = (3 * 5) + ((count - 3) * 3)
    else:
        symptom_load = (3 * 5) + (3 * 3) + ((count - 6) * 1)
        
    symptom_load = min(symptom_load, 35) # Soft Max

    # --- 1.5 High Risk Specific Bonus ---
    # Fixes "Rare but Critical Clusters"
    # Specific clinical signs that aren't quite "Red Flags" (911) but strongly suggest doctor visit.
    high_risk_keywords = ["stiff neck", "confusion", "rash", "light sensitivity", "slurred", "drooping", "weakness", "numbness"]
    high_risk_score = 0
    combined_text = (user_msg_lower + " " + " ".join(symptoms)).lower()
    
    for key in high_risk_keywords:
        if key in combined_text:
            high_risk_score = 15
            break # Apply bonus once
    
    # --- 2. Severity Modifiers (Capped) ---
    severity_score = 0
    severity_weights = {
        # Low Urgency
        "mild": -5, "low": -5, "light": -5, "slight": -5, "bit": -2, "come and go": -5,
        # Moderate
        "moderate": 5, "constant": 5, "persistent": 5, "annoying": 2,
        # High Urgency
        "severe": 15, "heavy": 15, "intense": 15, "sharp": 10, "worst": 20, 
        "unbearable": 20, "agony": 20, "extreme": 15, "sudden": 10
    }
    
    detected_modifiers = []
    for word, weight in severity_weights.items():
        if word in user_msg_lower:
            severity_score += weight
            detected_modifiers.append(word)
            
    # Apply Cap (-10 to +20)
    severity_score = max(-10, min(severity_score, 20))

    # --- 3. Duration Scaling (Time) ---
    stored_duration = state.get("duration_context", "") or ""
    import re
    duration_match = re.search(r"(\d+)\s*(days?|weeks?|months?|hours?|years?)", user_msg_lower)
    
    current_duration = stored_duration 
    if duration_match:
        current_duration = duration_match.group(0)
    elif "yesterday" in user_msg_lower:
        current_duration = "1 day"

    duration_score = 0
    days = 0
    
    if current_duration:
        if "chronic" in current_duration or "long time" in current_duration:
            duration_score += 15
        elif "hour" in current_duration or "minute" in current_duration:
            duration_score += 10 
        else:
            d_match = re.search(r"(\d+)\s*days?", current_duration)
            w_match = re.search(r"(\d+)\s*weeks?", current_duration)
            m_match = re.search(r"(\d+)\s*months?", current_duration)
            
            if d_match: days = int(d_match.group(1))
            if w_match: days = int(w_match.group(1)) * 7
            if m_match: days = int(m_match.group(1)) * 30
            
            if days <= 3:
                duration_score += days * 3
            elif days <= 14:
                duration_score += (3*3) + ((days-3) * 1.5)
            else:
                duration_score += (3*3) + (11*1.5) + ((days-14) * 0.5)
                
    duration_score = min(duration_score, 35) 

    # --- 4. Red Flag / Critical Checks (Safety Net) ---
    critical_score = 0
    
    red_flags = [
        "chest pain", "breathing", "unconscious", "faint", "bleeding", "hemorrhage", "heart attack", "stroke", "suicide"
    ]
    if any(rf in combined_text for rf in red_flags):
        critical_score = 50
    
    if predictions:
        top_disease = predictions[0]
        if top_disease["confidence"] > 0.4:
            critical_score += int(top_disease["confidence"] * 10)
            CRIT_LIST = ["sepsis", "meningitis", "pneumonia", "appendicitis", "myocardial"]
            if any(c in top_disease["disease"].lower() for c in CRIT_LIST):
                critical_score += 30

    # --- 5. Demographics (Risk Factors) ---
    demographic_score = 0
    age = patient_info.get("age")
    if age:
        if age < 5: demographic_score += 15
        elif age < 10: demographic_score += 10
        elif age > 75: demographic_score += 20
        elif age > 60: demographic_score += 10
        
    if patient_info.get("is_pregnant"): demographic_score += 20
    if patient_info.get("history"): demographic_score += len(patient_info["history"]) * 5

    # --- FINAL CALCULATION ---
    final_score = symptom_load + high_risk_score + severity_score + duration_score + demographic_score + critical_score
    
    final_score = max(0, min(int(final_score), 100))
    
    reasoning = f"Sx({len(symptoms)}):{symptom_load} RiskBonus:{high_risk_score} Mod:{severity_score} Time:{duration_score} Demog:{demographic_score} Crit:{critical_score}"
    
    logger.info(f"""
    Urgency Breakdown:
    {reasoning}
    => TOTAL: {final_score}
    """)
    
    return {
        "urgency_score": final_score,
        "duration_context": current_duration,
        "urgency_reasoning": reasoning
    }

# --- 4. Symptom Completion (Reasoning) ---
async def symptom_completion(state: AgentState) -> AgentState:
    """
    Decides NEXT STEP: Ask Info, Ask Symptoms, Booking, or Advice.
    Prioritizes getting MVP patient info (Age, Gender, Comorbidities).
    """
    urgency = state.get("urgency_score", 0)
    predictions = state.get("diagnosis_probabilities", [])
    current_symptoms = list(state.get("symptoms") or [])
    asked_questions = list(state.get("asked_questions") or [])
    patient_info = state.get("patient_info") or {}
    
    logger.info(f"[Decision] Info: {patient_info}")
    
    # CONSTANTS
    MAX_QUESTIONS = 6 # Slight increase for flow
    
    # Rule 1: High Urgency -> Escalate Immediately (skip details if critically urgent)
    if urgency >= 85:
        state["next_step"] = "escalate"
        return state
        
    # Rule 2: Collect Basic Demographics (Age)
    # If we don't have age, ask for it.
    if patient_info.get("age") is None:
        # Check if we already asked
        if "ask_demographics" not in asked_questions:
            logger.info("Missing Age/Gender. Asking demographics.")
            asked_questions.append("ask_demographics")
            return {
                "next_step": "ask_demographics",
                "missing_information": "age_gender",
                "asked_questions": asked_questions
            }
            
    # Rule 3: Collect Medical History for Mid-Aged/Elderly
    # If Age > 40 and we haven't checked history/sugar etc.
    age = patient_info.get("age", 0)
    if age > 40 and "history" not in patient_info:
        # We assume empty history = unknown if we haven't asked explicitly
        # To avoid asking repeatedly, check if we asked "ask_history"
        if "ask_history" not in asked_questions:
            logger.info("Patient > 40. Asking medical history.")
            asked_questions.append("ask_history")
            return {
                "next_step": "ask_history",
                "missing_information": "comorbidities",
                "asked_questions": asked_questions
            }

    # Rule 4: Normal Symptom Triage Flow
    if not predictions:
        state["next_step"] = "ask" 
        state["missing_information"] = "main symptoms"
        return state

    questions_count = len([q for q in asked_questions if not q.startswith("ask_")])
    
    # Force Stop
    if questions_count >= 5: # Limit symptom questions
        decision = "booking" if urgency > 40 else "advice"
        return {"next_step": decision}
        
    # Confidence Stop
    top_disease = predictions[0]
    if top_disease["confidence"] > 0.90 and questions_count >= 2:
        decision = "booking" if urgency > 40 else "advice"
        return {"next_step": decision}
        
    # Select Next Symptom Question
    classifier = get_biobert_classifier()
    expected_symptoms = classifier.get_symptoms_for_disease(top_disease["disease"])
    
    def normalize(s: str) -> str: return s.replace("_", " ").lower().strip()
    asked_normalized = {normalize(q) for q in asked_questions}
    known_normalized = {normalize(s) for s in current_symptoms}
    phys_system = state.get("physiological_system", "General")
    
    candidate_symptoms = []
    for expected in expected_symptoms:
        clean_expected = normalize(expected)
        if clean_expected not in asked_normalized and clean_expected not in known_normalized:
            relevance = 1.0
            sym_sys = identify_system(clean_expected)
            if phys_system != "General" and sym_sys == phys_system: relevance = 2.0 
            candidate_symptoms.append((clean_expected, relevance))
            
    candidate_symptoms.sort(key=lambda x: x[1], reverse=True)
    target = candidate_symptoms[0][0] if candidate_symptoms else None
             
    if target:
        asked_questions.append(target)
        return {
            "next_step": "ask",
            "missing_information": target,
            "asked_questions": asked_questions
        }
    else:
        decision = "booking" if urgency > 40 else "advice"
        return {"next_step": decision}

# --- 5. Response Generator (LLM) ---
async def response_generator(state: AgentState) -> AgentState:
    """
    Generates final response. Handles new demographic flows.
    """
    step = state.get("next_step")
    urgency = state.get("urgency_score", 0)
    llm = HybridLLM()
    response = ""
    
    if step == "escalate":
        response = "URGENT: Your symptoms indicate a possible medical emergency. Please visit the nearest hospital or contact emergency services immediately."
        if urgency > 90:
             response += " Do not delay."
    
    elif step == "ask_demographics":
        response = "Before we proceed, could you please share your age? This helps me assess the situation better (e.g., if you are under 10 or over 60)."
        
    elif step == "ask_history":
        response = "Given your age, do you have any existing medical conditions I should know about, such as diabetes (sugar), high blood pressure, or heart conditions?"
        
    elif step == "booking":
        response = "Based on your symptoms and details, it is advisable to consult a doctor. Would you like assistance in booking an appointment nearby?"
        
    elif step == "advice":
        if urgency > 30:
            response = "You might consider visiting a pharmacy for over-the-counter relief. However, if symptoms persist for more than 24 hours, please see a doctor."
        else:
            response = "It sounds like general discomfort. Ensure you get plenty of rest and stay hydrated. If you don't feel better soon, consult a physician."
        
    elif step == "ask":
        target = state.get("missing_information", "more details")
        prompt = f"""
        You are a Medical Triage Agent.
        Task: Ask ONE simple question to check for symptom: '{target}'.
        
        Context: 
        - Patient Symptoms: {state.get('symptoms', [])}
        
        Constraints:
        - Do NOT diagnose.
        - Concise (max 15 words).
        - Professional but empathetic.
        """
        response = await llm.complete(prompt)
        response = response.strip('"')

    response = response.strip('"').strip()
    if not response:
        response = "Could you tell me a bit more about your symptoms?"

    return {
        "final_response": response,
        "agent_response": response
    }
