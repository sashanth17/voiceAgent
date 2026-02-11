import json
import re
from typing import Dict, Any, List
from app.agent.state import AgentState
from app.agent.llms.biobert import get_biobert_classifier
from app.agent.llms.hybrid_llm import HybridLLM
from app.agent.logger import logger

from app.agent.llms.symptom_extractor import get_symptom_extractor
from app.agent.utils.system_mapping import identify_system, SYSTEM_KEYWORDS

# --- 0. Helper: Extract Demographics (Age/Gender/Child Context) ---
def extract_demographics(text: str, current_info: dict) -> dict:
    """
    Extracts age, gender, child references, and duration from text.
    Returns updated info dict with: 
    - age (int)
    - gender (str)
    - is_child_reference (bool)
    - duration_days (int) - NEW
    - duration_text (str)
    """
    info = current_info.copy()
    text_lower = text.lower()
    
    # ✅ Enhanced: Detect "my child" / "my kid" / "my son" / "my daughter"
    child_keywords = ["my child", "my kid", "my son", "my daughter", "my baby", "my toddler"]
    if any(ck in text_lower for ck in child_keywords):
        info["is_child_reference"] = True
    
    # Age extraction improved to catch "child is X" or "X year old"
    age_match = re.search(r'\b(\d{1,2})\s*year', text_lower) 
    if age_match:
        try:
            val = int(age_match.group(1))
            if 0 <= val <= 120:
                info["age"] = val
        except:
            pass
            
    # Also check simpler "I am 25" pattern if the above failed
    if "age" not in info:
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?|yrs?|yo)\b', text_lower)
        if age_match:
            try:
                val = int(age_match.group(1))
                if 0 <= val <= 120:
                    info["age"] = val
            except:
                pass

    # Gender extraction from context
    if "son" in text_lower or "he" in text_lower.split():
        info["gender"] = "male"
    elif "daughter" in text_lower or "she" in text_lower.split():
        info["gender"] = "female"
    elif any(w in text_lower for w in ["female", "woman", "girl", "lady"]):
        info["gender"] = "female"
    elif any(w in text_lower for w in ["male", "man", "boy", "gentleman"]):
        info["gender"] = "male"

    # ✅ Restored: Pregnancy
    if any(w in text_lower for w in ["pregnant", "pregnancy", "expecting baby"]):
        info["is_pregnant"] = True

    # ✅ Restored: Medical History (Simple Keywords)
    history = []
    conditions = {
        "diabetes": ["diabetes", "sugar", "diabetic"],
        "hypertension": ["bp", "blood pressure", "hypertension", "high bp"],
        "asthma": ["asthma", "wheezing", "breathing issue"],
        "heart_condition": ["heart disease", "cardiac", "heart attack"]
    }
    for cond, keywords in conditions.items():
        if any(k in text_lower for k in keywords):
            history.append(cond)
            
    if history:
        # Append to existing history
        existing = info.get("history", [])
        info["history"] = list(set(existing + history))

    # ✅ Enhanced Duration Extraction -> duration_days
    # Match patterns: "3 days", "for 3 days", "since 3 days"
    duration_match = re.search(r"(?:for|since)?\s*(\d+)\s*(days?|weeks?|months?|hours?|years?)", text_lower)
    if duration_match:
        try:
            num = int(duration_match.group(1))
            unit = duration_match.group(2)
            
            # Calculate days
            days = 0
            if "day" in unit: days = num
            elif "week" in unit: days = num * 7
            elif "month" in unit: days = num * 30
            elif "year" in unit: days = num * 365
            elif "hour" in unit: days = 1 # Treat as 1 day for logic
            
            info["duration_days"] = days
            # preserved original text
            info["duration_text"] = duration_match.group(0).replace("for ", "").replace("since ", "").strip()
        except:
            pass
        
    elif "yesterday" in text_lower:
        info["duration_days"] = 1
        info["duration_text"] = "1 day"
    elif "today" in text_lower or "now" in text_lower:
        info["duration_days"] = 0 # < 1 day
        info["duration_text"] = "1 day"

    return info

# --- 1. Symptom Extraction Agent ---
# --- 1. Symptom Extraction Agent ---
async def symptom_extractor(state: AgentState) -> AgentState:
    """
    Extracts symptoms and manages critical state (demographics, child mode).
    Primary logic: NER -> Fast Path (Yes/No) -> LLM Fallback.
    """
    input_text = state.get("query") or state.get("user_message", "")
    existing_symptoms = list(state.get("symptoms") or [])
    last_target = state.get("missing_information")
    
    # --- Demographics Update ---
    current_info = state.get("patient_info") or {}
    # FIX: Pass current_info to merge properly
    new_info = extract_demographics(input_text, current_info)
    state["patient_info"] = new_info
    
    # Log detected info
    logger.info(f"Patient Info Updated: {new_info}")
    
    # --- 🔵 CHILD MODE LOGIC ---
    # Activate Child Mode if age <= 10
    age = new_info.get("age")
    if age is not None and age <= 10:
        state["is_child"] = True
        state["child_mode_active"] = True
    elif new_info.get("is_child_reference"):
        state["is_child"] = True
        # Don't activate full mode without age, but flag it
    
    # --- Intent Classification (Optimized) ---
    llm = HybridLLM()
    extracted_intent = "MEDICAL" # Default
    
    # Only run intent check if input is short or ambiguous to save latency
    # or if we found demographics but no obvious symptoms yet
    if len(input_text.split()) < 20: 
        intent_prompt = f"""
        Analyze the input text for medical intent.
        Input: "{input_text}"
        
        Categories:
        - GREETING: "hi", "hello", "good morning", "hey"
        - GENERAL: "what is fever?", "is covid allowed?", "who are you?", "tell me about yourself"
        - MEDICAL: "I have fever", "pain in leg", "yes", "no", "headache", "my child has fever"
        - DEMOGRAPHICS: "I am 25", "female", "my age is 40"
        - ACKNOWLEDGMENT: "ok", "okay", "thanks", "thank you", "alright", "got it", "sure"
        - PRANK: nonsense or irrelevant (e.g. "I am batman", "sing a song")
        
        If it contains symptoms or medical conditions or specific "yes/no" to a symptom question, choose MEDICAL.
        If it's a simple acknowledgment after receiving advice, choose ACKNOWLEDGMENT.
        
        Return JSON STRICTLY: {{ "intent": "Category" }}
        """
        try:
            intent_res = await llm.complete(intent_prompt)
            # clean json
            start = intent_res.find('{')
            end = intent_res.rfind('}')
            if start != -1:
                idata = json.loads(intent_res[start:end+1])
                extracted_intent = idata.get("intent", "MEDICAL")
                logger.info(f"Detected Intent: {extracted_intent}")
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")

    # Store intent in payload for downstream agents
    current_payload = state.get("payload") or {}
    current_payload["intent"] = extracted_intent
    state["payload"] = current_payload

    new_symptoms = []
    
    # Handling Intents
    if extracted_intent == "GREETING":
        return {"next_step": "greet"}
    elif extracted_intent == "GENERAL":
        return {"next_step": "answer_general"}
    elif extracted_intent == "PRANK":
        return {"next_step": "handle_prank"}
    elif extracted_intent == "ACKNOWLEDGMENT":
         # Logic handled in symptom_completion
         pass

    # --- PHASE 1: Try Specialized Main Model (NER) ---
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

    # --- PHASE 1.5: Fuzzy Keyword Matching ---
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

    # --- PHASE 2: Fallback Logic ---
    # Optimized: Split Fast Path for Child vs Adult
    
    is_answering_question = last_target is not None and not last_target.startswith("ask_") and len(input_text.split()) < 5
    llm_needed = True
    
    if state.get("child_mode_active"):
        # 👶 CHILD FAST PATH
        # Child mode questions are usually status checks (eating, playing).
        # Only skip extraction if the answer is SHORT (simple status update).
        if len(input_text.split()) < 5:
            llm_needed = False
            logger.info("Child Mode (Short): Skipping LLM for status update.")
        else:
            # Long input -> might include new symptoms ("He stopped eating and has rash")
            logger.info("Child Mode (Long): Checking complex input.")
            llm_needed = True
        
    elif is_answering_question:
        # 🧑 ADULT FAST PATH
        # Adult questions are usually "Do you have X?". Only add if YES.
        text_lower = input_text.lower().strip()
        affirmative = ["yes", "yeah", "yep", "sure", "correct", "right"]
        negative = ["no", "nope", "nah", "not really", "incorrect"]
        
        # Check for simple affirmative
        if any(aff == text_lower or text_lower.startswith(aff + " ") for aff in affirmative):
            logger.info(f"Fast Path (Adult): User confirmed '{last_target}'")
            new_symptoms.append(last_target)
            llm_needed = False
            
        # Check for simple negative
        elif any(neg == text_lower or text_lower.startswith(neg + " ") for neg in negative):
            logger.info(f"Fast Path (Adult): User denied '{last_target}'")
            llm_needed = False
            
    if llm_needed and (not new_symptoms or is_answering_question):
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
    return {"symptoms": final_list, "patient_info": current_info, "payload": current_payload}

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
    Calculates urgency score (0-100) using STRICT RURAL LOGIC.
    Prioritizes Danger Signs. Handles Child Rules.
    """
    raw_symptoms = state.get("symptoms", [])
    patient_info = state.get("patient_info", {})
    user_msg_lower = state.get("user_message", "").lower()
    
    # --- 1. Duration Extraction ---
    stored_duration = state.get("duration_context", "") or ""
    import re
    # Match patterns: "3 days", "for 3 days", "since 3 days"
    duration_match = re.search(r"(?:for|since)?\s*(\d+)\s*(days?|weeks?|months?|hours?|years?)", user_msg_lower)
    
    current_duration = stored_duration 
    if duration_match:
        # Extract the full match (e.g., "3 days" or "for 3 days")
        current_duration = duration_match.group(0).replace("for ", "").replace("since ", "").strip()
    elif "yesterday" in user_msg_lower:
        current_duration = "1 day"
    elif "today" in user_msg_lower or "now" in user_msg_lower:
        current_duration = "1 day"

    # Convert to days for logic
    days_cnt = 0
    if current_duration:
        d_match = re.search(r"(\d+)\s*days?", current_duration)
        w_match = re.search(r"(\d+)\s*weeks?", current_duration)
        m_match = re.search(r"(\d+)\s*months?", current_duration)
        y_match = re.search(r"(\d+)\s*years?", current_duration)
        
        if d_match: days_cnt = int(d_match.group(1))
        if w_match: days_cnt = int(w_match.group(1)) * 7
        if m_match: days_cnt = int(m_match.group(1)) * 30
        if y_match: days_cnt = int(y_match.group(1)) * 365
        
        # Hours/Since morning = < 1 day (0 days logic)
        if "hour" in current_duration: days_cnt = 0.5 

    # --- 2. Danger Signs (Adult & Child) ---
    # Difficulty breathing, Not eating/drinking (Child), Continuous vomiting, Very weak, Worsening, Serious conditions
    danger_signs = [
        "breathing", "breath", "unconscious", "faint", "not eating", "not drinking", 
        "vomit", "throwing up", "weak", "not responding", "worse", "worsening",
        "diabetes", "heart", "pregnant", "pregnancy", "chest pain", "seizure", "fit", "fits"
    ]
    combined_text = (user_msg_lower + " " + " ".join([str(s).lower() for s in raw_symptoms])).lower()
    
    is_danger = False
    for sign in danger_signs:
        if sign in combined_text:
            is_danger = True
            break
            
    # --- 3. Child Specific Rules (Age <= 10) ---
    age = patient_info.get("age")
    is_child = age is not None and age <= 10
    
    # Defaults
    final_score = 0
    reasoning = ""

    if is_child:
        # Child Danger Signs (Immediate High)
        child_danger_signs = ["not drinking", "not eating", "vomiting", "weak", "seizure", "fit"]
        if is_danger or any(sign in combined_text for sign in child_danger_signs):
             final_score = 90
             reasoning = "Child High Urgency: Danger signs detected."
        else:
             # Child Mild Rule
             mild_symptoms = ["fever", "cold", "cough", "sneeze", "runny nose"]
             is_mild = any(s in combined_text for s in mild_symptoms)
             
             # 🔹 CONTEXT-AWARE URGENCY: Check child interactive responses
             # Boost urgency if child is NOT eating/drinking/playing
             negative_responses = ["no", "not", "isn't", "cant", "cannot", "won't", "doesn't"]
             has_negative = any(nr in user_msg_lower for nr in negative_responses)
             
             # Check if child is active/eating (positive indicators)
             positive_indicators = ["yes", "eating", "drinking", "playing", "active", "normal", "fine", "ok", "better"]
             has_positive = any(pi in user_msg_lower for pi in positive_indicators)
             
             # Check if worsening mentioned
             is_worsening = any(w in user_msg_lower for w in ["worse", "worsening", "getting bad"])
             
             if is_mild and days_cnt <= 2:
                  # Check contextual answers
                  if has_negative and not has_positive:
                      # Child not eating/drinking/playing despite mild symptoms → increase urgency
                      final_score = 60
                      reasoning = "Child Low Duration BUT concerning activity/eating status."
                  else:
                      # Force LOW Urgency (normal case)
                      final_score = 20
                      reasoning = "Child Low Urgency: Mild symptoms ≤ 2 days."
             elif days_cnt > 2:
                  # Child Moderate Logic 
                  if is_worsening or (has_negative and "eating" in user_msg_lower):
                      final_score = 75
                      reasoning = "Child Moderate-High: Worsening or not eating after > 2 days."
                  elif has_positive:
                      final_score = 35
                      reasoning = "Child Moderate: Duration > 2 days but showing positive signs."
                  else:
                      final_score = 45 
                      reasoning = "Child Moderate: Duration > 2 days. Monitor eating/activity."
             else:
                  # Fallback for other child symptoms
                   final_score = 35
                   reasoning = "Child General: Standard observation."
        
        # 🔹 AGE-BASED BOOST (New Request)
        if age <= 5:
            final_score += 15
            reasoning += " (+15 Age <= 5)"
        elif age <= 10:
            final_score += 10
            reasoning += " (+10 Age <= 10)"

        final_score = min(final_score, 100)
    else:
        # --- Adult Logic ---
        if is_danger:
            final_score = 90
            reasoning = "Adult Danger Signs Detected."
        else:
            # Standard Logic
            symptom_load = min(len(raw_symptoms) * 10, 40)
            
            # Duration Score
            dur_score = 0
            if days_cnt > 2: dur_score += 10
            if days_cnt > 7: dur_score += 20
            
            # Severity from text
            sev_score = 0
            if "severe" in user_msg_lower or "pain" in user_msg_lower or "high" in user_msg_lower:
                sev_score = 15
                
            final_score = symptom_load + dur_score + sev_score
            final_score = min(final_score, 80) # Cap if no danger signs

    # === Conditional Risk Boost Layer (User Request 449) ===
    # Applied AFTER base score calculation to refine risk
    
    # 1. Child Fever + Duration > 2 days
    is_fever = any("fever" in s.lower() for s in raw_symptoms)
    if is_child and is_fever and days_cnt > 2:
        final_score += 20
        reasoning += " (+20 Child Fever > 2 days)"
        
    # 2. Elderly (>= 60) + Dizziness/Weakness
    is_elderly = age is not None and age >= 60
    has_dizzy_weak = any(s in combined_text for s in ["dizzy", "dizziness", "weakness", "faint", "lightheaded", "weak"])
    if is_elderly and has_dizzy_weak:
        final_score += 25
        reasoning += " (+25 Elderly Dizziness/Weakness)"
        
    # 3. Comorbidity (Diabetes, Heart Disease, BP)
    history = patient_info.get("history", [])
    has_comorbidity = any(c in ["diabetes", "hypertension", "heart_condition", "heart disease", "bp", "sugar", "high bp"] for c in history)
    if has_comorbidity:
        final_score += 15
        reasoning += " (+15 Comorbidity Risk)"
        
    # 4. Duration > 7 days (Chronic Flag)
    if days_cnt > 7:
        final_score += 20
        reasoning += " (+20 Chronic Duration > 7 days)"
        
    # 5. Severe Description
    severe_keywords = ["severe", "unbearable", "very high", "extreme", "worst"]
    if any(k in user_msg_lower for k in severe_keywords):
        final_score += 20
        reasoning += " (+20 Reported Severe Pain)"
        
    # Final Cap
    final_score = min(final_score, 100)

    logger.info(f"""
    Urgency Breakdown:
    {reasoning}
    => TOTAL: {final_score}
    """)
    
    return {
        "urgency_score": int(final_score),
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
    
    # Check Intent from Symptom Extractor
    input_intent = state.get("payload", {}).get("intent")
    
    if input_intent == "GREETING":
        return {"next_step": "greet"}
    elif input_intent == "GENERAL":
        return {"next_step": "answer_general"}
    elif input_intent == "PRANK":
        return {"next_step": "handle_prank"}
    elif input_intent == "DEMOGRAPHICS":
        return {"next_step": "ask_symptoms_convo"}
    elif input_intent == "ACKNOWLEDGMENT":
        # User said "ok" or "thanks" after receiving advice
        # Check if we already gave advice/booking
        last_step = state.get("next_step", "")
        if last_step in ["advice", "booking", "escalate"]:
            return {"next_step": "end_conversation"}
        # Otherwise, treat as neutral (continue flow)

    logger.info(f"[Decision] Info: {patient_info}")
    
    # Rule 0: No Symptoms? ask them FIRST (Before Age/Diagnosis)
    if not current_symptoms:
        # If we haven't asked for symptoms explicitly yet (prevents loops if extraction fails repeatedly)
        # But here we just want to ensure we don't jump to diagnosis.
        return {"next_step": "ask_symptoms_convo"}

    # CONSTANTS
    MAX_QUESTIONS = 6 # Slight increase for flow
    
    # Rule 1: High Urgency -> Escalate Immediately (skip details if critically urgent)
    if urgency >= 70:
        state["next_step"] = "escalate"
        return state
        
    # Rule 2: Collect Basic Demographics (Age)
    # Update: Prioritize Age extraction based on duration risk
    dur_days = patient_info.get("duration_days")
    
    # If Duration > 2 days (Risk Factor) AND Age unknown -> FORCE ask
    if dur_days and dur_days > 2 and patient_info.get("age") is None:
        if "ask_demographics" not in asked_questions:
            logger.info("Duration > 2 days + Missing Age -> Prioritizing Age Question")
            asked_questions.append("ask_demographics")
            return {
                "next_step": "ask_demographics",
                "missing_information": "age_gender",
                "asked_questions": asked_questions
            }
            

    
    # ✅ CHILD_INTERACTIVE_MODE: Contextual Follow-Up
    age = patient_info.get("age", 0)
    # Use flag from extractor ("my son") OR explicit age
    is_active_child = state.get("is_child") or (age is not None and 0 < age <= 10)
    
    duration_context = state.get("duration_context", "")
    
    if is_active_child and current_symptoms:
        logger.info("🔹 CHILD_INTERACTIVE_MODE activated")
        
        # Step 1: Duration Check (If not known, ask for it)
        if not duration_context and "ask_child_duration" not in asked_questions:
            logger.info("Child: Asking duration")
            asked_questions.append("ask_child_duration")
            return {
                "next_step": "ask_child_duration",
                "missing_information": "child_duration",
                "asked_questions": asked_questions
            }

        # Parse duration days locally if needed (already in patient_info usually)
        days_cnt = dur_days if dur_days is not None else 0
        
        # Check for danger signs in current conversation
        user_msg_lower = state.get("user_message", "").lower()
        danger_keywords = ["not drinking", "not eating", "vomiting", "weak", "breathing", "breath", "seizure", "worse"]
        has_danger_sign = any(kw in user_msg_lower for kw in danger_keywords)
        
        if has_danger_sign:
            logger.info("Child: Danger sign detected in conversation - escalating")
            return {
                "next_step": "escalate",
                "child_mode_active": False,
                "child_stage": None
            }
        
        # Step 2: Duration ≤ 2 days → Ask status questions
        if days_cnt <= 2:
            # Ask eating status
            if "ask_child_eating" not in asked_questions:
                logger.info("Child ≤ 2 days: Asking eating status")
                asked_questions.append("ask_child_eating")
                return {
                    "next_step": "ask_child_eating",
                    "missing_information": "child_eating",
                    "asked_questions": asked_questions
                }
            
            # Ask activity/playing status
            if "ask_child_playing" not in asked_questions:
                logger.info("Child ≤ 2 days: Asking activity status")
                asked_questions.append("ask_child_playing")
                return {
                    "next_step": "ask_child_playing",
                    "missing_information": "child_playing",
                    "asked_questions": asked_questions
                }
            
            # Ask breathing status
            if "ask_child_breathing" not in asked_questions:
                logger.info("Child ≤ 2 days: Asking breathing status")
                asked_questions.append("ask_child_breathing")
                return {
                    "next_step": "ask_child_breathing",
                    "missing_information": "child_breathing",
                    "asked_questions": asked_questions
                }
            
            # All questions asked for mild case → provide advice
            logger.info("Child ≤ 2 days: All status questions answered → advice")
            return {"next_step": "advice"}
        
        # Step 3: Duration > 2 days → Ask improvement/worsening
        else:
            # Ask if improving or getting worse
            if "ask_child_improvement" not in asked_questions:
                logger.info("Child > 2 days: Asking improvement status")
                asked_questions.append("ask_child_improvement")
                return {
                    "next_step": "ask_child_improvement",
                    "missing_information": "child_improvement",
                    "asked_questions": asked_questions
                }
            
            # Ask if very weak or sleepy
            if "ask_child_weakness" not in asked_questions:
                logger.info("Child > 2 days: Asking weakness/sleepiness")
                asked_questions.append("ask_child_weakness")
                return {
                    "next_step": "ask_child_weakness",
                    "missing_information": "child_weakness",
                    "asked_questions": asked_questions
                }
            
            # All questions asked for moderate case → provide guidance
            logger.info("Child > 2 days: All follow-ups answered → booking/advice")
            decision = "booking" if urgency > 40 else "advice"
            return {
                "next_step": decision,
                "child_mode_active": False,
                "child_stage": None
            }
            
    # Rule 3: Collect Medical History for Mid-Aged/Elderly
    # If Age > 40 and we haven't checked history/sugar etc.
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

    # Rule 4: Normal Symptom Triage Flow (Only if we have symptoms)
    if not predictions:
        # This shouldn't happen if we have symptoms, but as fallback
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
    
    # 🔒 RURAL-SAFE QUESTION POOL (User Request)
    # Block intrusive/rare/diagnostic questions.
    inappropriate_keywords = [
        "sexual", "intimate", "intercourse", "partner", "relationship",
        "drug", "injection", "needle", "substance",
        "hiv", "aids", "std", "sti", "genital",
        "abortion", "miscarriage", "pregnancy test",
        "travel", "foreign", "contact with", "history of", 
        "rare", "unusual", "syndrome", "mass", "lump"
    ]
    
    # Conditional Block: Weight Loss (Unless > 7 days)
    dur = patient_info.get("duration_days", 0)
    if not dur or dur <= 7:
        inappropriate_keywords.extend(["weight", "appetite"])
    
    # Common symptom patterns for mild conditions
    common_symptoms_patterns = [
        "fever", "cold", "cough", "sneeze", "headache", "fatigue", "body ache",
        "runny nose", "sore throat", "congestion", "chills", "weakness"
    ]
    is_likely_common_condition = any(
        pattern in " ".join(current_symptoms).lower() 
        for pattern in common_symptoms_patterns
    )
    
    candidate_symptoms = []
    for expected in expected_symptoms:
        clean_expected = normalize(expected)
        
        # Skip if already asked or known
        if clean_expected in asked_normalized or clean_expected in known_normalized:
            continue
        
        # 🔒 Filter out inappropriate questions
        is_inappropriate = any(kw in clean_expected for kw in inappropriate_keywords)
        if is_inappropriate:
            logger.info(f"⚠️ Skipping inappropriate question: {clean_expected}")
            continue
        
        # Filter out irrelevant questions for common conditions
        if is_likely_common_condition:
            # For common conditions, only ask about respiratory/general symptoms
            irrelevant_for_common = [
                "weight", "appetite loss", "night sweats", "lymph", "rash",
                "patches", "spots", "lesion", "ulcer"
            ]
            if any(irr in clean_expected for irr in irrelevant_for_common):
                logger.info(f"⚠️ Skipping irrelevant question for common condition: {clean_expected}")
                continue
        
        # Calculate relevance
        relevance = 1.0
        sym_sys = identify_system(clean_expected)
        if phys_system != "General" and sym_sys == phys_system:
            relevance = 2.0
        
        # Boost very common follow-up symptoms
        common_followups = ["duration", "severity", "temperature", "breathing", "eating", "drinking"]
        if any(cf in clean_expected for cf in common_followups):
            relevance += 0.5
        
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
        return {
            "next_step": decision,
            "child_mode_active": False,
            "child_stage": None
        }

# --- 5. Response Generator (LLM) ---
async def response_generator(state: AgentState) -> AgentState:
    """
    Generates final response using structured, safe medical triage logic.
    Strictly avoids diagnosis. Uses tiered urgency levels.
    """
    step = state.get("next_step")
    urgency = state.get("urgency_score", 0)
    current_symptoms = state.get("symptoms", [])
    patient_info = state.get("patient_info", {})
    llm = HybridLLM()
    response = ""
    
    # Conversational Intents
    if step == "greet":
        response = "Hi! I am your AI Medical Assistant. I can help you check your symptoms and guide you to the right care. How are you feeling today?"
        
    elif step == "answer_general":
        query = state.get("query") or state.get("user_message", "")
        prompt = f"""
        You are a helpful Medical Assistant.
        The user asked: "{query}"
        Answer the question accurately, briefly, and professionally.
        After answering, politely ask if they have any personal symptoms related to this.
        Do not diagnose.
        """
        response = await llm.complete(prompt)
        
    elif step == "handle_prank":
        response = "I am a medical health assistant designed to help with health concerns. Please tell me about your symptoms so I can assist you."
        
    elif step == "ask_symptoms_convo":
        response = "Could you please describe what symptoms you are experiencing?"
        age = state.get("patient_info", {}).get("age")
        if age:
            response = f"Thank you. Since you are {age} years old, could you tell me more about your symptoms?"
    
    elif step == "end_conversation":
        response = "Take care! Feel free to reach out if you have any other health concerns. Stay safe!"

    # Medical Questioning
    elif step == "ask_demographics":
        response = "Before we proceed, could you please share your age? This helps me assess the situation better (e.g., if you are under 10 or over 60)."
        
    elif step == "ask_history":
        response = "Given your age, do you have any existing medical conditions I should know about, such as diabetes, high blood pressure, or heart conditions?"

    # 🔹 CHILD_INTERACTIVE_MODE Responses
    elif step == "ask_child_duration":
        response = "Since when has the child had these symptoms? For example, since this morning, yesterday, or a few days ago?"
    
    elif step == "ask_child_eating":
        response = "Is the child eating and drinking normally?"
    
    elif step == "ask_child_playing":
        response = "Is the child playing or active as usual?"
    
    elif step == "ask_child_breathing":
        response = "Is the child breathing comfortably, or is there any difficulty in breathing?"
    
    elif step == "ask_child_improvement":
        response = "Over the past days, is the child getting better or getting worse?"
    
    elif step == "ask_child_weakness":
        response = "Is the child very weak or unusually sleepy?"

    elif step == "ask":
        target = state.get("missing_information", "more details")
        prompt = f"""
        🔒 ROLE CONSTRAINT
        You are a Rural Medical Triage Assistant.
        You are NOT a hospital doctor.
        You do NOT perform diagnosis.
        You do NOT explore unrelated medical history.
        You only assess urgency and guide safely.
        
        🔒 YOUR TASK
        Ask ONE simple question to check for symptom: '{target}'.
        
        Context: 
        - Patient Symptoms So Far: {state.get('symptoms', [])}
        - Patient Age: {state.get('patient_info', {}).get('age', 'unknown')}
        
        🔒 QUESTION RESTRICTIONS (STRICT)
        You may ONLY ask questions from these categories:
        
        ✅ ALLOWED:
        - Duration: "Since when?", "How many days?"
        - Severity (simple): "Is body very hot?", "Is pain mild or strong?"
        - Danger signs: breathing difficulty, not eating/drinking, vomiting, weakness, worsening
        - Activity: "Can you eat/drink?", "Can you walk?"
        
        ❌ STRICTLY FORBIDDEN - Never ask about:
        - Sexual history or intimate contact
        - Weight loss (unless severe long-term case)
        - Rare disease screening questions
        - Lab test details
        - Detailed medical history unrelated to current urgency
        - Numerical temperature values
        - Patches, spots, lesions, rashes (unless primary complaint)
        
        🔒 RESPONSE STYLE RULE
        - Use simple rural-friendly English
        - No medical terminology
        - No disease naming
        - Max 12 words
        - Conversational and empathetic
        
        If '{target}' is inappropriate or irrelevant for common fever/cold:
        → Skip it and ask about duration or severity instead.
        
        Generate ONE simple question now:
        """
        response = await llm.complete(prompt)
        response = response.strip('"')
    
    # Escalation
    elif step == "escalate":
        age = patient_info.get("age")
        is_child = age is not None and age <= 10
        if is_child:
            response = "This condition is serious for a small child. Please take the child to the nearest hospital immediately. Do not wait."
        else:
            response = "Based on your symptoms, I strongly recommend seeking immediate medical attention. Please visit the nearest hospital or call emergency services."

    # Final Advice (Rural Logic)
    # Final Advice (Rural Logic) - Unified Thresholds
    elif step in ["advice", "booking"]:
        # Standardize Thresholds:
        # < 40: Home Care
        # 40-70: Clinic
        # >= 70: Hospital
        
        age = patient_info.get("age")
        is_child = state.get("is_child") or (age is not None and age <= 10)
        
        if is_child:
            # Child Phrasing (Parent-Centered)
            if urgency >= 70:
                response = "This condition looks serious for a child. Please take the child to the nearest hospital immediately. Do not wait."
            elif urgency >= 40:
                response = "It is better to show the child to a nearby doctor soon for a checkup to be safe."
            else:
                response = "The child seems stable. Keep the child hydrated, give light food, and rest. Watch closely for changes."
        else:
            # Adult Phrasing
            if urgency >= 70:
                response = "This condition is serious. Please go to the nearest hospital immediately. Do not delay."
            elif urgency >= 40:
                response = "It is better to visit a nearby doctor or clinic for a checkup. Would you like help finding one?"
            else:
                # Neutral phrasing, no diagnosis
                response = "This condition may be mild and temporary. Rest well and drink plenty of water. If it gets worse, please see a doctor."


    response = response.strip('"').strip()
    if not response:
        response = "Could you tell me a bit more about your symptoms?"

    return {
        "final_response": response,
        "agent_response": response
    }
