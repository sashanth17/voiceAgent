
# Maps symptoms/keywords to physiological systems for anchoring reasoning.

SYSTEM_KEYWORDS = {
    "Cardiovascular": ["chest", "heart", "palpitation", "pulse", "angina", "pressure"],
    "Respiratory": ["breath", "lung", "cough", "wheeze", "sneeze", "cold", "flu", "choke"],
    "Gastrointestinal": ["stomach", "abdomen", "nausea", "vomit", "bowel", "diarrhea", "stool", "appetite", "belly", "gut"],
    "Neurological": ["headache", "dizziness", "numb", "seizure", "faint", "confusion", "memory", "tremor", "vertigo", "vision"],
    "Musculoskeletal": ["joint", "muscle", "bone", "knee", "back", "leg", "arm", "stiff", "swell", "ache", "shoulder", "neck"],
    "Dermatological": ["skin", "rash", "itch", "spot", "burn", "lesion", "blister", "redness"],
    "Constitutional": ["fever", "fatigue", "weakness", "sweat", "weight", "chill"],
    "Genitourinary": ["urine", "bladder", "kidney", "painful urination", "flank"]
}

def identify_system(text):
    """
    Identifies the physiological system based on keywords in the text.
    Returns the system name (str) or 'General' if unclear.
    """
    text = text.lower()
    scores = {sys: 0 for sys in SYSTEM_KEYWORDS}
    
    for sys, keywords in SYSTEM_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[sys] += 1
                
    # Get max
    best_sys, score = max(scores.items(), key=lambda x: x[1])
    
    if score > 0:
        return best_sys
    return "General"
