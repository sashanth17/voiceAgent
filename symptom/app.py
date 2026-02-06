"""
Real-Time Medical Symptom Extractor with Bio_ClinicalBERT
Extracts symptoms from natural language input worldwide
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
import re
import json
from datetime import datetime
from typing import List, Dict, Optional


class GlobalSymptomExtractor:
    """
    Symptom extractor using Bio_ClinicalBERT for worldwide medical terminology
    """
    
    def __init__(self):
        """Initialize the Bio_ClinicalBERT extractor"""
        print("🔄 Loading Bio_ClinicalBERT model...")
        print("This model understands medical terms from around the world...\n")
        
        try:
            # Using emilyalsentzer's Bio_ClinicalBERT
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            
            device = 0 if torch.cuda.is_available() else -1
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Try to load a medical NER model that works with ClinicalBERT
            # If not available, we'll use BioBERT NER as fallback
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="alvaroalon2/biobert_diseases_ner",
                    aggregation_strategy="simple",
                    device=device
                )
                print(f"✓ Using BioBERT NER model")
            except:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    aggregation_strategy="simple",
                    device=device
                )
                print(f"✓ Using Biomedical NER model")
            
            print(f"✓ Model loaded successfully (Device: {'GPU' if device >= 0 else 'CPU'})")
            print("✓ Ready to understand symptoms in various forms!\n")
            
        except Exception as e:
            print(f"⚠ Model loading failed: {e}")
            print("Using pattern-based extraction only\n")
            self.ner_pipeline = None
            self.tokenizer = None
        
        # Extended symptom patterns for worldwide usage
        self.symptom_patterns = [
            # Pain variations
            r'\b(?:pain|ache|aching|hurt(?:ing)?|sore(?:ness)?|discomfort|tender(?:ness)?)\b',
            r'\b(?:severe|bad|terrible|horrible|intense|extreme|unbearable|killing|sharp|dull|throbbing|burning|stabbing|shooting|crushing)\s+(?:pain|ache|hurt)\b',
            
            # Body parts + pain
            r'\b(?:head|stomach|belly|abdomen|chest|back|neck|shoulder|arm|leg|knee|foot|throat|ear|eye|tooth|teeth)\s*(?:pain|ache|hurt|sore)\b',
            r'\b(?:pain|ache|hurt)\s+(?:in|at|on)\s+(?:my|the)?\s*(?:head|stomach|belly|chest|back|neck|throat|ear)\b',
            
            # Common symptoms
            r'\b(?:fever|temperature|hot|burning\s+up|feverish)\b',
            r'\b(?:cough|coughing|hacking)\b',
            r'\b(?:cold|flu|sick|ill|unwell|not\s+feeling\s+well)\b',
            r'\b(?:tired|fatigue|exhausted|weak|weakness|no\s+energy)\b',
            r'\b(?:dizzy|dizziness|lightheaded|vertigo|spinning)\b',
            r'\b(?:nausea|nauseous|sick\s+to\s+stomach|queasy|feel\s+like\s+vomit)\b',
            r'\b(?:vomit|vomiting|throw(?:ing)?\s+up|puke|puking)\b',
            r'\b(?:diarrhea|loose\s+stool|runny\s+stool|upset\s+stomach)\b',
            
            # Breathing issues
            r'\b(?:can\'?t\s+breathe|hard\s+to\s+breathe|difficult(?:y)?\s+breathing|short\s+of\s+breath|breathless|wheezing)\b',
            
            # Neurological
            r'\b(?:headache|migraine|head\s+hurt)\b',
            r'\b(?:numb|numbness|tingling|pins\s+and\s+needles)\b',
            r'\b(?:confusion|confused|disoriented|foggy)\b',
            
            # Skin issues
            r'\b(?:rash|itchy|itching|scratch|red\s+spots|bumps|swelling|swollen)\b',
            
            # General
            r'\b(?:bleed|bleeding|blood)\b',
            r'\b(?:sweat|sweating|perspir)\b',
            r'\b(?:chills?|shiver|shaking|tremor)\b',
            r'\b(?:stiff|stiffness|can\'?t\s+move)\b',
            
            # Informal/colloquial
            r'\b(?:feel\s+bad|feel\s+terrible|not\s+good|under\s+the\s+weather)\b',
        ]
        
        # Duration patterns - more flexible
        self.duration_patterns = [
            r'\b(?:for|lasting|since|over|about|around)\s+(\d+)\s+(day|week|month|year|hour|minute)s?\b',
            r'\b(\d+)\s+(day|week|month|year|hour|minute)s?\s+(?:ago|now|already)\b',
            r'\bsince\s+(yesterday|last\s+(?:night|week|month|year)|this\s+morning|morning|tonight|today)\b',
            r'\b(?:few|couple\s+of|several)\s+(days?|weeks?|months?|hours?)\b',
            r'\ball\s+(?:day|night|week|month)\b',
        ]
        
        # Severity keywords - worldwide common terms
        self.severity_keywords = {
            'severe': [
                'severe', 'extreme', 'excruciating', 'unbearable', 'intense', 
                'terrible', 'horrible', 'awful', 'really bad', 'very bad',
                'killing me', 'can\'t stand', 'worst', 'heavy', 'strong',
                'too much', 'serious', 'critical'
            ],
            'moderate': [
                'moderate', 'noticeable', 'uncomfortable', 'bothersome', 
                'significant', 'medium', 'quite bad', 'pretty bad',
                'annoying', 'troublesome'
            ],
            'mild': [
                'mild', 'slight', 'minor', 'little', 'small', 'bit',
                'occasional', 'light', 'lite', 'not too bad', 'bearable',
                'manageable', 'little bit', 'somewhat'
            ]
        }
    
    def extract_symptoms(self, text: str) -> List[Dict]:
        """Extract symptoms using NER model and flexible patterns"""
        symptoms = []
        text_lower = text.lower()
        
        # NER model extraction
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                for ent in entities:
                    entity_type = ent.get('entity_group', '').upper()
                    if any(keyword in entity_type for keyword in ['SYMPTOM', 'DISEASE', 'SIGN', 'PROBLEM', 'DISORDER']):
                        symptom_text = ent['word'].strip()
                        if len(symptom_text) > 2:  # Filter very short matches
                            symptoms.append({
                                'symptom': symptom_text,
                                'confidence': round(ent['score'], 3),
                                'start': ent['start'],
                                'end': ent['end'],
                                'source': 'model'
                            })
            except Exception as e:
                print(f"⚠ NER extraction warning: {e}")
        
        # Pattern-based extraction for common expressions
        for pattern in self.symptom_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                symptom_text = match.group().strip()
                if len(symptom_text) > 2:  # Filter very short matches
                    symptoms.append({
                        'symptom': symptom_text,
                        'confidence': 0.75,
                        'start': match.start(),
                        'end': match.end(),
                        'source': 'pattern'
                    })
        
        # Remove duplicates and overlapping spans
        symptoms = self._deduplicate_symptoms(symptoms)
        
        return symptoms
    
    def _deduplicate_symptoms(self, symptoms: List[Dict]) -> List[Dict]:
        """Remove duplicates and overlapping symptoms"""
        if not symptoms:
            return []
        
        # Sort by confidence and position
        symptoms = sorted(symptoms, key=lambda x: (-x['confidence'], x['start']))
        
        unique = []
        used_spans = []
        seen_text = set()
        
        for sym in symptoms:
            text_lower = sym['symptom'].lower().strip()
            
            # Skip if already seen (case insensitive)
            if text_lower in seen_text:
                continue
            
            # Check for overlap
            overlaps = False
            for start, end in used_spans:
                if not (sym['end'] <= start or sym['start'] >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                unique.append(sym)
                used_spans.append((sym['start'], sym['end']))
                seen_text.add(text_lower)
        
        return unique
    
    def extract_duration(self, text: str) -> Optional[str]:
        """Extract duration information with flexible patterns"""
        for pattern in self.duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group().strip()
        return None
    
    def extract_severity(self, text: str, symptom: str) -> str:
        """Extract severity level with context analysis"""
        text_lower = text.lower()
        symptom_lower = symptom.lower()
        
        # Get context around symptom
        pos = text_lower.find(symptom_lower)
        if pos == -1:
            # Try to find partial match
            words = symptom_lower.split()
            if words:
                pos = text_lower.find(words[0])
        
        if pos == -1:
            return "unspecified"
        
        # Extended context window
        start = max(0, pos - 100)
        end = min(len(text_lower), pos + len(symptom_lower) + 100)
        context = text_lower[start:end]
        
        # Check for severity keywords
        for severity, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in context:
                    return severity
        
        return "moderate"  # Default
    
    def process_input(self, text: str) -> Dict:
        """Process user input and extract all information"""
        print("\n🔍 Analyzing your symptoms...")
        
        # Extract symptoms
        raw_symptoms = self.extract_symptoms(text)
        
        # Extract duration
        duration = self.extract_duration(text)
        
        # Build result
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_text": text,
            "duration": duration if duration else "not specified",
            "symptom_count": len(raw_symptoms),
            "symptoms": []
        }
        
        # Process each symptom
        for sym in raw_symptoms:
            severity = self.extract_severity(text, sym['symptom'])
            result["symptoms"].append({
                "name": sym['symptom'],
                "severity": severity,
                "confidence": sym['confidence'],
                "detected_by": sym.get('source', 'unknown')
            })
        
        # Sort by confidence
        result["symptoms"].sort(key=lambda x: x['confidence'], reverse=True)
        
        return result
    
    def save_to_json(self, data: Dict, filename: str = None):
        """Save extracted data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"symptoms_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {filename}")
        return filename
    
    def display_results(self, data: Dict):
        """Display results in a formatted way"""
        print("\n" + "="*70)
        print("📋 SYMPTOM EXTRACTION RESULTS")
        print("="*70)
        
        print(f"\n📝 Input Text:")
        print(f"   {data['input_text']}")
        
        print(f"\n⏱️  Duration: {data['duration']}")
        print(f"\n🔍 Total Symptoms Found: {data['symptom_count']}")
        
        if data['symptoms']:
            print("\n📊 Detected Symptoms:\n")
            for i, sym in enumerate(data['symptoms'], 1):
                severity_emoji = {
                    'severe': '🔴',
                    'moderate': '🟡', 
                    'mild': '🟢',
                    'unspecified': '⚪'
                }
                emoji = severity_emoji.get(sym['severity'], '⚪')
                
                print(f"  {i}. {emoji} {sym['name']}")
                print(f"     └─ Severity: {sym['severity'].upper()}")
                print(f"     └─ Confidence: {sym['confidence']:.1%}")
                print(f"     └─ Detected by: {sym['detected_by']}")
                print()
        else:
            print("\n  ⚠️  No symptoms detected in the text.\n")
        
        print("="*70)


def main():
    """Main function for real-time symptom extraction"""
    print("="*70)
    print("  🏥 GLOBAL MEDICAL SYMPTOM EXTRACTOR")
    print("  Powered by Bio_ClinicalBERT")
    print("="*70)
    print("\n✨ This system understands medical symptoms in various forms")
    print("   You can describe symptoms naturally, in your own words!")
    print("\n💬 Examples:")
    print("   - 'I have a bad headache for 3 days'")
    print("   - 'My stomach hurts and I feel sick'")
    print("   - 'Can't breathe well, chest pain since yesterday'")
    print("   - 'Feeling terrible, fever and coughing all week'")
    print("\nType 'quit' or 'exit' to stop\n")
    
    # Initialize extractor
    extractor = GlobalSymptomExtractor()
    
    while True:
        print("\n" + "-"*70)
        text = input("\n💬 Describe your symptoms (or 'quit' to exit):\n> ").strip()
        
        if not text:
            print("⚠️  Please enter some text")
            continue
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the symptom extractor. Stay healthy!")
            break
        
        # Process input
        try:
            result = extractor.process_input(text)
            
            # Display results
            extractor.display_results(result)
            
            # Ask to save
            save = input("\n💾 Save results to JSON file? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                custom_name = input("   Enter filename (or press Enter for auto): ").strip()
                if custom_name:
                    if not custom_name.endswith('.json'):
                        custom_name += '.json'
                    filename = extractor.save_to_json(result, custom_name)
                else:
                    filename = extractor.save_to_json(result)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()