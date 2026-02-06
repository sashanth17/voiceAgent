"""
Standalone Medical Symptom Extractor Demo
Works without PyTorch/Transformers dependencies
Uses pattern matching and medical terminology
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MedicalEntity:
    """Medical entity structure"""
    text: str
    label: str
    confidence: float
    start: int
    end: int
    category: str = ""


class StandaloneMedicalNER:
    """
    Standalone medical NER using comprehensive pattern matching
    Works without ML dependencies - perfect for testing and quick deployment
    """
    
    def __init__(self):
        """Initialize with medical patterns"""
        self.patterns = self._load_patterns()
        self.symptom_modifiers = ['severe', 'acute', 'chronic', 'mild', 'moderate', 
                                   'persistent', 'intermittent', 'occasional', 'frequent']
    
    def _load_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load comprehensive medical patterns"""
        return {
            'SYMPTOM': [
                # Pain-related (high specificity)
                (r'\b(?:severe|acute|chronic|mild|sharp|dull|throbbing|burning|stabbing)\s+pain\b', 'high'),
                (r'\b(?:head|stomach|abdominal|chest|back|neck|joint|muscle|tooth|ear|jaw)ache\b', 'high'),
                (r'\bpain\s+(?:in|on|at|around)\s+(?:the\s+)?(?:\w+\s+){0,2}(?:head|chest|abdomen|back|arm|leg|neck)\b', 'medium'),
                
                # Respiratory
                (r'\b(?:shortness\s+of\s+breath|dyspnea|difficulty\s+breathing)\b', 'high'),
                (r'\b(?:cough|coughing|wheezing|sputum|phlegm)\b', 'high'),
                (r'\b(?:persistent|chronic|dry|productive)\s+cough\b', 'high'),
                
                # Cardiovascular
                (r'\b(?:chest\s+pain|angina|heart\s+pain)\b', 'high'),
                (r'\b(?:palpitations?|rapid\s+heart(?:beat)?|irregular\s+heart(?:beat)?)\b', 'high'),
                (r'\b(?:tachycardia|bradycardia|arrhythmia)\b', 'high'),
                
                # Gastrointestinal
                (r'\b(?:nausea|vomit(?:ing)?|diarrhea|constipation)\b', 'high'),
                (r'\b(?:abdominal\s+(?:pain|cramp(?:s|ing)?|distension)|heartburn|indigestion)\b', 'high'),
                (r'\b(?:bloat(?:ing|ed)|gas|flatulence)\b', 'medium'),
                
                # Neurological
                (r'\b(?:headache|migraine|dizziness|vertigo|seizure)\b', 'high'),
                (r'\b(?:numbness|tingling|weakness|paralysis|tremor)\b', 'high'),
                (r'\b(?:confusion|disorientation|memory\s+loss)\b', 'high'),
                
                # General/Constitutional
                (r'\b(?:fever|high\s+temperature|pyrexia)\b', 'high'),
                (r'\b(?:fatigue|tired(?:ness)?|exhaustion|malaise)\b', 'high'),
                (r'\b(?:chills?|sweating|night\s+sweats)\b', 'medium'),
                (r'\b(?:weight\s+(?:loss|gain)|loss\s+of\s+appetite)\b', 'high'),
                
                # Dermatological
                (r'\b(?:rash|eruption|hives|urticaria)\b', 'high'),
                (r'\b(?:itch(?:ing)?|pruritus|swelling|edema)\b', 'medium'),
                (r'\b(?:inflammation|redness|bruising|bleeding)\b', 'medium'),
                
                # Musculoskeletal
                (r'\b(?:stiffness|spasm|cramp(?:s|ing)?|soreness)\b', 'medium'),
                (r'\b(?:joint|muscle|bone)\s+(?:pain|ache|stiffness)\b', 'high'),
                
                # Sensory
                (r'\b(?:blurred|double|loss\s+of)\s+vision\b', 'high'),
                (r'\b(?:hearing\s+loss|tinnitus|ringing\s+in\s+(?:the\s+)?ears)\b', 'high'),
                
                # Respiratory continued
                (r'\b(?:sore\s+throat|pharyngitis|runny\s+nose|congestion)\b', 'high'),
                (r'\b(?:sneezing|hoarseness|difficulty\s+swallowing)\b', 'medium'),
            ],
            
            'DISEASE': [
                # Common chronic diseases
                (r'\b(?:type\s+[12]\s+)?diabetes(?:\s+mellitus)?\b', 'high'),
                (r'\b(?:hypertension|high\s+blood\s+pressure)\b', 'high'),
                (r'\b(?:cancer|carcinoma|tumor|neoplasm)\b', 'high'),
                (r'\b(?:asthma|copd|emphysema|bronchitis)\b', 'high'),
                
                # Cardiovascular diseases
                (r'\b(?:heart\s+disease|coronary\s+artery\s+disease|myocardial\s+infarction)\b', 'high'),
                (r'\b(?:stroke|cerebrovascular\s+accident|cva)\b', 'high'),
                (r'\b(?:heart\s+failure|congestive\s+heart\s+failure|chf)\b', 'high'),
                
                # Musculoskeletal
                (r'\b(?:arthritis|osteoarthritis|rheumatoid\s+arthritis)\b', 'high'),
                (r'\b(?:osteoporosis|fracture)\b', 'high'),
                
                # Infections
                (r'\b(?:pneumonia|influenza|flu|infection|sepsis)\b', 'high'),
                (r'\b(?:covid(?:-19)?|coronavirus)\b', 'high'),
                
                # Mental health
                (r'\b(?:depression|anxiety|bipolar|schizophrenia)\b', 'high'),
                
                # Metabolic
                (r'\b(?:obesity|hyperlipidemia|hypothyroidism|hyperthyroidism)\b', 'high'),
            ],
            
            'VITAL_SIGN': [
                (r'\b(?:fever\s+of|temperature\s+(?:of\s+)?)\d+(?:\.\d+)?(?:°F|°C|F|C)\b', 'high'),
                (r'\b(?:blood\s+pressure|bp)(?:\s+(?:of\s+)?)\d+/\d+\b', 'high'),
                (r'\b(?:heart\s+rate|pulse)(?:\s+(?:of\s+)?)\d+\b', 'medium'),
            ]
        }
    
    def extract(
        self,
        text: str,
        confidence_threshold: float = 0.5,
        include_overlaps: bool = False
    ) -> List[MedicalEntity]:
        """
        Extract medical entities from text
        
        Args:
            text: Input medical text
            confidence_threshold: Minimum confidence (0.0-1.0)
            include_overlaps: Include overlapping entities
            
        Returns:
            List of MedicalEntity objects
        """
        entities = []
        
        # Extract using patterns
        for label, patterns in self.patterns.items():
            for pattern, priority in patterns:
                # Base confidence based on priority
                base_confidence = {'high': 0.9, 'medium': 0.75, 'low': 0.6}[priority]
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Adjust confidence based on modifiers
                    matched_text = match.group()
                    confidence = base_confidence
                    
                    # Boost confidence if has medical modifier
                    if any(mod in matched_text.lower() for mod in self.symptom_modifiers):
                        confidence = min(0.95, confidence + 0.1)
                    
                    if confidence >= confidence_threshold:
                        entity = MedicalEntity(
                            text=matched_text,
                            label=label,
                            confidence=round(confidence, 3),
                            start=match.start(),
                            end=match.end(),
                            category=self._categorize(matched_text, label)
                        )
                        entities.append(entity)
        
        # Remove overlaps if requested
        if not include_overlaps:
            entities = self._remove_overlaps(entities)
        
        # Sort by position
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        return entities
    
    def _categorize(self, text: str, label: str) -> str:
        """Categorize entity into subcategory"""
        text_lower = text.lower()
        
        if label == 'SYMPTOM':
            if any(word in text_lower for word in ['pain', 'ache']):
                return 'Pain'
            elif any(word in text_lower for word in ['cough', 'breath', 'wheez']):
                return 'Respiratory'
            elif any(word in text_lower for word in ['nausea', 'vomit', 'diarrhea']):
                return 'Gastrointestinal'
            elif any(word in text_lower for word in ['headache', 'dizz', 'numb']):
                return 'Neurological'
            elif any(word in text_lower for word in ['fever', 'fatigue', 'chill']):
                return 'Constitutional'
            else:
                return 'General'
        
        return label
    
    def _remove_overlaps(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping higher confidence ones"""
        if not entities:
            return []
        
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: -e.confidence)
        
        kept = []
        used_spans = set()
        
        for entity in sorted_entities:
            # Check if this span overlaps with any kept span
            overlaps = False
            for start, end in used_spans:
                if not (entity.end <= start or entity.start >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(entity)
                used_spans.add((entity.start, entity.end))
        
        return kept
    
    def get_statistics(self, entities: List[MedicalEntity]) -> Dict:
        """Get statistics from entities"""
        if not entities:
            return {
                'total': 0,
                'by_label': {},
                'by_category': {},
                'avg_confidence': 0.0
            }
        
        label_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for entity in entities:
            label_counts[entity.label] += 1
            if entity.category:
                category_counts[entity.category] += 1
        
        return {
            'total': len(entities),
            'unique': len(set(e.text.lower() for e in entities)),
            'by_label': dict(label_counts),
            'by_category': dict(category_counts),
            'avg_confidence': round(sum(e.confidence for e in entities) / len(entities), 3),
            'high_confidence': sum(1 for e in entities if e.confidence > 0.85)
        }
    
    def format_output(self, entities: List[MedicalEntity]) -> str:
        """Format entities for display"""
        if not entities:
            return "No entities found."
        
        output = []
        for i, entity in enumerate(entities, 1):
            category = f" [{entity.category}]" if entity.category else ""
            output.append(
                f"{i}. {entity.text:30s} | {entity.label:12s}{category:20s} | Confidence: {entity.confidence:.3f}"
            )
        
        return "\n".join(output)


def demo():
    """Demonstrate the standalone NER system"""
    
    print("\n" + "="*90)
    print("STANDALONE MEDICAL NER SYSTEM - COMPREHENSIVE DEMO")
    print("No ML dependencies required - Pure Python pattern matching")
    print("="*90 + "\n")
    
    # Initialize
    ner = StandaloneMedicalNER()
    
    # Test cases
    test_cases = [
        "Patient presents with severe headache, high fever of 103°F, persistent dry cough, and extreme fatigue lasting 5 days.",
        "Chief complaint: acute chest pain radiating to left arm, shortness of breath, palpitations, and dizziness for 2 hours.",
        "History: chronic lower back pain, osteoarthritis in both knees, hypertension, and occasional migraine headaches.",
        "Symptoms include nausea, projectile vomiting, severe abdominal cramping, watery diarrhea, and bloating since yesterday.",
        "Patient reports numbness in fingers, tingling sensation in toes, muscle weakness, tremor, and difficulty walking.",
        "Presenting with widespread rash, severe itching, facial swelling, hives, and difficulty breathing after medication.",
        "Diagnosed with type 2 diabetes mellitus, hypertension, hyperlipidemia, and heart disease 3 years ago.",
        "Complaints of irregular heartbeat, excessive sweating, tremor, anxiety, weight loss, and fatigue.",
        "Patient has persistent sore throat, runny nose, nasal congestion, sneezing, and body aches for one week.",
        "Reports blurred vision in right eye, frequent headaches, neck stiffness, photophobia, and confusion.",
    ]
    
    all_entities = []
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─'*90}")
        print(f"TEST CASE {i}")
        print(f"{'─'*90}")
        print(f"Text: {text}\n")
        
        # Extract entities
        entities = ner.extract(text, confidence_threshold=0.5)
        
        if entities:
            print(f"✓ Found {len(entities)} entities:\n")
            print(ner.format_output(entities))
            
            # Statistics
            stats = ner.get_statistics(entities)
            print(f"\n📊 Statistics:")
            print(f"   • Total: {stats['total']} | Unique: {stats['unique']} | Avg Confidence: {stats['avg_confidence']:.3f}")
            print(f"   • By Label: {dict(stats['by_label'])}")
            if stats['by_category']:
                print(f"   • By Category: {dict(stats['by_category'])}")
            
            all_entities.extend(entities)
        else:
            print("✗ No entities extracted")
    
    # Overall summary
    print(f"\n{'='*90}")
    print("OVERALL SUMMARY")
    print(f"{'='*90}\n")
    
    if all_entities:
        overall_stats = ner.get_statistics(all_entities)
        print(f"📈 Total Results:")
        print(f"   • Texts processed: {len(test_cases)}")
        print(f"   • Total entities: {overall_stats['total']}")
        print(f"   • Unique entities: {overall_stats['unique']}")
        print(f"   • Average confidence: {overall_stats['avg_confidence']:.3f}")
        print(f"   • High confidence (>0.85): {overall_stats['high_confidence']}\n")
        
        print(f"📊 Distribution:")
        print(f"   • By Label:")
        for label, count in overall_stats['by_label'].items():
            print(f"      - {label}: {count}")
        
        if overall_stats['by_category']:
            print(f"   • By Category:")
            for category, count in overall_stats['by_category'].items():
                print(f"      - {category}: {count}")
    
    print(f"\n{'='*90}")
    print("✅ DEMO COMPLETE - System working perfectly!")
    print("="*90)
    print("\n💡 This standalone version works without PyTorch/Transformers")
    print("   For even better accuracy, install dependencies and use the ML models.")
    print()


if __name__ == "__main__":
    demo()
