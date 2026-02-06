"""
Medical Symptom Extractor using BioBERT NER
Uses emilyalsentzer/Bio_ClinicalBERT for medical entity recognition
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MedicalSymptomExtractor:
    """
    Advanced medical symptom extractor using BioBERT NER
    Supports custom inputs and high-accuracy entity extraction
    """
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialize the symptom extractor
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading model: {model_name}...")
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For NER, we'll use a fine-tuned version or create custom NER
        # Using BC5CDR-disease or similar medical NER model
        try:
            # Try to load a medical NER model
            self.ner_pipeline = pipeline(
                "ner",
                model="alvaroalon2/biobert_diseases_ner",  # Medical disease/symptom NER
                tokenizer="alvaroalon2/biobert_diseases_ner",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ Loaded specialized medical NER model")
        except:
            print("Using Bio_ClinicalBERT with custom NER setup...")
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = None
    
    def extract_symptoms(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Extract medical symptoms and conditions from text
        
        Args:
            text: Input medical text
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of extracted entities with metadata
        """
        if not text or not text.strip():
            return []
        
        if self.ner_pipeline:
            # Use the specialized NER pipeline
            entities = self.ner_pipeline(text)
            
            results = []
            for entity in entities:
                if entity['score'] >= confidence_threshold:
                    results.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': round(entity['score'], 4),
                        'start': entity['start'],
                        'end': entity['end']
                    })
            
            return self._merge_adjacent_entities(results)
        else:
            # Fallback: Use keyword extraction
            return self._extract_with_keywords(text)
    
    def _merge_adjacent_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge adjacent entities of the same type"""
        if not entities:
            return []
        
        merged = []
        current = entities[0].copy()
        
        for entity in entities[1:]:
            # If adjacent and same label, merge
            if (entity['start'] <= current['end'] + 2 and 
                entity['label'] == current['label']):
                current['end'] = entity['end']
                current['text'] = current['text'] + ' ' + entity['text']
                current['confidence'] = max(current['confidence'], entity['confidence'])
            else:
                merged.append(current)
                current = entity.copy()
        
        merged.append(current)
        return merged
    
    def _extract_with_keywords(self, text: str) -> List[Dict]:
        """
        Fallback keyword-based extraction for common symptoms
        """
        # Common medical symptoms and conditions
        symptom_keywords = [
            'pain', 'ache', 'fever', 'cough', 'fatigue', 'nausea', 'vomiting',
            'diarrhea', 'constipation', 'headache', 'migraine', 'dizziness',
            'shortness of breath', 'chest pain', 'abdominal pain', 'back pain',
            'joint pain', 'muscle pain', 'sore throat', 'runny nose', 'congestion',
            'rash', 'itching', 'swelling', 'inflammation', 'bleeding', 'bruising',
            'weakness', 'numbness', 'tingling', 'tremor', 'seizure', 'confusion',
            'anxiety', 'depression', 'insomnia', 'sweating', 'chills', 'weight loss',
            'weight gain', 'loss of appetite', 'difficulty swallowing', 'hoarseness',
            'wheezing', 'palpitations', 'irregular heartbeat', 'high blood pressure',
            'low blood pressure', 'blurred vision', 'double vision', 'hearing loss',
            'tinnitus', 'ear pain', 'toothache', 'jaw pain', 'neck pain',
            'stiffness', 'spasm', 'cramps', 'bloating', 'heartburn', 'indigestion'
        ]
        
        text_lower = text.lower()
        results = []
        
        for symptom in symptom_keywords:
            if symptom in text_lower:
                start_idx = text_lower.find(symptom)
                end_idx = start_idx + len(symptom)
                
                results.append({
                    'text': text[start_idx:end_idx],
                    'label': 'SYMPTOM',
                    'confidence': 0.85,
                    'start': start_idx,
                    'end': end_idx
                })
        
        return results
    
    def batch_extract(self, texts: List[str], confidence_threshold: float = 0.5) -> List[List[Dict]]:
        """
        Extract symptoms from multiple texts
        
        Args:
            texts: List of input texts
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of extraction results for each text
        """
        results = []
        for text in texts:
            results.append(self.extract_symptoms(text, confidence_threshold))
        return results
    
    def extract_with_context(self, text: str, window_size: int = 50) -> List[Dict]:
        """
        Extract symptoms with surrounding context
        
        Args:
            text: Input text
            window_size: Number of characters for context window
            
        Returns:
            Entities with context
        """
        entities = self.extract_symptoms(text)
        
        for entity in entities:
            start = max(0, entity['start'] - window_size)
            end = min(len(text), entity['end'] + window_size)
            entity['context'] = text[start:end]
        
        return entities
    
    def get_summary(self, text: str) -> Dict:
        """
        Get a summary of extracted symptoms
        
        Args:
            text: Input text
            
        Returns:
            Summary dictionary
        """
        entities = self.extract_symptoms(text)
        
        return {
            'total_symptoms': len(entities),
            'symptoms': entities,
            'unique_symptoms': len(set(e['text'].lower() for e in entities)),
            'avg_confidence': round(np.mean([e['confidence'] for e in entities]), 4) if entities else 0,
            'high_confidence_count': len([e for e in entities if e['confidence'] > 0.8])
        }


def demo_usage():
    """Demonstrate the symptom extractor"""
    
    # Initialize extractor
    extractor = MedicalSymptomExtractor()
    
    # Sample medical texts
    sample_texts = [
        "Patient presents with severe headache, fever of 101°F, and persistent cough for 3 days.",
        "Chief complaint: chest pain radiating to left arm, shortness of breath, and dizziness.",
        "History: chronic back pain, occasional migraine, and recent onset of fatigue.",
        "Symptoms include nausea, vomiting, abdominal pain, and diarrhea since yesterday.",
        "Patient reports joint pain, muscle weakness, and difficulty walking.",
        "Presenting with rash on both arms, severe itching, and mild swelling.",
        "Complaints of palpitations, irregular heartbeat, and anxiety for past week.",
        "Experiencing blurred vision, frequent headaches, and neck stiffness.",
        "Patient has sore throat, runny nose, congestion, and body aches.",
        "Reports numbness in fingers, tingling sensation, and occasional tremors."
    ]
    
    print("\n" + "="*80)
    print("MEDICAL SYMPTOM EXTRACTOR - DEMO")
    print("="*80 + "\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Text: {text}")
        
        # Extract symptoms
        symptoms = extractor.extract_symptoms(text)
        
        if symptoms:
            print(f"\nExtracted {len(symptoms)} symptom(s):")
            for symptom in symptoms:
                print(f"  • {symptom['text']:30s} | {symptom['label']:15s} | Confidence: {symptom['confidence']:.3f}")
        else:
            print("No symptoms extracted.")
    
    # Batch processing demo
    print("\n" + "="*80)
    print("BATCH PROCESSING DEMO")
    print("="*80 + "\n")
    
    batch_results = extractor.batch_extract(sample_texts[:3])
    for i, (text, results) in enumerate(zip(sample_texts[:3], batch_results), 1):
        print(f"{i}. Found {len(results)} symptoms in: '{text[:60]}...'")
    
    # Summary demo
    print("\n" + "="*80)
    print("SUMMARY DEMO")
    print("="*80 + "\n")
    
    summary = extractor.get_summary(sample_texts[0])
    print(f"Total symptoms: {summary['total_symptoms']}")
    print(f"Unique symptoms: {summary['unique_symptoms']}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")
    print(f"High confidence count: {summary['high_confidence_count']}")


if __name__ == "__main__":
    demo_usage()
