"""
Production-Ready Medical Symptom NER System
Complete system with API, batch processing, and evaluation
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from typing import List, Dict, Optional, Union
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class Entity:
    """Entity data class"""
    text: str
    label: str
    confidence: float
    start: int
    end: int
    context: Optional[str] = None


class ProductionMedicalNER:
    """
    Production-ready medical NER system with multiple models and fallback
    """
    
    def __init__(
        self,
        primary_model: str = "alvaroalon2/biobert_diseases_ner",
        fallback_enabled: bool = True,
        device: Optional[int] = None
    ):
        """
        Initialize NER system
        
        Args:
            primary_model: Primary model to use
            fallback_enabled: Enable keyword fallback
            device: Device to use (-1 for CPU, 0+ for GPU)
        """
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        self.device = device
        self.fallback_enabled = fallback_enabled
        
        # Load primary model
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=primary_model,
                aggregation_strategy="simple",
                device=device
            )
            self.model_loaded = True
            print(f"✓ Loaded model: {primary_model}")
            print(f"✓ Using device: {'GPU' if device >= 0 else 'CPU'}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model_loaded = False
        
        # Medical terminology for fallback
        self.medical_patterns = self._load_medical_patterns()
        
    def _load_medical_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive medical patterns"""
        return {
            'SYMPTOM': [
                # Pain-related
                r'\b(?:severe|acute|chronic|mild|sharp|dull|throbbing|burning|stabbing)\s+pain\b',
                r'\bpain(?:ful)?\b',
                r'\b(?:head|stomach|abdominal|chest|back|neck|joint|muscle)ache\b',
                
                # Respiratory
                r'\b(?:short(?:ness)?\s+of\s+breath|dyspnea|wheezing|cough(?:ing)?|sputum)\b',
                
                # Cardiovascular
                r'\b(?:palpitations?|tachycardia|bradycardia|arrhythmia)\b',
                r'\b(?:chest\s+pain|angina|heart\s+pain)\b',
                
                # Gastrointestinal
                r'\b(?:nausea|vomit(?:ing)?|diarrhea|constipation|bloating)\b',
                r'\b(?:abdominal\s+(?:pain|cramps|distension)|heartburn|indigestion)\b',
                
                # Neurological
                r'\b(?:headache|migraine|dizziness|vertigo|seizure|tremor)\b',
                r'\b(?:numbness|tingling|weakness|paralysis|confusion)\b',
                
                # General
                r'\b(?:fever|chills|fatigue|malaise|weakness)\b',
                r'\b(?:sweating|night\s+sweats|hot\s+flashes)\b',
                r'\b(?:weight\s+(?:loss|gain)|loss\s+of\s+appetite)\b',
                
                # Dermatological
                r'\b(?:rash|itching|pruritus|swelling|edema|inflammation)\b',
                
                # Musculoskeletal
                r'\b(?:stiffness|spasm|cramps?|soreness)\b',
            ],
            'DISEASE': [
                r'\b(?:diabetes|hypertension|cancer|asthma|copd)\b',
                r'\b(?:arthritis|pneumonia|bronchitis|infection)\b',
                r'\b(?:heart\s+disease|stroke|myocardial\s+infarction)\b',
            ]
        }
    
    def extract(
        self,
        text: str,
        confidence_threshold: float = 0.5,
        use_fallback: bool = True,
        return_context: bool = False,
        context_window: int = 50
    ) -> List[Entity]:
        """
        Extract medical entities from text
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence score
            use_fallback: Use pattern-based fallback
            return_context: Include surrounding context
            context_window: Characters for context
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        # Primary model extraction
        if self.model_loaded:
            try:
                raw_entities = self.ner_pipeline(text)
                
                for ent in raw_entities:
                    if ent['score'] >= confidence_threshold:
                        entity = Entity(
                            text=ent['word'].strip(),
                            label=ent['entity_group'],
                            confidence=round(ent['score'], 4),
                            start=ent['start'],
                            end=ent['end']
                        )
                        entities.append(entity)
            except Exception as e:
                print(f"Model extraction failed: {e}")
        
        # Fallback to pattern matching
        if use_fallback and self.fallback_enabled:
            pattern_entities = self._extract_with_patterns(text, confidence_threshold)
            entities.extend(pattern_entities)
        
        # Remove duplicates
        entities = self._deduplicate_entities(entities)
        
        # Add context if requested
        if return_context:
            for entity in entities:
                start = max(0, entity.start - context_window)
                end = min(len(text), entity.end + context_window)
                entity.context = text[start:end]
        
        return entities
    
    def _extract_with_patterns(self, text: str, confidence: float) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        for label, patterns in self.medical_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        text=match.group(),
                        label=label,
                        confidence=confidence,
                        start=match.start(),
                        end=match.end()
                    )
                    entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate and overlapping entities"""
        if not entities:
            return []
        
        # Sort by start position and confidence
        entities = sorted(entities, key=lambda e: (e.start, -e.confidence))
        
        deduplicated = []
        used_spans = set()
        
        for entity in entities:
            span = (entity.start, entity.end)
            
            # Check for overlap
            overlaps = False
            for used_start, used_end in used_spans:
                if not (entity.end <= used_start or entity.start >= used_end):
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
                used_spans.add(span)
        
        return deduplicated
    
    def batch_extract(
        self,
        texts: List[str],
        confidence_threshold: float = 0.5,
        show_progress: bool = True
    ) -> List[List[Entity]]:
        """
        Batch extraction from multiple texts
        
        Args:
            texts: List of input texts
            confidence_threshold: Minimum confidence
            show_progress: Show progress indicator
            
        Returns:
            List of entity lists
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts, 1):
            entities = self.extract(text, confidence_threshold)
            results.append(entities)
            
            if show_progress and i % 10 == 0:
                print(f"Processed {i}/{total} texts...")
        
        return results
    
    def get_statistics(self, entities: List[Entity]) -> Dict:
        """Get statistics from extracted entities"""
        if not entities:
            return {
                'total': 0,
                'by_label': {},
                'avg_confidence': 0.0,
                'confidence_distribution': {}
            }
        
        label_counts = defaultdict(int)
        confidence_scores = []
        
        for entity in entities:
            label_counts[entity.label] += 1
            confidence_scores.append(entity.confidence)
        
        return {
            'total': len(entities),
            'unique': len(set(e.text.lower() for e in entities)),
            'by_label': dict(label_counts),
            'avg_confidence': round(np.mean(confidence_scores), 4),
            'min_confidence': round(min(confidence_scores), 4),
            'max_confidence': round(max(confidence_scores), 4),
            'high_confidence_count': sum(1 for e in entities if e.confidence > 0.8)
        }
    
    def export_to_json(self, entities: List[Entity], filepath: str):
        """Export entities to JSON file"""
        data = [asdict(e) for e in entities]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Exported {len(entities)} entities to {filepath}")
    
    def load_from_json(self, filepath: str) -> List[Entity]:
        """Load entities from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [Entity(**item) for item in data]


def comprehensive_demo():
    """Comprehensive demonstration of the system"""
    
    print("\n" + "="*80)
    print("PRODUCTION MEDICAL NER SYSTEM - COMPREHENSIVE DEMO")
    print("="*80 + "\n")
    
    # Initialize system
    ner = ProductionMedicalNER()
    
    # Test cases
    test_cases = [
        {
            'text': "Patient presents with severe headache, high fever of 103°F, persistent dry cough, and fatigue lasting 5 days.",
            'description': "Multiple symptoms"
        },
        {
            'text': "Chief complaint: acute chest pain radiating to left arm, shortness of breath, and palpitations for 2 hours.",
            'description': "Cardiac symptoms"
        },
        {
            'text': "History: chronic lower back pain, osteoarthritis in both knees, and occasional migraine headaches.",
            'description': "Chronic conditions"
        },
        {
            'text': "Symptoms include nausea, projectile vomiting, severe abdominal cramping, and watery diarrhea since last night.",
            'description': "GI symptoms"
        },
        {
            'text': "Patient reports numbness in fingers, tingling sensation in toes, muscle weakness, and difficulty walking.",
            'description': "Neurological symptoms"
        },
        {
            'text': "Presenting with widespread rash, severe itching, facial swelling, and difficulty breathing after medication.",
            'description': "Allergic reaction"
        },
        {
            'text': "Diagnosed with type 2 diabetes mellitus, hypertension, and hyperlipidemia 3 years ago.",
            'description': "Multiple diseases"
        },
        {
            'text': "Complaints of irregular heartbeat, excessive sweating, tremor, anxiety, and unintentional weight loss.",
            'description': "Thyroid symptoms"
        },
        {
            'text': "Patient has persistent sore throat, runny nose, nasal congestion, and body aches for one week.",
            'description': "Upper respiratory"
        },
        {
            'text': "Reports blurred vision in right eye, frequent headaches, neck stiffness, and photophobia.",
            'description': "Visual/neuro symptoms"
        }
    ]
    
    # Process each case
    all_results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {case['description']}")
        print(f"{'='*80}")
        print(f"Text: {case['text']}\n")
        
        # Extract entities
        entities = ner.extract(case['text'], confidence_threshold=0.3)
        
        if entities:
            print(f"✓ Extracted {len(entities)} entities:\n")
            for entity in entities:
                print(f"  📌 {entity.text:30s} | {entity.label:15s} | Confidence: {entity.confidence:.3f}")
            
            # Statistics
            stats = ner.get_statistics(entities)
            print(f"\n  Statistics:")
            print(f"    • Total entities: {stats['total']}")
            print(f"    • Unique entities: {stats['unique']}")
            print(f"    • Average confidence: {stats['avg_confidence']:.3f}")
            print(f"    • High confidence (>0.8): {stats['high_confidence_count']}")
            
            if stats['by_label']:
                print(f"    • By label: {dict(stats['by_label'])}")
        else:
            print("  ✗ No entities extracted")
        
        all_results.append(entities)
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}\n")
    
    total_entities = sum(len(r) for r in all_results)
    all_entities = [e for r in all_results for e in r]
    
    if all_entities:
        overall_stats = ner.get_statistics(all_entities)
        print(f"Total texts processed: {len(test_cases)}")
        print(f"Total entities extracted: {overall_stats['total']}")
        print(f"Unique entities: {overall_stats['unique']}")
        print(f"Average confidence: {overall_stats['avg_confidence']:.3f}")
        print(f"Confidence range: {overall_stats['min_confidence']:.3f} - {overall_stats['max_confidence']:.3f}")
        print(f"\nEntity distribution:")
        for label, count in overall_stats['by_label'].items():
            print(f"  • {label}: {count}")
    
    # Batch processing demo
    print(f"\n{'='*80}")
    print("BATCH PROCESSING DEMO")
    print(f"{'='*80}\n")
    
    batch_texts = [case['text'] for case in test_cases[:5]]
    batch_results = ner.batch_extract(batch_texts, show_progress=False)
    
    print(f"Processed {len(batch_texts)} texts in batch:")
    for i, (text, results) in enumerate(zip(batch_texts, batch_results), 1):
        print(f"  {i}. Found {len(results)} entities in: '{text[:60]}...'")
    
    print(f"\n{'='*80}")
    print("✓ DEMO COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    comprehensive_demo()
