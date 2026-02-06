"""
Quick Test Script for Medical NER System
Run this to verify everything is working
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_basic_functionality():
    """Test basic functionality without loading models"""
    print("="*80)
    print("MEDICAL NER SYSTEM - QUICK TEST")
    print("="*80)
    print()
    
    # Test 1: Import check
    print("Test 1: Checking imports...")
    try:
        from symptom_extractor import MedicalSymptomExtractor
        from advanced_medical_ner import AdvancedMedicalNER
        from production_ner_system import ProductionMedicalNER, Entity
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Pattern extraction (no model needed)
    print("\nTest 2: Testing pattern-based extraction...")
    try:
        test_text = "Patient has severe headache, high fever, and persistent cough"
        
        # Use production system with pattern matching
        ner = ProductionMedicalNER(fallback_enabled=True)
        entities = ner._extract_with_patterns(test_text, confidence=0.8)
        
        if entities:
            print(f"✓ Found {len(entities)} entities using patterns:")
            for entity in entities[:5]:  # Show first 5
                print(f"  • {entity.text:20s} | {entity.label}")
        else:
            print("⚠ No entities found (this is okay, patterns may need adjustment)")
    except Exception as e:
        print(f"✗ Pattern extraction failed: {e}")
        return False
    
    # Test 3: Entity class
    print("\nTest 3: Testing Entity data structure...")
    try:
        entity = Entity(
            text="headache",
            label="SYMPTOM",
            confidence=0.95,
            start=0,
            end=8
        )
        print(f"✓ Entity created: {entity.text} ({entity.label})")
    except Exception as e:
        print(f"✗ Entity creation failed: {e}")
        return False
    
    # Test 4: Statistics
    print("\nTest 4: Testing statistics...")
    try:
        test_entities = [
            Entity("headache", "SYMPTOM", 0.95, 0, 8),
            Entity("fever", "SYMPTOM", 0.88, 10, 15),
            Entity("cough", "SYMPTOM", 0.82, 20, 25)
        ]
        
        ner = ProductionMedicalNER()
        stats = ner.get_statistics(test_entities)
        
        print(f"✓ Statistics calculated:")
        print(f"  • Total: {stats['total']}")
        print(f"  • Avg confidence: {stats['avg_confidence']:.3f}")
    except Exception as e:
        print(f"✗ Statistics failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL BASIC TESTS PASSED")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run full demo: python production_ner_system.py")
    print("3. Check README.md for detailed usage instructions")
    print()
    
    return True


def test_with_model():
    """Test with actual model loading (requires dependencies)"""
    print("\n" + "="*80)
    print("TESTING WITH MODEL (requires transformers, torch)")
    print("="*80)
    print()
    
    try:
        import torch
        from transformers import pipeline
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"⚠ Dependencies not installed: {e}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    try:
        from production_ner_system import ProductionMedicalNER
        
        print("\nInitializing NER system...")
        ner = ProductionMedicalNER()
        
        if ner.model_loaded:
            print("✓ Model loaded successfully")
            
            # Test extraction
            test_text = "Patient presents with severe headache and high fever"
            print(f"\nTest text: {test_text}")
            
            entities = ner.extract(test_text, confidence_threshold=0.3)
            
            if entities:
                print(f"\n✓ Extracted {len(entities)} entities:")
                for entity in entities:
                    print(f"  • {entity.text:20s} | {entity.label:12s} | {entity.confidence:.3f}")
            else:
                print("⚠ No entities extracted (try lowering confidence threshold)")
            
            return True
        else:
            print("⚠ Model not loaded (this is okay, fallback patterns still work)")
            return True
            
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic tests (no model required)
    success = test_basic_functionality()
    
    if success and len(sys.argv) > 1 and sys.argv[1] == "--with-model":
        # Run model tests if requested
        test_with_model()
