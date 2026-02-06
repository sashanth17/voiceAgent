# Medical Symptom NER Extractor

A production-ready Named Entity Recognition (NER) system for extracting medical symptoms, diseases, and conditions from clinical text using BioBERT models.

## Features

✅ **Multiple Models**: Uses Emily Alsentzer's Bio_ClinicalBERT and specialized medical NER models  
✅ **High Accuracy**: Optimized for medical terminology with confidence scoring  
✅ **Custom Training**: Support for fine-tuning on your own datasets  
✅ **Batch Processing**: Efficient processing of multiple texts  
✅ **Pattern Fallback**: Regex-based extraction for missed entities  
✅ **Production Ready**: Complete with error handling, logging, and API support  
✅ **1000+ Medical Terms**: Comprehensive coverage of symptoms and conditions

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install transformers torch numpy datasets scikit-learn pandas accelerate
```

## Quick Start

### Basic Usage

```python
from symptom_extractor import MedicalSymptomExtractor

# Initialize extractor
extractor = MedicalSymptomExtractor()

# Extract symptoms from text
text = "Patient presents with severe headache, fever of 101°F, and persistent cough."
symptoms = extractor.extract_symptoms(text)

for symptom in symptoms:
    print(f"{symptom['text']} - {symptom['label']} (confidence: {symptom['confidence']:.3f})")
```

### Production System

```python
from production_ner_system import ProductionMedicalNER

# Initialize production system
ner = ProductionMedicalNER()

# Extract entities with context
entities = ner.extract(
    text="Patient has chest pain and shortness of breath",
    confidence_threshold=0.5,
    return_context=True,
    context_window=50
)

for entity in entities:
    print(f"Entity: {entity.text}")
    print(f"Label: {entity.label}")
    print(f"Confidence: {entity.confidence:.3f}")
    print(f"Context: {entity.context}\n")
```

### Advanced NER with Training

```python
from advanced_medical_ner import AdvancedMedicalNER

# Initialize advanced system
ner = AdvancedMedicalNER()
ner.load_model()

# Prepare training data (texts and BIO labels)
train_texts = [
    "Patient has severe headache and fever",
    "Complains of chest pain and shortness of breath"
]

train_labels = [
    ['O', 'O', 'B-SYMPTOM', 'I-SYMPTOM', 'O', 'B-SYMPTOM'],
    ['O', 'O', 'B-SYMPTOM', 'I-SYMPTOM', 'O', 'B-SYMPTOM', 'I-SYMPTOM', 'I-SYMPTOM']
]

# Fine-tune model
ner.train(
    train_texts=train_texts,
    train_labels=train_labels,
    output_dir="./my_custom_model",
    num_epochs=3
)

# Use trained model
predictions = ner.predict("Patient reports migraine and nausea")
```

## Available Models

### 1. Simple Symptom Extractor (`symptom_extractor.py`)
- Easy to use
- Fast inference
- Good for basic symptom extraction
- Built-in fallback with keyword matching

### 2. Advanced NER System (`advanced_medical_ner.py`)
- Custom training support
- BIO tagging scheme
- Multiple entity types (SYMPTOM, DISEASE, MEDICATION, PROCEDURE)
- Evaluation metrics

### 3. Production System (`production_ner_system.py`)
- Most comprehensive
- Pattern-based fallback
- Batch processing
- Statistics and reporting
- JSON import/export
- Context extraction

## Entity Types

The system recognizes the following entity types:

- **SYMPTOM**: Pain, fever, cough, nausea, fatigue, etc.
- **DISEASE**: Diabetes, hypertension, cancer, etc.
- **MEDICATION**: Drug names (in advanced models)
- **PROCEDURE**: Medical procedures (in advanced models)

## Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "Patient has headache and fever",
    "Complains of chest pain",
    "Reports nausea and vomiting"
]

# Method 1: Simple batch
results = extractor.batch_extract(texts)

# Method 2: Production batch with progress
ner = ProductionMedicalNER()
results = ner.batch_extract(texts, show_progress=True)

for i, (text, entities) in enumerate(zip(texts, results)):
    print(f"\nText {i+1}: {text}")
    print(f"Found {len(entities)} entities")
```

## Statistics and Analysis

```python
from production_ner_system import ProductionMedicalNER

ner = ProductionMedicalNER()
entities = ner.extract(text)

# Get statistics
stats = ner.get_statistics(entities)
print(f"Total entities: {stats['total']}")
print(f"Unique entities: {stats['unique']}")
print(f"Average confidence: {stats['avg_confidence']}")
print(f"By label: {stats['by_label']}")

# Get summary
summary = extractor.get_summary(text)
print(summary)
```

## Export and Import

```python
# Export entities to JSON
ner.export_to_json(entities, "extracted_entities.json")

# Load entities from JSON
loaded_entities = ner.load_from_json("extracted_entities.json")
```

## Customization

### Adjust Confidence Threshold

```python
# Higher threshold = more precision, lower recall
high_confidence = ner.extract(text, confidence_threshold=0.8)

# Lower threshold = more recall, lower precision
all_entities = ner.extract(text, confidence_threshold=0.3)
```

### Enable/Disable Fallback

```python
# With fallback (more entities)
ner = ProductionMedicalNER(fallback_enabled=True)

# Without fallback (only model predictions)
ner = ProductionMedicalNER(fallback_enabled=False)
```

### GPU Acceleration

```python
# Use GPU if available (device=0)
ner = ProductionMedicalNER(device=0)

# Force CPU (device=-1)
ner = ProductionMedicalNER(device=-1)
```

## Performance Tips

1. **Use GPU**: 10-50x faster for large batches
2. **Batch Processing**: More efficient than processing individually
3. **Adjust Confidence**: Balance precision vs recall based on use case
4. **Enable Fallback**: Catches entities the model might miss
5. **Cache Model**: Initialize once, use multiple times

## Example Output

```
Text: Patient presents with severe headache, fever of 101°F, and persistent cough.

Extracted Entities:
  • severe headache        | SYMPTOM  | Confidence: 0.945
  • fever                  | SYMPTOM  | Confidence: 0.892
  • persistent cough       | SYMPTOM  | Confidence: 0.878

Statistics:
  • Total entities: 3
  • Unique entities: 3
  • Average confidence: 0.905
  • High confidence (>0.8): 3
```

## Running Demos

```bash
# Simple extractor demo
python symptom_extractor.py

# Advanced NER demo
python advanced_medical_ner.py

# Production system comprehensive demo
python production_ner_system.py
```

## Supported Medical Terms

The system recognizes 1000+ medical terms including:

**Symptoms**: pain, ache, fever, cough, fatigue, nausea, vomiting, diarrhea, headache, dizziness, shortness of breath, chest pain, abdominal pain, etc.

**Diseases**: diabetes, hypertension, cancer, asthma, COPD, arthritis, pneumonia, stroke, etc.

**Conditions**: inflammation, infection, bleeding, swelling, rash, etc.

## Evaluation

```python
from advanced_medical_ner import AdvancedMedicalNER

ner = AdvancedMedicalNER()
ner.load_model()

# Ground truth entities
ground_truth = [
    {'text': 'headache', 'label': 'SYMPTOM'},
    {'text': 'fever', 'label': 'SYMPTOM'}
]

# Predictions
predictions = ner.predict(text)

# Evaluate
metrics = ner.evaluate_predictions(predictions, ground_truth)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

## Troubleshooting

### Model Not Loading
```python
# Check if running on correct device
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Try CPU if GPU fails
ner = ProductionMedicalNER(device=-1)
```

### Low Accuracy
- Lower confidence threshold
- Enable fallback patterns
- Fine-tune on domain-specific data
- Use production system with multiple strategies

### Out of Memory
- Reduce batch size
- Use CPU instead of GPU
- Process texts individually
- Truncate very long texts

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU acceleration)

## License

MIT License

## Citation

If you use Emily Alsentzer's Bio_ClinicalBERT model:

```
@misc{alsentzer2019publicly,
    title={Publicly Available Clinical BERT Embeddings},
    author={Emily Alsentzer and John R. Murphy and Willie Boag and Wei-Hung Weng and Di Jin and Tristan Naumann and Matthew B. A. McDermott},
    year={2019},
    eprint={1904.03323},
    archivePrefix={arXiv}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example code
3. Run the demo scripts to verify setup

## Version

1.0.0 - Production Ready Medical NER System
