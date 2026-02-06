"""
Advanced Medical NER System with Custom Training
Supports fine-tuning on custom datasets for improved accuracy
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from typing import List, Dict, Optional
import json
from collections import defaultdict


class AdvancedMedicalNER:
    """
    Advanced medical NER system with training capabilities
    """
    
    # BIO tagging scheme for medical entities
    LABEL_MAP = {
        'O': 0,          # Outside
        'B-SYMPTOM': 1,  # Beginning of symptom
        'I-SYMPTOM': 2,  # Inside symptom
        'B-DISEASE': 3,  # Beginning of disease
        'I-DISEASE': 4,  # Inside disease
        'B-MEDICATION': 5,  # Beginning of medication
        'I-MEDICATION': 6,  # Inside medication
        'B-PROCEDURE': 7,   # Beginning of procedure
        'I-PROCEDURE': 8,   # Inside procedure
    }
    
    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        """Initialize the NER system"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_path: Optional[str] = None):
        """Load pre-trained or fine-tuned model"""
        if model_path:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(self.LABEL_MAP),
                id2label=self.ID2LABEL,
                label2id=self.LABEL_MAP
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.LABEL_MAP),
                id2label=self.ID2LABEL,
                label2id=self.LABEL_MAP
            )
        
        self.model.to(self.device)
        print(f"✓ Model loaded on {self.device}")
    
    def prepare_training_data(self, texts: List[str], labels: List[List[str]]) -> Dataset:
        """
        Prepare data for training
        
        Args:
            texts: List of input texts
            labels: List of label sequences (BIO format)
            
        Returns:
            HuggingFace Dataset
        """
        # Tokenize and align labels
        tokenized_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        aligned_labels = []
        for i, label_seq in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(self.LABEL_MAP.get(label_seq[word_idx], 0))
                else:
                    # For subword tokens, use same label or -100
                    label_ids.append(self.LABEL_MAP.get(label_seq[word_idx], 0))
                
                previous_word_idx = word_idx
            
            aligned_labels.append(label_ids)
        
        # Create dataset
        dataset_dict = {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': torch.tensor(aligned_labels)
        }
        
        return Dataset.from_dict({k: v.tolist() for k, v in dataset_dict.items()})
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[List[str]],
        output_dir: str = "./medical_ner_model",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune the model on custom data
        
        Args:
            train_texts: Training texts
            train_labels: Training labels in BIO format
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        if self.model is None:
            self.load_model()
        
        # Prepare dataset
        train_dataset = self.prepare_training_data(train_texts, train_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no"
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✓ Model saved to {output_dir}")
    
    def predict(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Predict entities in text
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence
            
        Returns:
            List of extracted entities
        """
        if self.model is None:
            self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for idx, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
            if offset[0] == offset[1]:  # Skip special tokens
                continue
            
            label = self.ID2LABEL[pred_id.item()]
            confidence = probabilities[idx][pred_id].item()
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label.split('-')[1]
                current_entity = {
                    'text': text[offset[0]:offset[1]],
                    'label': entity_type,
                    'confidence': confidence,
                    'start': offset[0].item(),
                    'end': offset[1].item()
                }
            
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                current_entity['end'] = offset[1].item()
                current_entity['text'] = text[current_entity['start']:current_entity['end']]
                current_entity['confidence'] = max(current_entity['confidence'], confidence)
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        # Filter by confidence
        return [e for e in entities if e['confidence'] >= confidence_threshold]
    
    def batch_predict(self, texts: List[str], confidence_threshold: float = 0.5) -> List[List[Dict]]:
        """Batch prediction"""
        return [self.predict(text, confidence_threshold) for text in texts]
    
    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Returns:
            Evaluation metrics
        """
        pred_entities = set((e['text'].lower(), e['label']) for e in predictions)
        true_entities = set((e['text'].lower(), e['label']) for e in ground_truth)
        
        tp = len(pred_entities & true_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }


# Example training data generator
def generate_sample_training_data() -> tuple:
    """Generate sample training data for demonstration"""
    
    texts = [
        "Patient has severe headache and fever",
        "Complains of chest pain and shortness of breath",
        "Experiencing nausea and abdominal pain",
        "Reports chronic back pain and fatigue",
        "Diagnosed with hypertension and diabetes",
    ]
    
    # BIO format labels (one per word)
    labels = [
        ['O', 'O', 'B-SYMPTOM', 'I-SYMPTOM', 'O', 'B-SYMPTOM'],
        ['O', 'O', 'B-SYMPTOM', 'I-SYMPTOM', 'O', 'B-SYMPTOM', 'I-SYMPTOM', 'I-SYMPTOM'],
        ['O', 'B-SYMPTOM', 'O', 'B-SYMPTOM', 'I-SYMPTOM'],
        ['O', 'O', 'B-SYMPTOM', 'I-SYMPTOM', 'O', 'B-SYMPTOM'],
        ['O', 'O', 'B-DISEASE', 'O', 'B-DISEASE'],
    ]
    
    return texts, labels


def demo_advanced_ner():
    """Demonstrate the advanced NER system"""
    
    print("\n" + "="*80)
    print("ADVANCED MEDICAL NER SYSTEM")
    print("="*80 + "\n")
    
    # Initialize
    ner = AdvancedMedicalNER()
    ner.load_model()
    
    # Test texts
    test_texts = [
        "Patient presents with severe migraine, high fever, and persistent cough.",
        "Chief complaint: acute chest pain radiating to left arm with dyspnea.",
        "History of chronic lower back pain, arthritis, and occasional vertigo.",
        "Symptoms: nausea, vomiting, diarrhea, and severe abdominal cramping.",
        "Patient reports palpitations, anxiety, and irregular heartbeat.",
    ]
    
    print("Predictions on test data:\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"--- Text {i} ---")
        print(f"Input: {text}")
        
        entities = ner.predict(text, confidence_threshold=0.3)
        
        if entities:
            print(f"Found {len(entities)} entities:")
            for entity in entities:
                print(f"  • {entity['text']:25s} | {entity['label']:12s} | {entity['confidence']:.3f}")
        else:
            print("  No entities found.")
        print()
    
    # Batch prediction
    print("\n" + "="*80)
    print("BATCH PREDICTION")
    print("="*80 + "\n")
    
    batch_results = ner.batch_predict(test_texts[:3])
    for i, (text, results) in enumerate(zip(test_texts[:3], batch_results), 1):
        print(f"{i}. Found {len(results)} entities in: '{text[:50]}...'")


if __name__ == "__main__":
    demo_advanced_ner()
