import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
from app.agent.logger import logger

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
# Using the local path identified earlier
DATASET_PATH = os.path.join(os.getcwd(), "symbipredict_2022.csv")
HF_TOKEN = "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"

class BioBERTClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BioBERTClassifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        logger.info(f"Initializing BioBERTClassifier with {MODEL_NAME}...")
        
        # 1. Load Data
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset not found at {DATASET_PATH}")
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
            
        self.df = pd.read_csv(DATASET_PATH)
        self.all_symptoms = self.df.columns[:-1].tolist()
        
        # 2. Build Disease Knowledge Base
        self.disease_profiles = {}
        grouped = self.df.groupby('prognosis').max()
        
        self.disease_prototypes = []
        self.disease_names = []
        
        for disease in grouped.index:
            symptoms = grouped.loc[disease]
            active_symptoms = symptoms[symptoms == 1].index.tolist()
            clean_symptoms = [s.replace('_', ' ') for s in active_symptoms]
            profile_text = f"Patient symptoms include {', '.join(clean_symptoms)}."
            
            self.disease_profiles[disease] = active_symptoms
            self.disease_prototypes.append(profile_text)
            self.disease_names.append(disease)
            
        # 3. Load BERT Model
        logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        self.model = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        
        # 4. Pre-compute Embeddings
        logger.info("Pre-computing disease embeddings...")
        self.disease_embeddings = self._get_embeddings(self.disease_prototypes)
        self._initialized = True
        logger.info("BioBERTClassifier initialization complete.")

    def _get_embeddings(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def predict_top_k(self, user_input_text, k=5):
        user_embedding = self._get_embeddings([user_input_text])
        similarities = torch.nn.functional.cosine_similarity(user_embedding, self.disease_embeddings)
        values, indices = torch.topk(similarities, k)
        
        results = []
        for i in range(k):
            idx = indices[i].item()
            score = values[i].item()
            results.append({
                "disease": self.disease_names[idx],
                "confidence": round(score, 3)
            })
        return results

    def get_symptoms_for_disease(self, disease):
        return self.disease_profiles.get(disease, [])

# Global singleton
biobert_classifier = None

def get_biobert_classifier():
    global biobert_classifier
    if biobert_classifier is None:
        biobert_classifier = BioBERTClassifier()
    return biobert_classifier
