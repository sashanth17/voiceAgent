import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
from app.agent.logger import logger

import faiss
import numpy as np
import torch.nn.functional as F

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
# Using the local path identified earlier
DATASET_PATH = os.path.join(os.getcwd(), "symbipredict_2022.csv")
CACHE_DIR = os.path.join(os.getcwd(), "cache")
HF_TOKEN = os.getenv("HF_TOKEN")

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
        
        # 4. Initialize FAISS
        self.disease_index = None
        self._build_or_load_index()
        self._initialized = True
        logger.info("BioBERTClassifier initialization complete.")

    def _get_embeddings(self, text_list):
        """Generates Mean Pooled & Normalized embeddings."""
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        normalized_embeddings = F.normalize(mean_pooled, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

    def _build_or_load_index(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
        index_path = os.path.join(CACHE_DIR, "disease_index_v2.faiss")
        
        if os.path.exists(index_path):
            logger.info("Loading Disease FAISS index...")
            self.disease_index = faiss.read_index(index_path)
        else:
            logger.info("Building Disease FAISS index...")
            embeddings = self._get_embeddings(self.disease_prototypes)
            
            d = embeddings.shape[1]
            self.disease_index = faiss.IndexFlatIP(d)
            self.disease_index.add(embeddings)
            
            faiss.write_index(self.disease_index, index_path)
            logger.info(f"Disease Index built with {self.disease_index.ntotal} vectors.")

    def predict_top_k(self, user_input_text, k=5):
        if not self.disease_index:
             self._build_or_load_index()
             
        user_embedding = self._get_embeddings([user_input_text])
        distances, indices = self.disease_index.search(user_embedding, k)
        
        results = []
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]
            if idx != -1:
                results.append({
                    "disease": self.disease_names[idx],
                    "confidence": float(score)
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
