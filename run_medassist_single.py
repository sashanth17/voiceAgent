
import os
import sys
import re
import pickle
import pandas as pd
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ==========================================
# CONFIGURATION
# ==========================================
# Paths
# Assuming this script is run from the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "symbipredict_2022.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Model
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
HF_TOKEN = "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"

# Pipeline Settings
MAX_QUESTIONS = 7
TOP_K_DISEASES = 5
INITIAL_CONFIDENCE_THRESHOLD = 0.02 

# Bayesian Parameters
PROB_SYMPTOM_GIVEN_DISEASE = 0.9   
PROB_SYMPTOM_ABSENT_GIVEN_DISEASE = 0.01 
PROB_NO_SYMPTOM_Given_DISEASE = 0.05 
PROB_NO_SYMPTOM_ABSENT_GIVEN_DISEASE = 0.99 

# Safety
EMERGENCY_SYMPTOMS = [
    "chest pain",
    "loss of consciousness",
    "shortness of breath",
    "severe bleeding",
    "difficulty breathing",
    "confusion",
    "sudden severe headache"
]

# ==========================================
# UTILS
# ==========================================
class PipelineLogger:
    @staticmethod
    def log(step, message, data=None):
        print(f"\n[{step}] {message}")
        if data:
            if isinstance(data, list):
                for item in data:
                    print(f"  - {item}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    print(f"  - {k}: {v}")
            else:
                print(f"  {data}")

    @staticmethod
    def error(message):
        print(f"\n[ERROR] {message}", file=sys.stderr)

    @staticmethod
    def warning(message):
        print(f"\n[WARNING] {message}")

# ==========================================
# MODEL: BioBERT
# ==========================================
class BioBERTEncoder:
    def __init__(self):
        PipelineLogger.log("Model", f"Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        self.model = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, text):
        """Generates Mean Pooled & Normalized embedding for a single text."""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        normalized_embedding = F.normalize(mean_pooled, p=2, dim=1)
        return normalized_embedding.cpu().numpy()

    def get_embeddings_batch(self, text_list, batch_size=32):
        all_embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            normalized = F.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu())
            
        return torch.cat(all_embeddings).numpy()

# ==========================================
# DATA: Knowledge Base
# ==========================================
class KnowledgeBase:
    def __init__(self, encoder, cache_dir=CACHE_DIR):
        self.encoder = encoder
        if not os.path.exists(DATASET_PATH):
             PipelineLogger.error(f"Dataset not found at {DATASET_PATH}")
             raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
             
        self.df = pd.read_csv(DATASET_PATH)
        self.all_possible_symptoms = self.df.columns[:-1].tolist()
        
        # Build Disease Profiles
        self.disease_profiles = {}
        grouped = self.df.groupby('prognosis').max()
        
        self.disease_names = []
        self.disease_texts = []
        
        for disease in grouped.index:
            symptoms = grouped.loc[disease]
            active_symptoms = symptoms[symptoms == 1].index.tolist()
            self.disease_profiles[disease] = set(active_symptoms)

        # Build Mandatory Symptoms
        mean_symptoms = self.df.groupby('prognosis').mean()
        self.mandatory_symptoms = {}
        
        for disease in mean_symptoms.index:
            symptoms = mean_symptoms.loc[disease]
            mandatory = symptoms[symptoms > 0.9].index.tolist()
            self.mandatory_symptoms[disease] = set(mandatory)
            
            clean_symptoms = [s.replace('_', ' ') for s in active_symptoms]
            text = f"Patient has {', '.join(clean_symptoms)}."
            
            self.disease_names.append(disease)
            self.disease_texts.append(text)
            
        # Initialize FAISS Indices
        self.disease_index = None 
        self.rag_index = None     
        self.rag_metadata = []    
        
        self._build_or_load_disease_index(cache_dir)
        self._load_rag_index(cache_dir)
        
    def _build_or_load_disease_index(self, cache_dir):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        index_path = os.path.join(cache_dir, "disease_index.faiss")
        
        if os.path.exists(index_path):
            PipelineLogger.log("DB", "Loading Disease FAISS index...")
            self.disease_index = faiss.read_index(index_path)
        else:
            PipelineLogger.log("DB", "Building Disease FAISS index...")
            embeddings = self.encoder.get_embeddings_batch(self.disease_texts)
            
            d = embeddings.shape[1]
            self.disease_index = faiss.IndexFlatIP(d)
            self.disease_index.add(embeddings)
            
            faiss.write_index(self.disease_index, index_path)
            PipelineLogger.log("DB", f"Disease Index built with {self.disease_index.ntotal} vectors.")

    def _load_rag_index(self, cache_dir):
        rag_path = os.path.join(cache_dir, "medical_docs.faiss")
        meta_path = os.path.join(cache_dir, "medical_docs_meta.pkl")
        
        if os.path.exists(rag_path) and os.path.exists(meta_path):
            try:
                self.rag_index = faiss.read_index(rag_path)
                with open(meta_path, 'rb') as f:
                    self.rag_metadata = pickle.load(f)
                PipelineLogger.log("DB", f"Loaded RAG Index with {self.rag_index.ntotal} docs.")
            except Exception as e:
                PipelineLogger.error(f"Failed to load RAG index: {e}")
        else:
            PipelineLogger.warning("No RAG index found. Run ingestion script first if needed.")

    def search_diseases(self, query_text, k=10):
        if not self.disease_index: return []
        query_vec = self.encoder.get_embedding([query_text]) 
        distances, indices = self.disease_index.search(query_vec, k)
        results = []
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]
            if idx != -1: 
                results.append((self.disease_names[idx], float(score)))
        return results

    def search_docs(self, query_text, k=3):
        if not self.rag_index: return []
        query_vec = self.encoder.get_embedding([query_text])
        distances, indices = self.rag_index.search(query_vec, k)
        results = []
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]
            if idx != -1 and idx < len(self.rag_metadata):
                doc = self.rag_metadata[idx]
                result_item = doc.copy()
                result_item["score"] = float(score)
                results.append(result_item)
        return results

    def get_symptoms_for_disease(self, disease_name):
        return self.disease_profiles.get(disease_name, set())

    def get_all_symptoms(self):
        return self.all_possible_symptoms

    def get_mandatory_symptoms(self, disease_name):
        return self.mandatory_symptoms.get(disease_name, set())

# ==========================================
# AGENTS
# ==========================================

class SymptomExtractionAgent:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.all_symptoms = self.kb.get_all_symptoms()
        
    def extract(self, text):
        found = []
        text = text.lower().replace('.', ' ').replace(',', ' ')
        sorted_syms = sorted(self.all_symptoms, key=len, reverse=True)
        
        def normalize_key(s):
            return s.replace('_', ' ').replace(' the ', ' ').replace(' in ', ' ').replace(' of ', ' ')
            
        text_norm = normalize_key(text)
        
        for sym in sorted_syms:
            readable = sym.replace('_', ' ')
            readable_norm = normalize_key(readable)
            if readable in text or readable_norm in text_norm:
                found.append(sym)
                
        return list(set(found))

class SafetyAgent:
    def calculate_risk_score(self, confirmed_symptoms, duration_text):
        score = 10 
        triggers = []
        for sym in confirmed_symptoms:
            norm_sym = sym.replace('_', ' ').lower()
            if norm_sym in EMERGENCY_SYMPTOMS:
                score += 70
                triggers.append(norm_sym)
        score += len(confirmed_symptoms) * 5
        duration_lower = duration_text.lower()
        is_acute = any(x in duration_lower for x in ['hour', 'minute', 'sudden', 'yesterday', 'today', 'just now'])
        if 'day' in duration_lower:
            match = re.search(r'(\d+)\s*day', duration_lower)
            if match and int(match.group(1)) <= 2:
                is_acute = True
        is_pain = any('pain' in s for s in confirmed_symptoms)
        if is_acute and is_pain:
            score += 20
        elif 'week' in duration_lower or 'month' in duration_lower or 'year' in duration_lower:
             score += 5
        return min(100, score)

    def check_for_emergency(self, text, extracted_symptoms=None):
        text_lower = text.lower()
        triggers = []
        for sym in EMERGENCY_SYMPTOMS:
            norm_sym = sym.replace('_', ' ').lower()
            if norm_sym in text_lower:
                triggers.append(norm_sym)
        if extracted_symptoms:
            for sym in extracted_symptoms:
                norm_sym = sym.replace('_', ' ').lower()
                if norm_sym in EMERGENCY_SYMPTOMS:
                    triggers.append(norm_sym)
        if triggers:
            msg = f"[!] URGENT: Symptoms indicating medical emergency detected ({', '.join(triggers)}). Please seek immediate medical attention."
            PipelineLogger.log("Safety", msg)
            return True, msg
        return False, ""

class QuestionPlanningAgent:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def select_next_question(self, current_hypotheses, asked_symptoms):
        if not current_hypotheses:
            return None
        top_diseases = [d for d, p in current_hypotheses if p > 0.01] 
        candidate_symptoms = set()
        for d in top_diseases:
            candidate_symptoms.update(self.kb.get_symptoms_for_disease(d))
        candidate_symptoms = candidate_symptoms - asked_symptoms
        if not candidate_symptoms:
            return None
            
        scores = []
        for sym in candidate_symptoms:
            diff_score = 0
            for i in range(len(top_diseases)):
                d1 = top_diseases[i]
                p1 = next(p for d, p in current_hypotheses if d == d1)
                has_s1 = sym in self.kb.get_symptoms_for_disease(d1)
                for j in range(i + 1, len(top_diseases)):
                    d2 = top_diseases[j]
                    p2 = next(p for d, p in current_hypotheses if d == d2)
                    has_s2 = sym in self.kb.get_symptoms_for_disease(d2)
                    if has_s1 != has_s2:
                        diff_score += (p1 + p2)
            scores.append((diff_score, sym))
        scores.sort(key=lambda x: x[0], reverse=True)
        if scores and scores[0][0] > 0:
            best = scores[0][1]
            PipelineLogger.log("Planner", f"Selected {best} (Diff Score: {scores[0][0]:.3f})")
            return best
        return list(candidate_symptoms)[0]

class DiagnosisAgent:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.disease_probs = {} 
        
    def initialize_priors(self, semantic_search_results):
        valid_scores = [(d, max(0.001, s)) for d, s in semantic_search_results]
        total_score = sum(s for d, s in valid_scores)
        if total_score == 0: total_score = 1
        self.disease_probs = {d: s/total_score for d, s in valid_scores}
        
    def update_beliefs(self, symptom, confidence, weight=1.0, symptoms_state=None):
        PipelineLogger.log("Diagnosis", f"Updating beliefs for {symptom} (Conf: {confidence:.2f}, W: {weight})")
        new_probs = {}
        total_prob = 0
        for disease, p_d in self.disease_probs.items():
            disease_symptoms = self.kb.get_symptoms_for_disease(disease)
            has_symptom_in_kb = symptom in disease_symptoms
            if has_symptom_in_kb:
                l_yes = PROB_SYMPTOM_GIVEN_DISEASE
                l_no = PROB_NO_SYMPTOM_Given_DISEASE
            else:
                l_yes = PROB_SYMPTOM_ABSENT_GIVEN_DISEASE 
                l_no = PROB_NO_SYMPTOM_ABSENT_GIVEN_DISEASE
            
            base_likelihood = (l_yes * confidence) + (l_no * (1.0 - confidence))
            weighted_likelihood = base_likelihood ** weight
            
            mandatory_symptoms = self.kb.get_mandatory_symptoms(disease)
            if symptom in mandatory_symptoms and confidence < 0.2:
                PipelineLogger.log("Diagnosis", f"Necessity Penalty: {disease} requires {symptom} (Absent)")
                weighted_likelihood *= 0.1
            
            updated_p = weighted_likelihood * p_d
            new_probs[disease] = updated_p
            total_prob += updated_p
            
        if total_prob > 0:
            for d in new_probs:
                new_probs[d] /= total_prob
        self.disease_probs = new_probs
        top_3 = sorted(self.disease_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        PipelineLogger.log("Diagnosis", "Top Hypotheses Update:", {d: round(p, 3) for d, p in top_3})

    def get_top_hypotheses(self, k=5):
        sorted_d = sorted(self.disease_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_d[:k]

    def get_rag_context(self, user_query, top_disease=None):
        results = []
        symptom_matches = self.kb.search_docs(user_query, k=3)
        for m in symptom_matches:
            m['retrieval_type'] = 'symptom_match'
            results.append(m)
        if top_disease:
            disease_matches = self.kb.search_docs(top_disease, k=2)
            for m in disease_matches:
                if not any(r['text'] == m['text'] for r in results):
                    m['retrieval_type'] = 'disease_info'
                    results.append(m)
        return results

class ReportGeneratorAgent:
    def generate_report(self, hypotheses, symptoms_state, patient_info=None, risk_score=0, duration="Unknown", rag_context=None, knowledge_base=None):
        top_disease = hypotheses[0] if hypotheses else ("Unknown", 0.0)
        report = []
        report.append("\n=== FINAL DIAGNOSIS REPORT ===")
        if patient_info: report.append(f"Patient Info: {patient_info}")
        report.append(f"Duration: {duration}")
        
        risk_label = "LOW"
        if risk_score > 70: risk_label = "HIGH"
        elif risk_score > 40: risk_label = "MEDIUM"
        report.append(f"Risk Score: {risk_score}/100 ({risk_label})")
        report.append(f"\n[+] Top Prediction: {top_disease[0].upper()}")
        report.append(f"   Confidence: {top_disease[1]*100:.1f}%")
        
        if knowledge_base and top_disease[0] != "Unknown":
            disease_symptoms = knowledge_base.get_symptoms_for_disease(top_disease[0])
            confirmed_symptoms = [s for s, v in symptoms_state.items() if v]
            supporting_evidence = []
            for s in confirmed_symptoms:
                if s in disease_symptoms:
                    supporting_evidence.append(s.replace('_', ' '))
            if supporting_evidence:
                report.append(f"   Supported by: {', '.join(supporting_evidence)}")
            else:
                report.append(f"   Supported by: Clinical text similarity (no direct symptom overlap in KB)")
        
        report.append("\n[?] Differential Diagnosis:")
        for d, p in hypotheses[1:4]:
            if p > 0.01:
                explanation = ""
                if knowledge_base:
                    ds = knowledge_base.get_symptoms_for_disease(d)
                    match_count = sum(1 for s in symptoms_state if symptoms_state[s] and s in ds)
                    explanation = f" ({match_count} symptom matches)"
                report.append(f"   - {d}: {p*100:.1f}%{explanation}")
                
        report.append("\n[v] Validated Symptoms:")
        present = [s.replace('_', ' ') for s, v in symptoms_state.items() if v]
        absent = [s.replace('_', ' ') for s, v in symptoms_state.items() if not v]
        report.append(f"   Present: {', '.join(present)}")
        report.append(f"   Absent: {', '.join(absent)}")
        
        if rag_context:
            report.append("\n[i] Clinical Context & Knowledge:")
            disease_infos = [r for r in rag_context if r.get('retrieval_type') == 'disease_info']
            for info in disease_infos:
                text = info['text'].replace('\n', ' ').strip()
                if len(text) > 300: text = text[:300] + "..."
                report.append(f"   > About Prediction: {text}")
            symptom_infos = [r for r in rag_context if r.get('retrieval_type') == 'symptom_match']
            seen_texts = set(d['text'] for d in disease_infos)
            count = 0
            for info in symptom_infos:
                if info['text'] not in seen_texts and count < 2:
                    text = info['text'].replace('\n', ' ').strip()
                    if len(text) > 200: text = text[:200] + "..."
                    report.append(f"   > Related: {text}")
                    seen_texts.add(info['text'])
                    count += 1
        return "\n".join(report)

# ==========================================
# MAIN PIPELINE
# ==========================================
class ClinicalPipeline:
    def __init__(self):
        self.encoder = BioBERTEncoder()
        self.kb = KnowledgeBase(self.encoder)
        self.safety_agent = SafetyAgent()
        self.extractor = SymptomExtractionAgent(self.kb)
        self.diagnosis_agent = DiagnosisAgent(self.kb)
        self.planner = QuestionPlanningAgent(self.kb)
        self.reporter = ReportGeneratorAgent()
        
        self.state = {
            "symptoms": {}, 
            "demographics": "",
            "messages": []
        }

    def run(self):
        print("\n=== MedAssist AI v2.0 (Single-File) ===")
        print("Initializing agents...\n")
        
        age = input("Patient Age: ").strip()
        gender = input("Patient Gender: ").strip()
        demographics = f"{age}-year-old {gender}"
        self.state["demographics"] = demographics
        
        user_input = input("\nDescribe your symptoms (e.g. 'I have chest pain'): ").strip()
        duration = input("How long have you had these symptoms? (e.g. '2 days', '1 week'): ").strip()
        self.state["duration"] = duration
        
        full_query = f"{demographics} patient with {user_input} for {duration}"
        
        is_emergency, msg = self.safety_agent.check_for_emergency(user_input)
        if is_emergency:
            print(f"\n{msg}\n")
            cont = input("Do you want to continue diagnosis? (y/n): ")
            if cont.lower() != 'y': return

        initial_symptoms = self.extractor.extract(user_input)
        for s in initial_symptoms:
            self.state["symptoms"][s] = True
            
        PipelineLogger.log("Pipeline", "Running semantic search/FAISS...")
        search_results = self.kb.search_diseases(full_query, k=25) 
        
        rag_hits = self.kb.search_docs(full_query, k=5)
        rag_found_diseases = []
        for hit in rag_hits:
            match = re.search(r"Disease:\s*([^.]+)", hit.get('text', ''))
            if match:
                rag_found_diseases.append(match.group(1).strip())
                
        existing_diseases = [d for d, s in search_results]
        all_kb_diseases = set(self.kb.disease_profiles.keys())
        
        for d in rag_found_diseases:
            valid_match = None
            if d in all_kb_diseases:
                valid_match = d
            else:
                for kb_d in all_kb_diseases:
                    if kb_d.lower() == d.lower():
                        valid_match = kb_d
                        break
            
            if valid_match and valid_match not in existing_diseases:
                PipelineLogger.log("Pipeline", f"RAG found relevant disease not in top-25: {valid_match}. Adding to priors.")
                search_results.append((valid_match, 0.5))
        
        self.diagnosis_agent.initialize_priors(search_results)
        for s in initial_symptoms:
            self.diagnosis_agent.update_beliefs(s, 1.0, weight=2.0)

        for q_idx in range(MAX_QUESTIONS):
            hypotheses = self.diagnosis_agent.get_top_hypotheses(k=TOP_K_DISEASES)
            
            top_prob = hypotheses[0][1]
            if len(hypotheses) > 1:
                second_prob = hypotheses[1][1]
                gap = top_prob - second_prob
            else:
                gap = 1.0

            if top_prob > 0.85 or (top_prob > 0.6 and gap > 0.3):
                PipelineLogger.log("Pipeline", f"Stopping early: Top confidence {top_prob:.2f} with gap {gap:.2f}")
                break
                
            next_symptom = self.planner.select_next_question(hypotheses, self.state["symptoms"].keys())
            if not next_symptom:
                PipelineLogger.log("Pipeline", "No useful questions left.")
                break
                
            readable_symptom = next_symptom.replace('_', ' ')
            print(f"\n[Question {q_idx+1}] Do you have {readable_symptom}?")
            print("   (Options: yes, no, sometimes, rarely, mostly)")
            ans = input("   > ").strip().lower()
            
            confidence = 0.0
            if ans in ['yes', 'y', 'sure', 'always']: confidence = 1.0
            elif ans in ['no', 'n', 'never']: confidence = 0.0
            elif ans in ['sometimes', 'maybe', 'possibly']: confidence = 0.5
            elif ans in ['rarely', 'seldom']: confidence = 0.25
            elif ans in ['mostly', 'often']: confidence = 0.75
            else:
                print("   [Assuming 'no']")
                confidence = 0.0
            
            self.state["symptoms"][next_symptom] = (confidence > 0.0)
            self.diagnosis_agent.update_beliefs(next_symptom, confidence)
            
            if confidence > 0.5:
                is_emerg, msg = self.safety_agent.check_for_emergency("", [next_symptom])
                if is_emerg: print(msg)

        confirmed_list = [s for s, v in self.state["symptoms"].items() if v]
        risk_score = self.safety_agent.calculate_risk_score(confirmed_list, self.state["duration"])
        final_hypotheses = self.diagnosis_agent.get_top_hypotheses(k=5)
        top_disease_name = final_hypotheses[0][0] if final_hypotheses else None
        rag_context = self.diagnosis_agent.get_rag_context(user_input, top_disease_name)
        
        report = self.reporter.generate_report(
            final_hypotheses, 
            self.state["symptoms"], 
            demographics, 
            risk_score, 
            self.state["duration"], 
            rag_context,
            knowledge_base=self.kb
        )
        print(report)

if __name__ == "__main__":
    pipeline = ClinicalPipeline()
    pipeline.run()
