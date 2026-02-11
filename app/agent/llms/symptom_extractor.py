from symptom.production_ner_system import ProductionMedicalNER
from app.agent.logger import logger

_ner_system = None

def get_symptom_extractor():
    global _ner_system
    if _ner_system is None:
        logger.info("Initializing Production Medical NER for Symptom Extraction...")
        try:
            # We use the advanced production system from the symptom directory
            _ner_system = ProductionMedicalNER(
                primary_model="alvaroalon2/biobert_diseases_ner",
                fallback_enabled=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Production NER: {e}")
            return None
    return _ner_system
