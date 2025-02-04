from owlready2 import get_ontology, sync_reasoner
from logger_config import setup_logger

logger = setup_logger("ontology_manager", "log/ontology_manager.log")

def load_ontology(path: str):
    """
    Carica l'ontologia da un file dato il percorso.
    Gestisce e logga eventuali errori.
    """
    try:
        logger.info(f"Caricamento dell'ontologia da {path}...")
        onto = get_ontology(path).load()
        logger.info("Ontologia caricata correttamente.")
        return onto
    except Exception as e:
        logger.error(f"Errore durante il caricamento dell'ontologia: {e}", exc_info=True)
        raise

def run_reasoner(ontology):
    """
    Esegue il ragionamento automatico sull'ontologia per inferire nuove conoscenze.
    """
    try:
        logger.info("Esecuzione del ragionamento...")
        with ontology:
            sync_reasoner()  # Utilizzare il ragionatore incorporato di Owlready2
        logger.info("Ragionamento completato con successo.")
        return ontology
    except Exception as e:
        logger.error(f"Errore durante il ragionamento: {e}", exc_info=True)
        raise