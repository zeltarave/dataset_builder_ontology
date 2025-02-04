from owlready2 import get_ontology, sync_reasoner

def load_ontology(path: str):
    """
    Carica l'ontologia da un file dato il percorso.
    """
    onto = get_ontology(path).load()
    return onto

def run_reasoner(ontology):
    """
    Esegue il ragionamento automatico sull'ontologia per inferire nuove conoscenze.
    """
    with ontology:
        sync_reasoner()  # Utilizza il ragionatore incorporato di Owlready2
    return ontology