import os
from ontology_manager import load_ontology, run_reasoner
from dataset_generator import extract_features, build_dataset
from populate_ontology import populate_ontology

def main():
    ontology_path = os.path.join("data", "large_ontology.owl")
    
    # Popola l'ontologia con dati generati automaticamente
    print("Popolamento dell'ontologia...")
    populate_ontology(ontology_path)

    # Carica l'ontologia
    print("Caricamento dell'ontologia...")
    onto = load_ontology(ontology_path)
    
    # Esegue il ragionamento automatico per inferire nuove conoscenze
    print("Esecuzione del ragionamento...")
    onto = run_reasoner(onto)
    
    # Estrae le caratteristiche per costruire il dataset
    print("Estrazione delle caratteristiche dagli individui...")
    data = extract_features(onto)
    output_csv = os.path.join("data", "dataset.csv")
    build_dataset(data, output_csv)

if __name__ == "__main__":
    main()