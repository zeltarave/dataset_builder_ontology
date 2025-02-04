import argparse
import os
from owl.populate_ontology import populate_ontology
from owl.ontology_manager import load_ontology, run_reasoner
from owl.dataset_generator import extract_features, build_dataset
from predictive_model.predictive_model import train_predictive_model

def cli_main():
    parser = argparse.ArgumentParser(
        description="Dataset Builder Ontology - Interfaccia a riga di comando"
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandi disponibili")

    # Popolare l'ontologia
    parser_populate = subparsers.add_parser("populate", help="Popola l'ontologia con dati generati")
    
    # Comando per estrarre il dataset
    parser_extract = subparsers.add_parser("extract", help="Estrae il dataset dall'ontologia")
    
    # Comando per addestrare il modello predittivo
    parser_train = subparsers.add_parser("train", help="Addestra il modello predittivo sul dataset")

    args = parser.parse_args()

    ontology_path = os.path.join("data", "large_ontology.owl")
    dataset_path = os.path.join("data", "dataset.csv")

    if args.command == "populate":
        print("Popolamento dell'ontologia...")
        populate_ontology()
        print("Ontologia popolata e salvata in", ontology_path)

    elif args.command == "extract":
        print("Caricamento dell'ontologia...")
        onto = load_ontology("file://" + os.path.abspath(ontology_path))
        print("Esecuzione del ragionamento...")
        onto = run_reasoner(onto)
        print("Estrazione dei dati...")
        data = extract_features(onto)
        build_dataset(data, dataset_path)
        print("Dataset estratto e salvato in", dataset_path)

    elif args.command == "train":
        print("Addestramento del modello predittivo...")
        train_predictive_model(dataset_path)
        print("Modello addestrato e valutato.")

    else:
        parser.print_help()

if __name__ == "__main__":
    cli_main()