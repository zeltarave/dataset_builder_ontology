import argparse
import os
from predictive_model.predictive_model import train_predictive_model
from owl.ontology_manager import OntologyManager

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

    # Comando per estrarre le feature testuali dei corsi
    parser_course = subparsers.add_parser("course_features", help="Estrae le feature testuali dai corsi")

    args = parser.parse_args()

    ontology_path = os.path.join("data", "ontology.owl")
    ontology_path = os.path.abspath(ontology_path)
    dataset_path = os.path.join("data", "dataset.csv")

    onto = OntologyManager(ontology_path)

    onto.load()

    if args.command == "populate":
        print("Popolamento dell'ontologia...")
        onto.populate()
        print("Ontologia popolata e salvata in", ontology_path)

    elif args.command == "extract":
        print("Caricamento dell'ontologia...")
        onto.load()
        print("Esecuzione del ragionamento...")
        onto.reason()
        print("Estrazione dei dati...")
        data = onto.extract_person_data()
        onto.build_dataset(data, dataset_path)
        print("Dataset estratto e salvato in", dataset_path)

    elif args.command == "train":
        print("Addestramento del modello predittivo...")
        train_predictive_model(dataset_path)
        print("Modello addestrato e valutato.")

    else:
        parser.print_help()

if __name__ == "__main__":
    cli_main()