import argparse
import os
from predictive_model.predictive_model import train_predictive_model, format_result
from predictive_model.grid_search_model import train_with_grid_search, format_result
from predictive_model.compare_model import compare_models
from owl.ontology_manager import OntologyManager
from pykeen_learner.learningKnowledge import pyKeenManager

ontology_path = os.path.join("data", "ontology.owl")
dataset_path = os.path.join("data", "dataset.csv")
onto = OntologyManager(ontology_path, dataset_path)

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

    # Comando per addestrare il modello con GridSearchCV
    parser_grid_search = subparsers.add_parser("grid_search", help="Addestra il modello predittivo utilizzando Grid Search con cross-validation")

    # Comando per confrontare i modelli
    parser_compare = subparsers.add_parser("compare_base_grid", help="Confronta le prestazioni tra il modello base e quello con Grid Search")

    # Comando per addestrare il modello con TransE
    parser_compare = subparsers.add_parser("learn_graph", help="Addestra il modello utilizzando TransE")


    args = parser.parse_args()

    if args.command == "populate":
        onto.populate()

    elif args.command == "extract":
        onto.extract_features()
        onto.build_dataset()

    elif args.command == "train":
        print("Addestramento del modello predittivo...")
        model_base, scaler_base, acc, report_base = train_predictive_model(dataset_path)
        print(f"Accuratezza: {acc}")
        print(report_base)
        print("Modello addestrato e valutato.")

    elif args.command == "grid_search":
        print("Addestramento del modello predittivo con Grid Search...")
        model_grid, acc, report = train_with_grid_search(dataset_path)
        result = format_result(acc, report)
        print(result)
        print("Modello ottimizzato addestrato con successo!")

    elif args.command == "compare_base_grid":
        print("Confronto tra il modello base e il modello grid search...")
        compare_models(dataset_path)
        print("Confronto completato.")

    elif args.command == "learn_graph":
        print("Addestramento del modello su un dataset di triple...")
        pyKeen = pyKeenManager()
        pyKeen = pyKeen.train_model()
        pyKeen.show_graphs()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    cli_main()