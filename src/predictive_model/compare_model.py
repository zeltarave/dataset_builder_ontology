import pandas as pd
from predictive_model.predictive_model import train_predictive_model , format_result_predictive
from predictive_model.grid_search_model import train_with_grid_search, format_result_grid

def compare_models(dataset_path, random_state=42):
    """
    Addestra due modelli predittivi (base e con GridSearchCV) e confronta le prestazioni.
    """
    result = "Risultati del Confronto dei Modelli:\n\n"
    
    # Addestramento modello base
    model_base, scaler_base, acc_base, report_base = train_predictive_model(dataset_path)
    result_base = format_result_predictive(acc_base, report_base)
    
    # Addestramento modello con GridSearchCV
    model_grid, outer_scores, outer_reports, mean_acc, std_acc = train_with_grid_search(dataset_path)
    result_grid = format_result_grid(outer_scores, outer_reports, mean_acc, std_acc)
    
    result = result_base + "\n\n" + result_grid
    return result
