import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from predictive_model.predictive_model import train_predictive_model 
from predictive_model.grid_search_model import train_with_grid_search 

def load_and_preprocess_dataset(dataset_path):
    """
    Carica il dataset, crea la variabile target 'teacher' e la feature 'num_courses_taken'.
    Restituisce X (features) e y (target).
    """
    df = pd.read_csv(dataset_path)
    
    df['teacher'] = df['courses_taught'].apply(
        lambda x: 0 if pd.isna(x) or x.strip() == "" else 1
    )
    
    df['num_courses_taken'] = df['courses_taken'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) and x.strip() != "" else 0
    )
    
    df['random_noise'] = df['random_noise'].apply(lambda x: float(x))
    df['random_noise1'] = df['random_noise1'].apply(lambda x: float(x))
    df['random_noise2'] = df['random_noise2'].apply(lambda x: float(x))
    df['random_noise3'] = df['random_noise3'].apply(lambda x: float(x))

    # Feature non lineari
    df['age_squared'] = df['age'] ** 2
    df['age_interaction'] = df['age'] * df['num_courses_taken']

    random_category_dummies = pd.get_dummies(df['random_category'], prefix='cat')

    # Seleziona le feature e il target
    X = df[['age', 'num_courses_taken', 'age_squared', 'age_interaction',
            'random_noise', 'random_noise1', 'random_noise2', 'random_noise3']]
    X = pd.concat([X, random_category_dummies], axis=1)
    y = df['teacher']
    return X, y

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Calcola le metriche di prestazione (accuracy e classification report)
    per il modello sul test set.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    result = (f"--- {model_name} --- \n Accuracy sul test set: {acc:.4f} \n Classification Report:\n {report}")
    return result

def compare_models(dataset_path, test_size=0.7, random_state=42):
    """
    Addestra due modelli predittivi (base e con GridSearchCV) e confronta le prestazioni.
    """
    result = "Risultati del Confronto dei Modelli:\n\n"
    X, y = load_and_preprocess_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Addestramento modello base
    model_base, scaler_base, acc_base, report_base = train_predictive_model(dataset_path)
    X_test_scaled = scaler_base.transform(X_test)
    result_base = evaluate_model(model_base, X_test_scaled, y_test, "Base Model")
    
    # Addestramento modello con GridSearchCV
    model_grid, acc_grid, report_grid = train_with_grid_search(dataset_path)
    result_grid = evaluate_model(model_grid, X_test, y_test, "GridSearch Model")
    
    result = result_base + "\n\n" + result_grid
    return result
