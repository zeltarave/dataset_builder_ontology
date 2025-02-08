import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_with_grid_search(dataset_path, test_size=0.7, random_state=42):
    """
    Carica il dataset, estrae le feature e il target, e utilizza GridSearchCV
    con 5-fold cross-validation per ottimizzare i parametri del modello di
    regressione logistica.
    
    Il dataset deve contenere:
      - Una colonna 'age'
      - Una colonna 'courses_taken', contenente i corsi separati da virgola.
      - Una colonna 'courses_taught' che, se non vuota, indica che la persona insegna.
    
    Il target 'teacher' viene creato: 1 se 'courses_taught' è valorizzata, 0 altrimenti.
    
    Args:
        dataset_path (str): percorso del file CSV contenente il dataset.
        random_state (int): seme per la riproducibilità.
    
    Returns:
        best_estimator: il miglior modello addestrato.
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Crea una pipeline con scaler e regressione logistica
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'))
    ])
    
    param_grid = [
        {
            'clf__C': np.logspace(-5, 5, 11),
            'clf__penalty': ['l1'],
            'clf__solver': ['liblinear', 'saga']
        },
        {
            'clf__C': np.logspace(-5, 5, 11),
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs']
        }
    ]
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # Effettua predizioni sul test set e calcola il report
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    return best_model, acc, report

def format_result(acc, report):
    result = ""
    result += f"Accuracy: {acc:.4f}\n"
    result += "Classification Report:\n" + report
    return result