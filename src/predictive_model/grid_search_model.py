import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
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

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Crea una pipeline con scaler e regressione logistica
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'))
    ])
    

    param_grid = [
        {
            'clf__C': np.logspace(-5, 5, 11),
            'clf__penalty': ['l1'],
            'clf__solver': ['liblinear','saga']
        },
        {
            'clf__C': np.logspace(-5, 5, 11),
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs']
        }
    ]

    outer_scores = []
    fold = 0


    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        print(f"Fold {fold}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        outer_scores.append(acc)
        print(f"Accuracy per il fold {fold}: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
    
    mean_acc = np.mean(outer_scores)
    print(f"Mean accuracy: {mean_acc:.4f}")
    std_acc = np.std(outer_scores)
    print(f"Standard deviation: {std_acc:.4f}")

def format_result(acc, report):
    result = ""
    result += f"Accuracy: {acc:.4f}\n"
    result += "Classification Report:\n" + report
    return result