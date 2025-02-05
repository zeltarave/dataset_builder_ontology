import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def train_predictive_model(dataset_path, train_ratio=0.7, random_state=42):
    """
    Addestra il modello predittivo base utilizzando il dataset.
    Se esistono file di suddivisione, li utilizza, altrimenti suddivide il dataset.
    Restituisce una tupla (modello, scaler, accuracy, classification report).
    """

    df = pd.read_csv(dataset_path)
    # Preprocessa il dataset: crea la colonna target 'teacher' e la feature 'num_courses_taken'
    df['teacher'] = df['courses_taught'].apply(lambda x: 0 if pd.isna(x) or x.strip() == "" else 1)
    df['num_courses_taken'] = df['courses_taken'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) and x.strip() != "" else 0
    )
    X = df[['age', 'num_courses_taken']]
    y = df['teacher']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_ratio, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    return model, scaler, acc, report


def format_result(acc, report):
    result = ""
    result += f"Accuracy: {acc:.4f}\n"
    result += "Classification Report:\n" + report
    return result