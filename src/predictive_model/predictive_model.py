import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_predictive_model(dataset_path, random_state=42):
    """
    Carica il dataset, estrae le feature e il target, addestra un modello predittivo e ne valuta le prestazioni.
    
    In questo esempio:
      - Target: 'teacher' (1 se la persona insegna almeno un corso, 0 altrimenti).
      - Feature: 'age' e il numero di corsi seguiti (derivato dalla colonna 'courses_taken').
    
    Args:
        dataset_path (str): Percorso del file CSV del dataset.
        random_state (int): Semenza per la riproducibilità.
        
    Returns:
        model: Il modello addestrato.
        scaler: Lo scaler usato per normalizzare le feature.
        report: Il report di valutazione (stringa).
    """

    df = pd.read_csv(dataset_path)
    
    # Creiamo la variabile target: 1 se la persona insegna almeno un corso, 0 altrimenti
    df['teacher'] = df['courses_taught'].apply(lambda x: 0 if pd.isna(x) or x.strip() == "" else 1)
    
    # Creiamo una feature aggiuntiva: il numero di corsi seguiti.
    df['num_courses_taken'] = df['courses_taken'].apply(lambda x: len(x.split(',')) if pd.notna(x) and x.strip() != "" else 0)
    
    # Selezioniamo le feature e il target
    X = df[['age', 'num_courses_taken']]
    y = df['teacher']
    
    # Suddividiamo il dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Addestriamo un modello di regressione logistica
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Effettuiamo le predizioni sul test set
    y_pred = model.predict(X_test_scaled)
    
    # Valutiamo le prestazioni del modello
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    return model, scaler, report

if __name__ == "__main__":
    dataset_path = os.path.join("data", "dataset.csv")
    train_predictive_model(dataset_path)