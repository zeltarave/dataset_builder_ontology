import os
import sys
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from decorators import error_handler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from owl.ontology_manager import OntologyManager
from predictive_model.predictive_model import train_predictive_model, format_result
from predictive_model.grid_search_model import train_with_grid_search, format_result
from predictive_model.compare_model import compare_models

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Necessario per gestire i messaggi flash

ONTOLOGY_PATH = os.path.join("data", "ontology.owl")
ONTOLOGY_PATH = os.path.abspath(ONTOLOGY_PATH)
DATASET_PATH = os.path.join("data", "dataset.csv")

onto = OntologyManager(ONTOLOGY_PATH)

onto.load()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/populate")
@error_handler("Errore nel popolamento dell'ontologia")
def populate():
    onto.populate()
    flash("Ontologia popolata con successo!", "success")
    return redirect(url_for("index"))

@app.route("/extract")
@error_handler("Errore nell'estrazione del dataset")
def extract():
    onto.reason()
    data = onto.extract_person_data()
    onto.build_dataset(data, DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    dataset_html = df.to_html(classes="table table-striped", index=False)

    flash("Dataset estratto e salvato con successo!", "success")
    return render_template("dataset.html", dataset_html=dataset_html)

@app.route("/train")
@error_handler("Errore nell'addestramento del modello predittivo")
def train():
    model_base, scaler_base, acc, report_base = train_predictive_model(DATASET_PATH)
    results = format_result(acc, report_base)
    flash("Modello predittivo addestrato e valutato!", "success")

    return render_template("train.html", report=results)

@app.route("/grid_search")
@error_handler("Errore nell'addestramento del modello con Grid Search")
def grid_search():
    model_grid, acc, report_grid = train_with_grid_search(DATASET_PATH)
    results = format_result(acc, report_grid)
    flash("Modello addestrato con Grid Search!", "success")
    return render_template("grid_search.html", report=results)

@app.route("/compare")
@error_handler("Errore nel confronto dei modelli")
def compare():
    results = compare_models(DATASET_PATH)
    flash("Confronto dei modelli completato!", "success")
    return render_template("compare.html", report=results)

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    app.run(debug=True)