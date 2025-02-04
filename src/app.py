import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from owl.populate_ontology import populate_ontology
from owl.ontology_manager import load_ontology, run_reasoner
from owl.dataset_generator import extract_features, build_dataset
from predictive_model.predictive_model import train_predictive_model

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, "..", "templates")

app = Flask(__name__, template_folder=template_dir)
app.secret_key = "your_secret_key"  # Necessario per gestire i messaggi flash

ONTOLOGY_PATH = os.path.join("data", "large_ontology.owl")
DATASET_PATH = os.path.join("data", "dataset.csv")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/populate")
def populate():
    try:
        populate_ontology()
        flash("Ontologia popolata con successo!", "success")
    except Exception as e:
        flash(f"Errore nel popolamento: {e}", "danger")
    return redirect(url_for("index"))

@app.route("/extract")
def extract():
    try:
        # Carica l'ontologia e processa l'estrazione
        onto = load_ontology("file://" + os.path.abspath(ONTOLOGY_PATH))
        onto = run_reasoner(onto)
        data = extract_features(onto)
        build_dataset(data, DATASET_PATH)

        df = pd.read_csv(DATASET_PATH)
        dataset_html = df.to_html(classes="table table-striped", index=False)

        flash("Dataset estratto e salvato con successo!", "success")
        return render_template("dataset.html", dataset_html=dataset_html)
    except Exception as e:
        flash(f"Errore nell'estrazione del dataset: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/train")
def train():
    try:
        model, scaler, report = train_predictive_model(DATASET_PATH)
        flash("Modello predittivo addestrato e valutato!", "success")

        return render_template("train.html", report=report)
    except Exception as e:
        flash(f"Errore nell'addestramento del modello: {e}", "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    app.run(debug=True)