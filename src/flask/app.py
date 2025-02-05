import os
import sys
from flask import Flask, render_template, redirect, url_for, flash, request
from decorators import error_handler
from flask_wtf.csrf import CSRFProtect
from forms import DataSplitForm
import pandas as pd
from sklearn.model_selection import train_test_split

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
csrf = CSRFProtect(app)

ONTOLOGY_PATH = os.path.join("data", "ontology.owl")
ONTOLOGY_PATH = os.path.abspath(ONTOLOGY_PATH)
DATASET_PATH = os.path.join("data", "dataset.csv")

onto = OntologyManager(ONTOLOGY_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/populate")
@error_handler("Errore nel popolamento dell'ontologia")
def populate():
    onto.load()
    onto.populate()
    flash("Ontologia popolata con successo!", "success")
    return redirect(url_for("index"))

@app.route("/extract")
@error_handler("Errore nell'estrazione del dataset")
def extract():
    onto.load()
    onto.reason()
    data = onto.extract_features()
    onto.build_dataset(data, DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    dataset_html = df.to_html(classes="table table-striped", index=False)

    flash("Dataset estratto e salvato con successo!", "success")
    return render_template("dataset.html", dataset_html=dataset_html)

@app.route("/train", methods = ["GET", "POST"])
@error_handler("Errore nell'addestramento del modello predittivo")
def train():
    # Inserisci il form per permettere all'utente di specificare la proporzione di training set
    form = DataSplitForm()
    if form.validate_on_submit():
        train_ratio = float(form.train_ratio.data)
        model, scaler, acc, report = train_predictive_model(DATASET_PATH, train_ratio=train_ratio)
        results = format_result(acc, report)
        flash("Modello addestrato con successo!", "success")
        return render_template("train.html", form=form, report=results)
    return render_template("train.html", form=form)

@app.route("/grid_search", methods = ["GET", "POST"])
@error_handler("Errore nell'addestramento del modello con Grid Search")
def grid_search():
    form = DataSplitForm()
    if form.validate_on_submit():
        train_ratio = float(form.train_ratio.data)
        model, acc, report = train_with_grid_search(DATASET_PATH, train_ratio=train_ratio)
        results = format_result(acc, report)
        flash("Modello addestrato con Grid Search!", "success")
        return render_template("grid_search.html", form=form, report=results)
    return render_template("grid_search.html", form=form)

@app.route("/compare")
@error_handler("Errore nel confronto dei modelli")
def compare():
    results = compare_models(DATASET_PATH)
    flash("Confronto dei modelli completato!", "success")
    return render_template("compare.html", report=results)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    app.run(debug=True, port=4996)