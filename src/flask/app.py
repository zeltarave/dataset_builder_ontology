import os
import sys
from flask import Flask, render_template, redirect, url_for, flash, request
from decorators import error_handler
from flask_wtf.csrf import CSRFProtect
from forms import DataSplitForm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pykeen_learner.learningKnowledge import pyKeenManager
from owl.ontology_manager import OntologyManager
from predictive_model.predictive_model import train_predictive_model, format_result_predictive
from predictive_model.grid_search_model import train_with_grid_search, format_result_grid
from predictive_model.compare_model import compare_models

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Necessario per gestire i messaggi flash
csrf = CSRFProtect(app)

ONTOLOGY_PATH = os.path.join("data", "ontology.owl")
ONTOLOGY_PATH = os.path.abspath(ONTOLOGY_PATH)
DATASET_PATH = os.path.join("data", "dataset.csv")

onto = OntologyManager(ONTOLOGY_PATH, DATASET_PATH)

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
    onto.extract_features()
    onto.build_dataset()

    df = pd.read_csv(DATASET_PATH)
    dataset_html = df.to_html(classes="table table-striped", index=False)

    flash("Dataset estratto e salvato con successo!", "success")
    return render_template("dataset.html", dataset_html=dataset_html)

@app.route("/train", methods = ["GET", "POST"])
@error_handler("Errore nell'addestramento del modello predittivo")
def train():
    form = DataSplitForm()
    if form.validate_on_submit():
        test_size = float(form.train_ratio.data)
        model, scaler, acc, report = train_predictive_model(DATASET_PATH, test_size=test_size)
        results = format_result_predictive(acc, report)
        flash("Modello addestrato con successo!", "success")
        return render_template("train.html", form=form, report=results)
    return render_template("train.html", form=form)

@app.route("/grid_search", methods = ["GET", "POST"])
@error_handler("Errore nell'addestramento del modello con Grid Search")
def grid_search():
    best_model, outer_scores, outer_reports, mean_acc, std_acc = train_with_grid_search(DATASET_PATH)
    results = format_result_grid(outer_scores, outer_reports, mean_acc, std_acc)
    flash("Modello addestrato con Grid Search!", "success")
    return render_template("grid_search.html", report=results)

@app.route("/compare")
@error_handler("Errore nel confronto dei modelli")
def compare():
    results = compare_models(DATASET_PATH)
    flash("Modelli addestrati con successo!", "success")
    return render_template("compare.html", report=results)


@app.route("/plot")
def plot():
    # Addestra il modello
    manager = pyKeenManager().train_model()

    plt = manager.pca()

    # Salva il grafico in un buffer in memoria e converte il buffer in stringa Base64
    buf = io.BytesIO()
    manager.pca().savefig(buf, format="png")
    pca2D = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.seek(0)
    buf = io.BytesIO()
    manager.tsne2D().savefig(buf, format="png")
    tsne2D = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.seek(0)
    buf = io.BytesIO()
    manager.pca3D().savefig(buf, format="png")
    pca3D = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.seek(0)
    
    # Passa l'immagine codificata al template
    return render_template("plot.html", image_pca2D=pca2D, image_tsne2D=tsne2D, image_pca3D=pca3D)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    app.run(debug=True, port=4996)