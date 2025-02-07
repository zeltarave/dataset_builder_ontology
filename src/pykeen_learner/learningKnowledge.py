import os
import sys
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from owlready2 import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from owl.ontology_manager import OntologyManager

ONTOLOGY_PATH = os.path.join(parent_dir, "..", "data", "ontology.owl")
ONTOLOGY_PATH = os.path.abspath(ONTOLOGY_PATH)

onto = OntologyManager(ONTOLOGY_PATH)

class pyKeenManager:
    def __init__(self):
        self.ontology = None
        self.triples = None
        self.has_name_triples = None
        self.entity_labels = None
        self.embeddings = None

    
    def extract_triples(self):
        onto.load()
        self.triples = []
        try:
            Person = onto.ontology.Person
        except AttributeError:
            raise ValueError("La classe 'Person' non è definita nell'ontologia.")

        for person in Person.instances():
            head = person.name 
            if hasattr(person, "takes") and person.takes:
                for course in person.takes:
                    self.triples.append((head, "takes", course.name))
            if hasattr(person, "teaches") and person.teaches:
                for course in person.teaches:
                    self.triples.append((head, "teaches", course.name))

    def extract_has_name(self):
        self.has_name_triples = []
        try:
            Person = onto.ontology.Person
        except AttributeError:
            raise ValueError("La classe 'Person' non è definita nell'ontologia.")
        
        for person in Person.instances():
            head = person.name 
            if hasattr(person, "has_name") and person.has_name:
                self.has_name_triples.append((head, "has_name", person.has_name[0]))



    def find_name(self, id):
        for triple in self.has_name_triples:
            if triple[0] == id:
                label = triple[2]
                return label
        return

    def train_model(self):
        self.extract_triples()
        self.extract_has_name()
        triples_array = np.array(self.triples)

        tf = TriplesFactory.from_labeled_triples(triples_array)
        print(tf)

        tf_train, tf_test, tf_valid = tf.split([0.8, 0.1, 0.1])

        result = pipeline(
            training=tf_train,
            testing=tf_test,
            validation=tf_valid,
            model="TransE",
            training_kwargs=dict(num_epochs=100),
        )
        print(result)

        entity_embedding = result.model.entity_representations[0]
        embeddings_tensor = entity_embedding()
        self.embeddings = embeddings_tensor.detach().cpu().numpy() 

        entity_to_id = tf.entity_to_id
        id_to_entity = {v: k for k, v in entity_to_id.items()}
        self.entity_labels = [id_to_entity[i] for i in range(len(id_to_entity))]

        return self

    def show_graphs(manager):
        manager.pca().show()
        manager.tsne2D().show()
        manager.pca3D().show()

    # # --- A) Usando PCA per una visualizzazione 2D
    def pca(self):
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=50, alpha=0.7)
        for i, label in enumerate(self.entity_labels):
            plt.annotate(self.find_name(label), (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.75)
        plt.title("Visualizzazione 2D delle entità (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        return plt

    # --- B) Usando t-SNE per una visualizzazione 2D (può evidenziare strutture non lineari)
    def tsne2D(self):
        tsne_2d = TSNE(n_components=2, random_state=42, init="random")
        embeddings_tsne_2d = tsne_2d.fit_transform(self.embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_tsne_2d[:, 0], embeddings_tsne_2d[:, 1], s=50, alpha=0.7, c='green')
        for i, label in enumerate(self.entity_labels):
            plt.annotate(self.find_name(label), (embeddings_tsne_2d[i, 0], embeddings_tsne_2d[i, 1]), fontsize=8, alpha=0.75)
        plt.title("Visualizzazione 2D delle entità (t-SNE)")
        plt.xlabel("Dimensione 1")
        plt.ylabel("Dimensione 2")
        plt.grid(True)
        return plt

    # --- C) Visualizzazione 3D con PCA
    def pca3D(self):
        pca_3d = PCA(n_components=3)
        embeddings_3d = pca_3d.fit_transform(self.embeddings)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], s=50, alpha=0.7)
        for i, label in enumerate(self.entity_labels):
            ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], self.find_name(label), size=8, zorder=1, color='k')
        ax.set_title("Visualizzazione 3D delle entità (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        return plt

if __name__ == "__main__":
    pyKeen = pyKeenManager()
    pyKeen.train_model()