import pandas as pd
import random 
import os
from faker import Faker
from owl.logger_config import setup_logger
from owlready2 import Thing, DataProperty, ObjectProperty, get_ontology, sync_reasoner

logger = setup_logger("dataset_generator", "log/dataset_generator.log")


class OntologyManager:
    def __init__(self, ontology_path):
        """
        Inizializzo il manager con il percorso dell'ontonologia.
        Il percorso può essere un URI (es: file://...) o un percorso locale
        """
        self.ontology_path = ontology_path
        self.ontology = None
    
    def load(self):
        """
        Carica l'ontologia dal percorso specificato.
        Se il file non esiste, crea una nuova ontologia.
        """
        try:
            file_path = self.ontology_path
            if file_path.startswith("file://"):
                file_path = file_path[7:]
            
            if not os.path.exists(file_path):
                logger.info(f"Il file {file_path} non esiste. Verrà creata una nuova ontologia.")
                self.ontology = get_ontology(self.ontology_path)
            else:
                logger.info(f"Caricamento dell'ontologia da {self.ontology_path}...")
                self.ontology = get_ontology(self.ontology_path).load()
            
            logger.info("Ontologia caricata (o creata) correttamente.")
        except Exception as e:
            logger.error(f"Errore nel caricamento dell'ontologia: {e}", exc_info=True)
            raise
        return self.ontology

    def populate(self):
        """
        Popola un'ontologia con dati casuali per scopi dimostrativi.
        """
        if self.ontology is None:
            raise ValueError("Ontologia non caricata. Caricare l'ontologia con load() prima di popolarla.")
        
        with self.ontology:
            class Person(Thing):
                pass

            class Course(Thing):
                pass

            class teaches(ObjectProperty):
                domain = [Person]
                range = [Course]

            class takes(ObjectProperty):
                domain = [Person]
                range = [Course]

            class has_age(DataProperty):
                domain = [Person]
                range = [int]

            class has_name(DataProperty):
                domain = [Person]
                range = [str]

            class course_title(DataProperty):
                domain = [Course]
                range = [str]

            class course_description(DataProperty):
                domain = [Course]
                range = [str]

            # Popolamento dell'ontologia con dati
            fake = Faker("it_IT")
            num_courses = 50
            courses = []
            for i in range(1, num_courses + 1):
                title = f"Corso {i}"
                course = Course(f"course_{i}")
                course.course_title = [title]
                course.course_description = [f"Descrizione per {title}"]
                courses.append(course)

            num_persons = 1000
            for i in range(1, num_persons + 1):
                person = Person(f"person_{i}")
                person.has_name = [fake.name()]
                person.has_age = [fake.random_int(min=18, max=80)]
                # Assegna casualmente 1-5 corsi a cui la persona è iscritta
                person.takes.extend(random.sample(courses, k=random.randint(1, 5)))
                # Per una percentuale delle persone, assegna anche un corso da insegnare
                if random.random() < 0.05:
                    person.teaches.append(random.choice(courses))

        try:
            self.ontology.save(file=self.ontology_path, format="rdfxml")
            logger.info(f"Ontologia popolata e salvata in {self.ontology_path}.")
        except Exception as e:
            logger.error(f"Errore nel salvataggio dell'ontologia: {e}", exc_info=True)
            raise
    
    def reason(self):
        """
        Esegue il ragionamento sull'ontologia (sincronizzazione del ragionatore).
        """
        if self.ontology is None:
            raise ValueError("Ontologia non caricata: chiamare load() prima di reason().")
        try:
            with self.ontology:
                sync_reasoner()
            logger.info("Ragionamento completato con successo.")
        except Exception as e:
            logger.error(f"Errore durante il ragionamento: {e}", exc_info=True)
            raise

    def extract_person_data(self):
        """
        Estrae le informazioni rilevanti dagli individui della classe Person
        presenti nell'ontologia.
        Per ogni persona vengono estratte:
        - Nome (has_name)
        - Età (has_age)
        - Corsi seguiti (takes)
        - Corsi insegnati (teaches)
        """
        if self.ontology is None:
            raise ValueError("Ontologia non caricata. Caricare l'ontologia con load() prima di estrarre i dati.")
        try:
            Person = self.ontology.Person
        except AttributeError:
            print("La classe 'Person' non è stata trovata nell'ontologia. Verifica la struttura.")
            return []
        
        data = []
        try:
            for person in Person.instances():
                row = {}
                row["name"] = person.has_name[0] if hasattr(person, "has_name") and person.has_name else None
                row["age"] = person.has_age[0] if hasattr(person, "has_age") and person.has_age else None

                # Estrazione dei corsi seguiti
                if hasattr(person, "takes") and person.takes:
                    courses_taken = []
                    for course in person.takes:
                        title = course.course_title[0] if hasattr(course, "course_title") and course.course_title else course.name
                        courses_taken.append(title)
                    row["courses_taken"] = ", ".join(courses_taken)
                else:
                    row["courses_taken"] = None
            
                # Estrazione dei corsi insegnati
                if hasattr(person, "teaches") and person.teaches:
                    courses_taught = []
                    for course in person.teaches:
                        title = course.course_title[0] if hasattr(course, "course_title") and course.course_title else course.name
                        courses_taught.append(title)
                    row["courses_taught"] = ", ".join(courses_taught)
                else:
                    row["courses_taught"] = None
                data.append(row)
            logger.info("Dati delle persone estratti con successo.")
            return data
        except Exception as e:
            logger.error(f"Errore durante l'estrazione delle caratteristiche: {e}", exc_info=True)
            return data
            
    def extract_course_data(self):
        """
        Estrae i dati degli individui della classe Course.
        Restituisce una lista di dizionari contenenti titolo e descrizione.
        """
        if self.ontology is None:
            raise ValueError("Ontologia non caricata.")
        try:
            Course = self.ontology.Course
        except AttributeError:
            raise ValueError("La classe 'Course' non è definita nell'ontologia.")

        data = []
        try:
            for course in Course.instances():
                row = {}
                row["title"] = course.course_title[0] if hasattr(course, "course_title") and course.course_title else None
                row["description"] = course.course_description[0] if hasattr(course, "course_description") and course.course_description else None
                data.append(row)
            logger.info("Dati dei corsi estratti con successo.")
            return data
        except Exception as e:
            logger.error(f"Errore durante l'estrazione dei dati dei corsi: {e}", exc_info=True)
            return data

    def build_dataset(self, data, output_path):
        """
        Costruisce un dataset a partire dai dati estratti e lo salva in formato CSV.
        """
        if not data:
            logger.warning("Nessun dato da scrivere nel dataset.")
            return
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Dataset salvato in {output_path}.")
        except Exception as e:
            logger.error(f"Errore durante la scrittura del dataset: {e}", exc_info=True)
