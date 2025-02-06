import pandas as pd
import random 
import os
from faker import Faker
from owl.logger_config import setup_logger
from owlready2 import Thing, DataProperty, ObjectProperty, get_ontology, sync_reasoner

if not os.path.exists("log"):
    os.makedirs("log", exist_ok=True)
logger = setup_logger("dataset_generator", "log/dataset_generator.log")


class OntologyManager:
    def __init__(self, ontology_path, output_path=None):
        """
        Inizializzo il manager con il percorso dell'ontonologia.
        Il percorso può essere un URI (es: file://...) o un percorso locale
        """
        self.ontology_path = ontology_path
        self.data = None
        self.output_path = output_path
        self.ontology = None
    
    def load(self):

        """
        Carica l'ontologia dal percorso specificato.
        Se il file non esiste, crea una nuova ontologia.
        """
        if not os.path.exists(self.ontology_path):
            file = open(self.ontology_path, 'w')
            file.close()
            logger.info("Ontologia non trovata, creazione di una nuova ontologia")
            try:
                logger.info(f"Caricamento dell'ontologia da {self.ontology_path}...")
                self.ontology = get_ontology("file://" + self.ontology_path).load()
                logger.info("Ontologia caricata correttamente.")
                return self.ontology
            except Exception as e:
                logger.error(f"Errore durante il caricamento dell'ontologia: {e}", exc_info=True)
                raise
        else:
            try:
                logger.info(f"Caricamento dell'ontologia da {self.ontology_path}...")
                self.ontology = get_ontology("file://" + self.ontology_path).load()
                logger.info("Ontologia caricata correttamente.")
                return self.ontology
            except Exception as e:
                logger.error(f"Errore durante il caricamento dell'ontologia: {e}", exc_info=True)
                raise
    
    def populate(self):
        """
        Popola un'ontologia con dati casuali per scopi dimostrativi.
        """
        logger.info("Popolamento dell'ontologia con dati casuali...")
        try:
            self.ontology = self.load()
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
        except Exception as e:
            logger.error(f"Errore durante il caricamento dell'ontologia: {e}", exc_info=True)


    
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

    def extract_features(self):
        """
        Estrae le informazioni rilevanti dagli individui della classe Person
        presenti nell'ontologia.
        Per ogni persona vengono estratte:
        - Nome (has_name)
        - Età (has_age)
        - Corsi seguiti (takes)
        - Corsi insegnati (teaches)
        """

        self.ontology = get_ontology("file://" + self.ontology_path).load()
        self.reason()

        self.data = []
        try:
            logger.info("Estrazione delle caratteristiche dagli individui...")

            Person = self.ontology.Person
            if Person is None:
                logger.error("Classe Person non trovata nell'ontologia.")
                raise ValueError("Classe Person non trovata nell'ontologia.")

            persons = list(Person.instances())
            logger.info(f"Numero di persone trovate: {len(persons)}")
            
            for person in persons:
                row = {}
                row["name"] = person.has_name[0] if hasattr(person, "has_name") and person.has_name else None
                row["age"] = person.has_age[0] if hasattr(person, "has_age") and person.has_age else None
                
                # Estrazione dei corsi seguiti 
                if hasattr(person, "takes") and person.takes:
                    courses_taken = []
                    for course in person.takes:
                        if hasattr(course, "course_title") and course.course_title:
                            courses_taken.append(course.course_title[0])
                        else:
                            courses_taken.append(course.name)
                    row["courses_taken"] = ", ".join(courses_taken)
                else:
                    row["courses_taken"] = None
                
                # Estrazione dei corsi insegnati
                if hasattr(person, "teaches") and person.teaches:
                    courses_taught = []
                    for course in person.teaches:
                        if hasattr(course, "course_title") and course.course_title:
                            courses_taught.append(course.course_title[0])
                        else:
                            courses_taught.append(course.name)
                    row["courses_taught"] = ", ".join(courses_taught)
                else:
                    row["courses_taught"] = None
                
                self.data.append(row)
            
            logger.info("Estrazione dati completata con successo.")
        except Exception as e:
            logger.error(f"Errore durante l'estrazione delle caratteristiche: {e}", exc_info=True)
            

    def build_dataset(self):
        """
        Costruisce un dataset a partire dai dati estratti e lo salva in formato CSV.
        """
        if not self.data:
            logger.warning("Nessun dato da scrivere nel dataset.")
            return
        
        try:
            df = pd.DataFrame(self.data)
            df.to_csv(self.output_path, index=False)
            logger.info(f"Dataset salvato in {self.output_path}.")
        except Exception as e:
            logger.error(f"Errore durante la scrittura del dataset: {e}", exc_info=True)
