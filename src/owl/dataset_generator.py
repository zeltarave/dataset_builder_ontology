import pandas as pd
from owl.logger_config import setup_logger

logger = setup_logger("dataset_generator", "log/dataset_generator.log")

def extract_features(ontology):
    """
    Estrae le informazioni rilevanti dagli individui della classe Person
    presenti nell'ontologia.
    Per ogni persona vengono estratte:
      - Nome (has_name)
      - Età (has_age)
      - Corsi seguiti (takes)
      - Corsi insegnati (teaches)
    """
    data = []
    try:
        logger.info("Estrazione delle caratteristiche dagli individui...")
        try:
            Person = ontology.Person
        except AttributeError:
            print("La classe 'Person' non è stata trovata nell'ontologia. Verifica la struttura.")
            return data

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
            
            data.append(row)
        
        logger.info("Estrazione dati completata con successo.")
        return data
    except Exception as e:
        logger.error(f"Errore durante l'estrazione delle caratteristiche: {e}", exc_info=True)
        return data

def build_dataset(data: list, output_path: str):
    """
    Converte la lista di dizionari in un DataFrame pandas e lo salva in formato CSV.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset salvato in {output_path}")