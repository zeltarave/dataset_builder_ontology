from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty
from faker import Faker
import random, os

fake = Faker("it_IT")

def populate_ontology():
    """
    Popola un'ontologia con dati casuali per scopi dimostrativi.
    """

    onto_path = os.path.join("data", "large_ontology.owl")
    file = open(onto_path, 'w')
    file.close()
    onto = get_ontology("file://" + onto_path).load()

    with onto:
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

        # --- Creazione di Corsi ---
        num_courses = 50  # Possiamo aumentare questo numero per generare più corsi
        courses = []
        for i in range(1, num_courses + 1):
            title = f"Corso {i}"
            course = Course(f"course_{i}")
            course.course_title = [title]
            courses.append(course)

        # --- Creazione di Persone e Assegnazione di Proprietà ---
        num_persons = 1000  # Possiamo aumentare questo numero per generare più persone
        persons = []
        for i in range(1, num_persons + 1):
            name = fake.name()
            person = Person(f"person_{i}")
            person.has_name = [name]

            person.has_age = [random.randint(18, 80)]

            # Assegna a questa persona uno o più corsi casuali
            num_courses_for_person = random.randint(1, 5)
            person_courses = random.sample(courses, num_courses_for_person)
            person.takes.extend(person_courses)

            # (Opzionale) In alcuni casi potremmo voler indicare che la persona insegna un corso
            if random.random() < 0.05:
                teacher_course = random.choice(courses)
                person.teaches.append(teacher_course)

            persons.append(person)

    # --- Salvataggio dell'Ontologia ---
    onto.save(file=onto_path, format="rdfxml")
    print(f"Ontologia creata e salvata in '{onto_path}' con {num_persons} persone e {num_courses} corsi.")
    return onto_path