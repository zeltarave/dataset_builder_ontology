import pandas as pd

def extract_features(ontology):
    """
    Estrae caratteristiche dagli individui dell'ontologia.
    """
    data = []
    for individual in ontology.individuals():
        features = {}
        features['nome'] = individual.name
        
        if hasattr(individual, 'has_value'):
            values = individual.has_value
            features['has_value'] = values[0] if values else None
        else:
            features['has_value'] = None
        
        data.append(features)
    
    return data

def build_dataset(data: list, output_path: str):
    """
    Converte la lista di dizionari in un DataFrame pandas e lo salva in formato CSV.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset salvato in {output_path}")