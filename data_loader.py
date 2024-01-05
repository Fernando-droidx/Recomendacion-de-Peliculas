from surprise import Dataset
from surprise import Reader
import os


def cargar_datos(ruta_ratings):
    # Define la estructura del conjunto de datos
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

    # Carga el conjunto de datos desde archivos CSV
    data = Dataset.load_from_file(ruta_ratings, reader=reader)

    # Retorna el objeto Dataset
    return data