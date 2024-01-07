# En tu archivo recomendation.py

from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import Dataset
from surprise import Reader
def cargar_datos(ruta_ratings):
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5), skip_lines=1)
    data = Dataset.load_from_file(ruta_ratings, reader=reader)
    return data

def entrenar_modelo(ruta_ratings):
    # Cargar datos
    data = cargar_datos(ruta_ratings)

    # Dividir datos en conjunto de entrenamiento y prueba
    trainset, testset = train_test_split(data, test_size=0.2)

    # Crear y entrenar el modelo
    modelo = SVD()
    modelo.fit(trainset)

    return modelo, trainset

def hacer_predicciones(modelo_entrenado, user_id):
    # Obtener ítems aún no calificados por el usuario
    items_no_calificados = [item for item in modelo_entrenado.trainset.all_items() if user_id not in modelo_entrenado.trainset.ur[user_id]]

    # Hacer predicciones para los ítems no calificados
    predicciones = [modelo_entrenado.predict(user_id, item_id) for item_id in items_no_calificados]

    # Ordenar las predicciones por puntuación descendente
    predicciones_ordenadas = sorted(predicciones, key=lambda x: x.est, reverse=True)

    # Obtener las 5 mejores recomendaciones
    top_recomendaciones = predicciones_ordenadas[:5]

    # Devolver las recomendaciones en un formato adecuado
    recomendaciones_dict = {pred.iid: pred.est for pred in top_recomendaciones}

    return recomendaciones_dict
