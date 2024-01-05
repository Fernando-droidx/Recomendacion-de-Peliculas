from surprise import SVD
from surprise.model_selection import train_test_split
from data_loader import cargar_datos

def entrenar_modelo(ruta_ratings):
    # Cargar datos
    data = cargar_datos(ruta_ratings)

    # Dividir datos en conjunto de entrenamiento y prueba
    trainset, testset = train_test_split(data, test_size=0.2)

    # Crear y entrenar el modelo
    modelo = SVD()
    modelo.fit(trainset)

    return modelo, trainset