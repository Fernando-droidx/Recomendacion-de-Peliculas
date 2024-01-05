import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from surprise import Dataset
from surprise.model_selection import train_test_split

#Cargamos los datos
def cargar_datos():

    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    ruta_datos = os.path.join(ruta_actual, 'data', 'ml-latest-small')


    ruta_movies = os.path.join(ruta_datos, 'movies.csv')
    ruta_ratings = os.path.join(ruta_datos, 'ratings.csv')

    # Carga los datos en DataFrames de pandas
    movies = pd.read_csv(ruta_movies)
    ratings = pd.read_csv(ruta_ratings)

    return movies, ratings

# Función para realizar análisis exploratorio de datos
def eda(movies, ratings):

    print("Estadísticas de Películas:")
    print(movies.describe())

    print("\nEstadísticas de Calificaciones:")
    print(ratings.describe())


    sns.histplot(ratings['rating'])
    plt.title('Distribución de Calificaciones')
    plt.show()


def recomendacion_popularidad(ratings, movies, n=5):

    promedio_calificaciones = ratings.groupby('movieId')['rating'].mean()

    peliculas_populares = promedio_calificaciones.sort_values(ascending=False).head(n)


    peliculas_populares = pd.Series(movies.set_index('movieId').loc[peliculas_populares.index]['title'].values,
                                    index=peliculas_populares.index)

    return peliculas_populares


def filtrado_colaborativo(ratings, movies, n=5):
    datos = Dataset.load_builtin('ml-latest-small')
    #Entrenando el modelo
    #Aquí, el 80% de los datos se usarán para entrenar el modelo y el 20% se usará para evaluar su rendimiento.
    trainset, testset = train_test_split(datos, test_size=0.2, random_state=42)
