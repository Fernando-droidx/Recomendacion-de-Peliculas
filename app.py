from flask import Flask, render_template, request
from utils import cargar_nombres_peliculas,obtener_nombre_pelicula
from recomendation import entrenar_modelo, hacer_predicciones,cargar_datos

app = Flask(__name__)

# Cargar datos
ruta_ratings = 'data/ml-latest-small/ratings.csv'
ruta_peliculas = 'data/ml-latest-small/movies.csv'
data_ratings, data_movies = cargar_datos(ruta_ratings, ruta_peliculas)

# Entrenar el modelo
modelo_entrenado, trainset = entrenar_modelo(data_ratings)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])

        # Obtener los nombres de las películas
        nombres_peliculas = cargar_nombres_peliculas(ruta_peliculas)

        # Hacer predicciones
        recommendations = hacer_predicciones(modelo_entrenado, user_id, nombres_peliculas)

        return render_template('recommendations.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
