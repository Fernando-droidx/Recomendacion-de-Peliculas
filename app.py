from flask import Flask, render_template, request
from data_loader import cargar_datos
from recomendation import entrenar_modelo, hacer_predicciones

app = Flask(__name__)

# Cargar datos y entrenar el modelo

ruta_ratings = 'data/ml-latest-small/ratings.csv'
modelo_entrenado, trainset = entrenar_modelo(ruta_ratings)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        recommendations = hacer_predicciones(modelo_entrenado, user_id)
        return render_template('recommendations.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
