from data_loader import cargar_datos
from recomendation import entrenar_modelo

# Cargar datos
ruta_ratings = 'data/ml-latest-small/ratings.csv'
data = cargar_datos(ruta_ratings)

# Entrenar el modelo
ruta_ratings = 'data/ml-latest-small/ratings.csv'
modelo_entrenado, trainset = entrenar_modelo(ruta_ratings)

# Obtener las mejores N recomendaciones para un usuario específico
usuario_id = 1
n = 10
recomendaciones_usuario = []

for movie_id in trainset.all_items():
    prediction = modelo_entrenado.predict(usuario_id, movie_id)
    recomendaciones_usuario.append((movie_id, prediction.est))

# Ordenar las recomendaciones por score
recomendaciones_usuario.sort(key=lambda x: x[1], reverse=True)

# Obtener las primeras N recomendaciones
top_recomendaciones = recomendaciones_usuario[:n]

print(f"Top {n} recomendaciones para el usuario {usuario_id}:")
for movie_id, score in top_recomendaciones:
    print(f"Película ID: {movie_id}, Score: {score}")
