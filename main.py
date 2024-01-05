
from data_explorer import cargar_datos, eda,recomendacion_popularidad

movies, ratings = cargar_datos()
eda(movies, ratings)


peliculas_populares = recomendacion_popularidad(ratings, movies, n=5)

# Imprimir las películas recomendadas
print("Películas Populares Recomendadas:")
print(peliculas_populares)

