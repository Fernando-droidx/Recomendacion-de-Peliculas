import pandas as pd
#Expresar datos en pandas
def cargar_nombres_peliculas(ruta_peliculas):
    peliculas_df = pd.read_csv(ruta_peliculas)
    nombres_peliculas = dict(zip(peliculas_df['movieId'], peliculas_df['title']))
    return nombres_peliculas

def obtener_nombre_pelicula(movie_id, nombres_peliculas):
    return nombres_peliculas.get(movie_id, f"Película {movie_id}")
