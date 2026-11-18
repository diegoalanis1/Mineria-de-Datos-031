from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import ast
from tabulate import tabulate

def nube_etiquetas(df, columna):

    palabras = []

    for etiquetas in df[columna].dropna():
        etiquetas = ast.literal_eval(etiquetas)
        palabras.extend(etiquetas)

    frecuencias = Counter(palabras)

    nube = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate_from_frequencies(frecuencias)

    plt.figure(figsize=(12, 6))
    plt.imshow(nube)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"Practica 9/{columna}_wordcloud.png")
    plt.close()

    print(f"\nPalabras mas frecuentes en {columna}:")
    print(tabulate(frecuencias.most_common(10),headers=[columna, 'Frecuencia'],tablefmt='orgtbl'))

    print("\n")


df = pd.read_csv("./Practica 1/dataset/steam.csv")

nube_etiquetas(df, "categories")
nube_etiquetas(df, "genres")
nube_etiquetas(df, "developers")
nube_etiquetas(df, "publishers")