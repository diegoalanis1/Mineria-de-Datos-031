from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import ast
from tabulate import tabulate

def cloud_column(df, columna, tipo):
    palabras = []

    if tipo == "lista":
        for p in df[columna].dropna():
            p = ast.literal_eval(p)
            palabras.extend(p)
    elif tipo == "texto":

        filler = ["the", "and", "a", "at", "to", "of", "in", "is", "you", "your", "with", "will", "when", "why", "what", "who", "be", "but", "an", "on", "as", "for", "has", "that", "this", "by", "from", "where", "can", "are", "or", "it", "into", "their", "have"]

        for p in df[columna].dropna():
            p = str(p).lower().split()
            for s in p:
                if s not in filler:
                    palabras.append(s)
    else:
        print("Tipo desconocido.")
        return

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
    print(tabulate(frecuencias.most_common(20),headers=[columna, 'Frecuencia'],tablefmt='orgtbl'))

    print("\n")


df = pd.read_csv("./Practica 1/dataset/steam.csv")


cloud_column(df, "categories", "lista")
cloud_column(df, "genres", "lista")
cloud_column(df, "developers", "lista")
cloud_column(df, "publishers", "lista")
cloud_column(df, "name", "texto")
cloud_column(df, "short_description", "texto")