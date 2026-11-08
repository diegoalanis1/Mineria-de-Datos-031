import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Cargar DF 
df = pd.read_csv("./Practica 1/dataset/steam.csv")

# Seleccionar las variables
datos = df[["metacritic_score", "pct_pos_total"]].copy()

# Eliminar datos faltantes o puntuaciones inválidas
datos = datos.dropna()

datos = datos[
    (datos["metacritic_score"] > 0) &
    (datos["pct_pos_total"] >= 0) &
    (datos["pct_pos_total"] <= 100)
]

# Preparar los datos para K-means
X = datos[["metacritic_score", "pct_pos_total"]]

# Estandarizar las variables
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Aplicar K-means
for k in (2,4):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    datos["cluster"] = modelo.fit_predict(X_escalado)

    # Obtener los centroides en su escala original
    centroides = escalador.inverse_transform(modelo.cluster_centers_)

    # Graficar los resultados
    plt.figure(figsize=(10, 7))

    for cluster in range(k):
        grupo = datos[datos["cluster"] == cluster]

        plt.scatter(
            grupo["metacritic_score"],
            grupo["pct_pos_total"],
            label=f"Cluster {cluster}",
            alpha=0.4,
            s=15
        )

    # Mostrar los centroides
    plt.scatter(
        centroides[:, 0],
        centroides[:, 1],
        marker=".",
        s=200,
        color="red",
        edgecolors="black",
        label="Centroides"
    )

    plt.title("K-means: Metacritic vs porcentaje positivo de Steam")
    plt.xlabel("Puntuación Metacritic")
    plt.ylabel("Porcentaje positivo de Steam")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Analizar los resultados
    print("Cantidad de juegos por cluster:")
    print(datos["cluster"].value_counts().sort_index())

    print("\nPromedio de cada variable por cluster:")
    print(
        datos.groupby("cluster")[
            ["metacritic_score", "pct_pos_total"]
        ].mean().round(2)
    )