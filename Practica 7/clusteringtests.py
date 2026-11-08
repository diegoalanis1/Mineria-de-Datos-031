import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#Cargar DF 
df = pd.read_csv("./Practica 1/dataset/steam.csv")

# 1. Preparar los datos
datos = df[["metacritic_score", "pct_pos_total"]].copy()

datos = datos.dropna()

datos = datos[
    (datos["metacritic_score"] > 0) &
    (datos["pct_pos_total"] >= 0) &
    (datos["pct_pos_total"] <= 100)
]

X = datos[["metacritic_score", "pct_pos_total"]]

# 2. Estandarizar
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# 3. Probar diferentes valores de k
resultados = []

valores_k = range(2, 11)

for k in valores_k:

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo.fit_predict(X_escalado)

    inercia = modelo.inertia_
    silhouette = silhouette_score(X_escalado, etiquetas)

    resultados.append({
        "k": k,
        "inercia": inercia,
        "silhouette": silhouette
    })

# 4. Mostrar resultados
resultados = pd.DataFrame(resultados)

print(resultados.round(4))

# 5. Graficar método del codo y Silhouette
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    resultados["k"],
    resultados["inercia"],
    marker="o"
)
axes[0].set_title("Método del codo")
axes[0].set_xlabel("Número de clusters (k)")
axes[0].set_ylabel("Inercia")
axes[0].set_xticks(valores_k)
axes[0].grid(alpha=0.3)

axes[1].plot(
    resultados["k"],
    resultados["silhouette"],
    marker="o"
)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Número de clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(valores_k)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Identificar el mejor k según Silhouette
mejor_k = resultados.loc[
    resultados["silhouette"].idxmax(), "k"
]

print(f"\nEl mejor k según Silhouette es: {mejor_k}")