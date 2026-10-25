import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)

def grafica_metacritic():
    
    colores = {
        "Negative": "red",
        "Mixed": "orange",
        "Positive": "blue"
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for etiqueta, color in colores.items():

        grupo = datos[datos["calidad"] == etiqueta]
        ax.scatter(
            grupo["metacritic_score"],
            grupo["num_reviews_total"],
            label=etiqueta,
            color=color,
            alpha=0.5,
            s=15
        )

    ax.set_xlabel("Puntaje en metacritic")
    ax.set_ylabel("Reseñas totales")
    ax.set_title("Clasificacion por puntajes de Steam")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

def knn(points, labels, new_points, k=5):
    predictions = []
    batch_size = 256

    for start in range(0, len(new_points), batch_size):
        batch = new_points[start:start + batch_size]

        # Distancias euclidianas vectorizadas
        distances = np.sqrt(
            np.sum(
                (batch[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2,
                axis=2
            )
        )

        nearest_indices = np.argpartition(
            distances, k - 1, axis=1
        )[:, :k]

        for row, indices in enumerate(nearest_indices):
            counts = Counter(labels[indices])
            max_count = max(counts.values())

            tied = {
                label for label, count in counts.items()
                if count == max_count
            }

            order = indices[np.argsort(distances[row, indices])]
            prediction = next(
                labels[i] for i in order if labels[i] in tied
            )
            predictions.append(prediction)

    return np.array(predictions)


def clasificacion_metacritic():

    variables = ["metacritic_score", "num_reviews_total"]
    X = datos[variables].to_numpy(dtype=float)
    y = datos["calidad"].to_numpy()

    # Dividir datos en train/test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Estandarizar usando unicamente los datos de entrenamiento
    medias = X_train.mean(axis=0)
    desviaciones = X_train.std(axis=0)
    desviaciones[desviaciones == 0] = 1

    X_train = (X_train - medias) / desviaciones
    X_test = (X_test - medias) / desviaciones

    predicciones = knn(X_train, y_train, X_test, k=5)

    print(f"Juegos analizados: {len(datos)}")
    print(f"Juegos de entrenamiento: {len(X_train)}")
    print(f"Juegos de prueba: {len(X_test)}")
    print("\nPrecision:", round(accuracy_score(y_test, predicciones), 4))
    print(
        "Precision balanceada:",
        round(balanced_accuracy_score(y_test, predicciones), 4)
    )

    print("\nReporte de clasificación:")
    print(classification_report(
        y_test,
        predicciones,
        labels=["Negative", "Mixed", "Positive"],
        zero_division=0
    ))

    return predicciones


#Cargar DF 
df = pd.read_csv("./Practica 1/dataset/steam.csv")

datos = df[
    (df["num_reviews_total"] >= 10) &
    (df["metacritic_score"] > 0) &
    (df["metacritic_score"].notna()) &
    (df["pct_pos_total"] >= 0)
].copy()

datos["calidad"] = np.select(
    [
        datos["pct_pos_total"] < 40,
        datos["pct_pos_total"] <= 70
    ],
    ["Negative", "Mixed"],
    default="Positive"
)

# Grafica de juegos por metacritic-reviews, clasificado por puntaje de steam.
grafica_metacritic()

# No termina siendo muy preciso, ya que la correlaccion entre puntajes de metacritic/numero de reseñas no termina siendo suficiente para predecir la clase, y hay un desbalance de datos.
# Existen juegos con alto metacritic y muchas reseñas que resultan tener relativamente bajos puntajes en steam por factores que el dataset no mide.
clasificacion_metacritic()