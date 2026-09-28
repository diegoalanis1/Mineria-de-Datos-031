import pandas as pd
import matplotlib.pyplot as plt
import ast

# Función auxiliar para la grafica 1
def classify_playermode(value):

    if pd.isna(value):
        return "Sin datos"

    try:
        categories = ast.literal_eval(value)

        if not isinstance(categories, list):
            return "Sin datos"

        single = "Single-player" in categories
        multi = "Multi-player" in categories

        if single and multi:
            return "Ambos"
        elif single:
            return "Single-player"
        elif multi:
            return "Multi-player"
        else:
            return "Sin datos"

    except (ValueError, SyntaxError):
        return "Sin datos"

# Grafica 1. Juegos lanzados por año-mes, clasificados por single/multi-player
def grafica_juegos_por_mes():

    monthly_games = (
        df_dates
        .groupby(["year_month", "tipo_multijugador"])
        .size()
        .unstack(fill_value=0)
    )

    categorias = [
        "Single-player",
        "Multi-player",
        "Ambos",
        "Sin datos"
    ]

    # Asegurar que todas las categorías existan
    for categoria in categorias:
        if categoria not in monthly_games.columns:
            monthly_games[categoria] = 0

    monthly_games = monthly_games[categorias]

    fig, ax = plt.subplots(figsize=(16, 7))

    monthly_games.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title("Cantidad de juegos lanzados por mes")
    ax.set_xlabel("Año")
    ax.set_ylabel("Cantidad de juegos")
    ax.legend(title="Tipo de juego")

    # Crear ticks para enero de cada año
    primer_anio = monthly_games.index[0].year

    # Obtener el último año de la gráfica
    ultimo_anio = monthly_games.index[-1].year

    posiciones = []
    etiquetas = []

    for anio in range(primer_anio + 1, ultimo_anio + 1):
        # Calcular posicion en base a 1970-01
        posiciones.append((anio-1970)*12)
        etiquetas.append(str(anio))

    ax.set_xticks(posiciones)
    ax.set_xticklabels(etiquetas)

    plt.tight_layout()
    plt.show()


# Grafica 2. Distribución de precios de juegos de paga
def grafica_distribucion_precios():

    # Juegos de paga entre $0 y $100 USD
    paid_games = df[
        (df["price"] > 0) &
        (df["price"] <= 100)
    ]

    #print("Juegos de paga entre $0 y $100:", len(paid_games))
    #print("Juegos de paga fuera del rango:", len(df[df["price"] > 100]))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        paid_games["price"],
        bins=20
    )

    ax.set_title(
        "Distribución de precios de juegos de paga ($0-$100 USD)"
    )
    ax.set_xlabel("Precio (USD)")
    ax.set_ylabel("Cantidad de juegos")

    plt.tight_layout()
    plt.show()

# Grafica 3. Histograma de puntajes de metacritic
def grafica_distribucion_metacritic():

    # Juegos con puntaje de metacritic
    df_metacritic = df[
        (df["metacritic_score"] > 0)
    ]

    print("Juegos con puntaje de metacritic:", len(df_metacritic))
    print("Juegos sin puntaje de metacritic:", len(df[df["metacritic_score"] <= 0]))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df_metacritic["metacritic_score"],
        bins=20
    )

    ax.set_title(
        "Distribución de puntajes de metacritic de juegos en Steam"
    )
    ax.set_xlabel("Metacritic Score")
    ax.set_ylabel("Cantidad de juegos")

    plt.tight_layout()
    plt.show()

# Grafica 4. Scatter plot: precio vs fecha
def grafica_precio_fecha():

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.scatter(
        df_dates["release_date"],
        df_dates["price"],
        s=8,
        alpha=0.4
    )

    ax.set_title(
        "Precio de los juegos según su fecha de lanzamiento"
    )
    ax.set_xlabel("Fecha de lanzamiento")
    ax.set_ylabel("Precio (USD)")

    plt.tight_layout()
    plt.show()

# Grafica 5. Scatter plot: metacritic vs reseñas
def grafica_puntajes():

    datos = df[
        (df["metacritic_score"] > 0) &
        (df["pct_pos_total"] >= 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        datos["metacritic_score"],
        datos["pct_pos_total"],
        s=8,
        alpha=0.4
    )

    ax.set_title(
        "Puntuación de Metacritic vs. porcentaje de reseñas positivas"
    )
    ax.set_xlabel("Puntuación de Metacritic")
    ax.set_ylabel("Reseñas positivas (%)")

    plt.tight_layout()
    plt.show()


# Grafica 6. Disponibilidad por sistema operativo
def grafica_sistemas_operativos():

    operating_systems = {
        "Windows": "windows",
        "Mac": "mac",
        "Linux": "linux"
    }

    colors = ['#266bde','#f72626']

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Generar los 3 pie subplots
    for ax, (name, column) in zip(
        axes,
        operating_systems.items()
    ):

        compatible = (df[column] == True).sum()
        incompatible = (df[column] == False).sum()

        counts = [
            compatible,
            incompatible
        ]

        labels = [
            "Sí",
            "No"
        ]

        ax.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%"
        )

        ax.set_title(f"Disponibilidad para {name}")

    fig.suptitle(
        "Disponibilidad de juegos por sistema operativo"
    )

    plt.tight_layout()

    fig.subplots_adjust(top=0.835)

    plt.show()

# Cargar y preparar datos
df = pd.read_csv("./Practica 1/dataset/steam.csv")

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df_dates = df.dropna(subset=["release_date"]).copy()

df_dates["year"] = df_dates["release_date"].dt.year
df_dates["year_month"] = df_dates["release_date"].dt.to_period("M")


# Preparar clasificación de categorías
df_dates["tipo_multijugador"] = (
    df_dates["categories"]
    .apply(classify_playermode)
)

# Ejecutar Graficas
grafica_juegos_por_mes()
grafica_distribucion_precios()
grafica_distribucion_metacritic()
grafica_precio_fecha()
grafica_puntajes()
grafica_sistemas_operativos()