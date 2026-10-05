import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
import ast

# Función auxiliar para las pruebas de kruskal
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

# Kruskal Wallis:
# Relacion entre tipo de multiplayer con precio, reseñas positivas y metacritic

def pruebas_kruskal():

    variables = {
        "price": "Precio",
        "pct_pos_total": "Porcentaje de reseñas positivas",
        "metacritic_score": "Puntuación de Metacritic"
    }

    categorias = [
        "Single-player",
        "Multi-player",
        "Ambos"
    ]

    print("\n" + "=" * 60)
    print("PRUEBAS KRUSKAL-WALLIS")
    print("=" * 60)

    for variable, nombre in variables.items():

        # Eliminar juegos sin categoría de multiplayer
        # y valores inexistentes
        datos = df[
            (df["tipo_multijugador"] != "Sin datos") &
            (df[variable].notna())
        ].copy()

        # Valores especiales que representan ausencia de datos
        if variable == "metacritic_score":
            datos = datos[datos[variable] > 0]

        elif variable == "pct_pos_total":
            datos = datos[datos[variable] >= 0]

        elif variable == "price":
            datos = datos[datos[variable] >= 0]

        grupos = []

        print(f"\n--- {nombre} ---")

        for categoria in categorias:

            grupo = datos[
                datos["tipo_multijugador"] == categoria
            ][variable]

            grupos.append(grupo)

            print(
                f"{categoria}: {len(grupo)} juegos"
            )

        # Aplicar Kruskal-Wallis
        estadistico, p = kruskal(*grupos)

        print(f"\nEstadístico H: {estadistico:.4f}")
        print(f"p-value: {p:.10g}")

        if p < 0.05:
            print(
                "Resultado: Existen diferencias estadísticamente "
                "significativas entre los grupos."
            )
        else:
            print(
                "Resultado: No existe evidencia suficiente de "
                "diferencias entre los grupos."
            )

# Mann Whitney:
# Relacion entre requerimiento de edad y precio

def prueba_edad_precio():

    datos = df[
        (df["required_age"].notna()) &
        (df["price"].notna()) &
        (df["price"] >= 0)
    ].copy()

    # Juegos sin restricción de edad
    grupo_sin_edad = datos[
        datos["required_age"] == 0
    ]["price"]

    # Juegos con alguna restricción de edad
    grupo_con_edad = datos[
        datos["required_age"] > 0
    ]["price"]

    estadistico, p = mannwhitneyu(
        grupo_sin_edad,
        grupo_con_edad,
        alternative="two-sided"
    )

    print("\n" + "=" * 60)
    print("PRUEBA MANN-WHITNEY: REQUIRED AGE VS PRECIO")
    print("=" * 60)

    print(
        f"\nJuegos sin restricción de edad: "
        f"{len(grupo_sin_edad)}"
    )

    print(
        f"Juegos con restricción de edad: "
        f"{len(grupo_con_edad)}"
    )

    print(
        f"\nMediana sin restricción: "
        f"${grupo_sin_edad.median():.2f}"
    )

    print(
        f"Mediana con restricción: "
        f"${grupo_con_edad.median():.2f}"
    )

    print(f"\nEstadístico U: {estadistico:.4f}")
    print(f"p-value: {p:.10g}")

    if p < 0.05:
        print(
            "Resultado: Existe una diferencia estadísticamente "
            "significativa entre los grupos."
        )
    else:
        print(
            "Resultado: No existe evidencia suficiente de una "
            "diferencia estadísticamente significativa."
        )


# Prueba de Chi Cuadrada:
# Relacion entre disponibilidad en mac y linux

def prueba_mac_linux():

    # Crear tabla de contingencia
    tabla = pd.crosstab(
        df["mac"],
        df["linux"]
    )

    chi2, p, grados_libertad, esperados = chi2_contingency(tabla)

    print("\n" + "=" * 60)
    print("PRUEBA CHI-CUADRADA: MAC VS LINUX")
    print("=" * 60)

    print("\nTabla de contingencia:")
    print(tabla)

    print(f"\nChi-cuadrada: {chi2:.4f}")
    print(f"p-value: {p:.10g}")

    if p < 0.05:
        print(
            "\nResultado: Existe una relación estadísticamente "
            "significativa entre la disponibilidad en Mac y Linux."
        )
    else:
        print(
            "\nResultado: No existe evidencia suficiente de una "
            "relación entre la disponibilidad en Mac y Linux."
        )

# Cargar datos
df = pd.read_csv("./Practica 1/dataset/steam.csv")

# Preparar clasificación de categorías
df["tipo_multijugador"] = (
    df["categories"]
    .apply(classify_playermode)
)

# Ejecutar las pruebas

pruebas_kruskal()
prueba_edad_precio()
prueba_mac_linux()