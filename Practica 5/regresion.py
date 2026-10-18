import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def regresion_mensual(mensual, titulo, nombre_y):

    mensual = mensual.dropna()

    # ------ Regresion lineal completa ------

    # Convertir los meses a fechas para graficar
    fechas = mensual.index.to_timestamp()

    # Ajustar modelo 1 de regresion
    x = np.array([mes.ordinal for mes in mensual.index])
    y = mensual.values
    modelo_completo = sm.OLS(y, sm.add_constant(x)).fit()

    # ------ Regresion de 2015 en adelante ------

    mascara_2015 = mensual.index >= pd.Period("2015-01", freq="M")

    # Ajustar modelo 2 de regresion
    mensual_2015 = mensual[mascara_2015]
    fechas_2015 = mensual_2015.index.to_timestamp()
    y_2015 = mensual_2015.values

    # Reiniciar X para que empiece en cero en enero de 2015
    x_2015 = x_2015 = np.array([mes.ordinal for mes in mensual_2015.index])

    modelo_2015 = sm.OLS(y_2015, sm.add_constant(x_2015)).fit()

    # Calcular valores predichos
    predicciones_completo = modelo_completo.predict(sm.add_constant(x))
    predicciones_2015 = modelo_2015.predict(sm.add_constant(x_2015))

    # ------------ Graficar ------------

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(
        fechas,
        y,
        s=15,
        alpha=0.5,
        label=nombre_y
    )

    ax.plot(
        fechas,
        predicciones_completo,
        color="red",
        label="Regresión: todos los años"
    )

    ax.plot(
        fechas_2015,
        predicciones_2015,
        color="green",
        label="Regresión: 2015 en adelante"
    )

    ax.set_title(titulo)
    ax.set_xlabel("Mes-año")
    ax.set_ylabel(nombre_y)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Mostrar resultados
    print(f"\n--- {titulo}: todos los años ---")
    print(f"R²: {modelo_completo.rsquared:.4f}")
    print(f"Pendiente: {modelo_completo.params[1]:.4f}")
    print(f"Intercepto: {modelo_completo.params[0]:.4f}")

    print(f"\n--- {titulo}: desde 2015 ---")
    print(f"R²: {modelo_2015.rsquared:.4f}")
    print(f"Pendiente: {modelo_2015.params[1]:.4f}")
    print(f"Intercepto: {modelo_2015.params[0]:.4f}")

    #print("\nResumen estadístico: todos los años")
    #print(modelo_completo.summary())

    #print("\nResumen estadístico: desde 2015")
    #print(modelo_2015.summary())


#Cargar DF 
df = pd.read_csv("./Practica 1/dataset/steam.csv")

#Preparar las columnas para acomodar por mes-año
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df_dates = df.dropna(subset=["release_date"]).copy()
df_dates["year"] = df_dates["release_date"].dt.year
df_dates["year_month"] = df_dates["release_date"].dt.to_period("M")


regresion_mensual(
    df_dates.groupby("year_month").size(),
    "Regresión lineal de juegos lanzados por mes",
    "Cantidad de juegos"
)

#
regresion_mensual(
    df_dates.groupby("year_month")["achievements"].mean(),
    "Regresión lineal de logros promedio por mes",
    "Logros promedio"
)

regresion_mensual(
    df_dates.groupby("year_month")["price"].mean(),
    "Regresión lineal de precio promedio por mes",
    "Precio promedio (USD)"
)