import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def regresion_mensual(mensual, titulo, nombre_y):

    mensual = mensual.dropna()

    x = np.array([mes.ordinal for mes in mensual.index])
    y = mensual.values

    # Convertir los meses a fechas para graficar
    fechas = mensual.index.to_timestamp()

    # Regresion de 2015 en adelante

    mascara_2015 = mensual.index >= pd.Period("2015-01", freq="M")

    # Ajustar modelo 2015 de regresion
    mensual_2015 = mensual[mascara_2015]
    fechas_2015 = mensual_2015.index.to_timestamp()
    y_2015 = mensual_2015.values

    # Reiniciar X para que empiece en cero en enero de 2015
    x_2015 = x_2015 = np.array([mes.ordinal for mes in mensual_2015.index])

    modelo_2015 = sm.OLS(y_2015, sm.add_constant(x_2015)).fit()

    # Regresion de 2021 en adelante
    
    mascara_2021 = mensual.index >= pd.Period("2021-01", freq="M")

    # Ajustar modelo 2021 de regresion
    mensual_2021 = mensual[mascara_2021]
    fechas_2021 = mensual_2021.index.to_timestamp()
    y_2021 = mensual_2021.values

    # Reiniciar X para que empiece en cero en enero de 2015
    x_2021 = x_2021 = np.array([mes.ordinal for mes in mensual_2021.index])

    modelo_2021 = sm.OLS(y_2021, sm.add_constant(x_2021)).fit()

    # Calcular valores predichos
    predicciones_2015 = modelo_2015.predict(sm.add_constant(x_2015))
    predicciones_2021 = modelo_2021.predict(sm.add_constant(x_2021))


    # Calcular bandas
    bandas_2015 = modelo_2015.get_prediction(sm.add_constant(x_2015)).summary_frame()
    bandas_2021 = modelo_2021.get_prediction(sm.add_constant(x_2021)).summary_frame()

    # Graficar

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(
        fechas,
        y,
        s=15,
        alpha=0.5,
        label=nombre_y
    )

    ax.plot(
        fechas_2015,
        predicciones_2015,
        color="red",
        label="Regresión: 2015 en adelante"
    )

    ax.fill_between(
        fechas_2015,
        bandas_2015["obs_ci_lower"],
        bandas_2015["obs_ci_upper"],
        color="red",
        alpha=0.2
    )

    ax.plot(
        fechas_2021,
        predicciones_2021,
        color="green",
        label="Regresión: 2021 en adelante"
    )

    ax.fill_between(
        fechas_2021,
        bandas_2021["obs_ci_lower"],
        bandas_2021["obs_ci_upper"],
        color="green",
        alpha=0.2
    )

    ax.set_title(titulo)
    ax.set_xlabel("Mes-año")
    ax.set_ylabel(nombre_y)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 12 meses siguientes en base a la regresion
    ultimo_mes = mensual.index.max()
    siguientes_meses = pd.period_range(
        start=ultimo_mes + 1,
        periods=12,
        freq="M"
    )

    x_siguientes = np.array([mes.ordinal for mes in siguientes_meses])

    pronostico_2015 = modelo_2015.predict(sm.add_constant(x_siguientes))
    pronostico_2021 = modelo_2021.predict(sm.add_constant(x_siguientes))

    pronosticos = pd.DataFrame({
        "Mes": siguientes_meses.astype(str),
        "Regresión desde 2015": np.round(pronostico_2015).astype(int),
        "Regresión desde 2021": np.round(pronostico_2021).astype(int)
    })

    print("\nPronóstico de juegos lanzados para los siguientes 12 meses:")
    print(pronosticos.to_string(index=False))



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