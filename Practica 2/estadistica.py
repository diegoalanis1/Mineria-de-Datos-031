import pandas as pd
from tabulate import tabulate 

#Cargar el dataset en un dataframe y guardar solo ciertas columnas
df = pd.read_csv("./Practica 1/dataset/steam.csv")

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

df["year_month"] = df["release_date"].dt.to_period("M")

# Agrupar por año-mes
stats_mensual = df.groupby("year_month").agg(
    total_juegos=("appid", "size"),
    precio_mensual=("price", "mean")
).reset_index()

print(stats_mensual.to_string(index=False))

stats_mensual.to_csv("./Practica 2/resultado.csv", index=False)