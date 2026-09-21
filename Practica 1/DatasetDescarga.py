import pandas as pd
import kagglehub
from tabulate import tabulate

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

def clean_owners(value):
    if pd.isna(value):
        return None

    # Split range: "100000000 - 200000000"
    values = str(value).split(" - ")

    if len(values) == 2:
        minimum = int(values[0])
        maximum = int(values[1])
        return int((minimum + maximum) / 2)

    return None

#Descargar el dataset original de kaggle (~400mb)
kagglehub.dataset_download('artermiloff/steam-games-dataset', path='games_march2025_cleaned.csv', output_dir='./dataset')

#Cargar el dataset en un dataframe y guardar solo ciertas columnas
df = pd.read_csv("./dataset/games_march2025_cleaned.csv")

columnas_a_usar = ['appid', 
'name', 
'release_date', 
'required_age', 
'price', 
'dlc_count', 
'short_description', 
'windows', 
'mac', 
'linux', 
'metacritic_score', 
'achievements', 
'recommendations', 
'developers', 
'publishers', 
'categories', 
'genres', 
'positive', 
'negative', 
'estimated_owners', 
'average_playtime_forever', 
'average_playtime_2weeks', 
'median_playtime_forever', 
'median_playtime_2weeks', 
'discount', 
'peak_ccu', 
'tags', 
'pct_pos_total', 
'num_reviews_total', 
'pct_pos_recent', 
'num_reviews_recent']

df = df[columnas_a_usar]

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

df["estimated_owners"] = df["estimated_owners"].apply(clean_owners)

print(df.head())

print(df.dtypes)

df.to_csv("./dataset/steam.csv", index=False)

#print_tabulate(df)