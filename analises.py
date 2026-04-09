import os
import pandas as pd

# ============================================================
# Config
# ============================================================

DATASET_DIR = "riot_dump/BR1/challenger/ranked_solo_420/dataset_ts_csv"

# ============================================================
# Carregar todos os CSV
# ============================================================

dfs = []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".csv"):
        path = os.path.join(DATASET_DIR, file)
        df = pd.read_csv(path)
        dfs.append(df)

dataset = pd.concat(dfs, ignore_index=True)

print("Dataset carregado")
print("Linhas:", len(dataset))
print("Colunas:", len(dataset.columns))
print()

# ============================================================
# 1. Número de partidas
# ============================================================

n_matches = dataset["matchId"].nunique()

print("Número de partidas:", n_matches)
print()

# ============================================================
# 2. Balanceamento das classes
# ============================================================

# 1 = blue win
# 0 = red win

matches_class = dataset.groupby("matchId")["y_blue_win"].first()

class_counts = matches_class.value_counts()

print("Balanceamento das classes")
print(class_counts)
print()

print("Proporção:")
print(class_counts / class_counts.sum())
print()

# ============================================================
# 3. Partidas por patch
# ============================================================

patch_counts = dataset.groupby("matchId")["patch"].first().value_counts().sort_index()

print("Partidas por patch")
print(patch_counts)
print()

# ============================================================
# 4. Campeões mais utilizados
# ============================================================

champion_cols = [c for c in dataset.columns if c.endswith("_championName")]

champions = []

# pegar apenas 1 linha por partida
matches_df = dataset.groupby("matchId").first().reset_index()

for col in champion_cols:
    champions.extend(matches_df[col].dropna().tolist())

champion_series = pd.Series(champions)

champion_counts = champion_series.value_counts()

print("Top 20 campeões mais utilizados:")
print(champion_counts.head(20))
print()

# ============================================================
# 5. Distribuição da duração das partidas
# ============================================================

duration = dataset.groupby("matchId")["t_min"].max()

print("Duração média das partidas (min):", duration.mean())
print("Duração mediana:", duration.median())
print("Duração máxima:", duration.max())
print("Duração mínima:", duration.min())
print()

# ============================================================
# 6. Frames por partida
# ============================================================

frames_per_match = dataset.groupby("matchId").size()

print("Frames médios por partida:", frames_per_match.mean())
print("Frames max:", frames_per_match.max())
print("Frames min:", frames_per_match.min())
print()

# ============================================================
# 7. Distribuição de kills por partida
# ============================================================

kills_blue = dataset.groupby("matchId")["kills_Blue"].max()
kills_red = dataset.groupby("matchId")["kills_Red"].max()

kills_total = kills_blue + kills_red

print("Kills médias por partida:", kills_total.mean())
print("Kills mediana:", kills_total.median())
print("Kills max:", kills_total.max())
print("Kills min:", kills_total.min())
print()

print("Análise concluída")